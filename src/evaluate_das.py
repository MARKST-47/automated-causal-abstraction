import sys, os
sys.path.append(os.path.join('..', '..'))

from sklearn.metrics import classification_report
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import random
from tqdm import tqdm, trange
from pyvene import set_seed
import argparse
from itertools import product
from causal_models import ArithmeticCausalModels, SimpleSummingCausalModels
from utils import arithmetic_input_sampler, save_results, construct_arithmetic_input, ruled_arithmetic_input_sampler

from transformers import (GPT2Tokenizer,
                          GPT2Config,
                          GPT2ForSequenceClassification)

from pyvene import (
    IntervenableModel
)


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path)
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def batched_random_sampler(data, batch_size):
    batch_indices = [_ for _ in range(int(len(data) / batch_size))]
    random.shuffle(batch_indices)
    for b_i in batch_indices:
        for i in range(b_i * batch_size, (b_i + 1) * batch_size):
            yield i

def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        total_count += 1
        correct_count += eval_pred == eval_label
    accuracy = float(correct_count) / float(total_count)
    return {"accuracy": accuracy}
    
def calculate_loss(logits, labels):
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, 28) # 28 is the number of classes
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device).long()
    loss = loss_fct(shift_logits, shift_labels)

    return loss

def intervention_id(intervention):
    if "P" in intervention:
        return 0
    
def tokenizePrompt(input, tokenizer):
    prompt = f"{input['X']}+{input['Y']}+{input['Z']}="
    return tokenizer.encode(prompt, padding=True, return_tensors='pt')

def eval_intervenable(intervenable, eval_data, batch_size, low_rank_dimension, min_class_value=3):
    # eval on all data
    eval_labels = []
    eval_preds = []
    with torch.no_grad():
        epoch_iterator = tqdm(DataLoader(eval_data, batch_size), desc=f"Test")
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")
            inputs["input_ids"] = inputs["input_ids"].squeeze()
            inputs["source_input_ids"] = inputs["source_input_ids"].squeeze(2)
            b_s = inputs["input_ids"].shape[0]
            _, counterfactual_outputs = intervenable(
                {"input_ids": inputs["input_ids"]},
                [{"input_ids": inputs["source_input_ids"][:, 0]}],
                {"sources->base": [0,1,2,3,4,5]},
                subspaces=[
                    [[_ for _ in range(low_rank_dimension)]] * batch_size
                ]
            )

            eval_labels += [inputs["labels"].type(torch.long).squeeze() - min_class_value]
            eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1)]
    report = classification_report(torch.cat(eval_labels).cpu(), torch.cat(eval_preds).cpu(), output_dict=True) # get the IIA
    return report

def main():

    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--model_path', type=str, help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--causal_model_type', type=str, choices=['arithmetic', 'simple'], default='arithmetic', help='choose between arithmetic or simple')
    parser.add_argument('--results_path', type=str, default='results/', help='path to the results folder')
    parser.add_argument('--n_testing', type=int, default=256, help='number of testing samples')
    parser.add_argument('--experiment', type=str, choices=['', 'original', 'ruled'], default='')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise argparse.ArgumentTypeError("Invalid model_path. Path does not exist.")

    os.makedirs(args.results_path, exist_ok=True)

    args.results_path = os.path.join(args.results_path, args.causal_model_type)
    os.makedirs(args.results_path, exist_ok=True)

    save_dir_path = os.path.join(args.results_path, 'plots')
    os.makedirs(save_dir_path, exist_ok=True)
    
    set_seed(args.seed)
    total_step = 0
    min_class_value = 3
    
    tokenizer = load_tokenizer('gpt2')
    model_config = GPT2Config.from_pretrained(args.model_path)
    model_config.pad_token_id = tokenizer.pad_token_id
    model = GPT2ForSequenceClassification.from_pretrained(args.model_path, config=model_config)
    model.resize_token_embeddings(len(tokenizer))

    if args.causal_model_type == 'arithmetic':
        arithmetic_family = ArithmeticCausalModels()
    elif args.causal_model_type == 'simple':
        arithmetic_family = SimpleSummingCausalModels()
    else:
        raise ValueError(f"Invalid causal model type: {args.causal_model_type}. Can only choose between arithmetic or simple.")
    
    numbers = range(1, 11)
    repeat = 3
    graph_size = len(numbers) ** repeat
    arrangements = list(product(numbers, repeat=repeat))
    tokenized_cache = {}
    for arrangement in arrangements:
        tokenized_cache[arrangement] = tokenizePrompt(construct_arithmetic_input(arrangement), tokenizer)

    if args.experiment == 'original' or args.experiment == '':
        sampler = arithmetic_input_sampler
    elif args.experiment == 'ruled':
        sampler = ruled_arithmetic_input_sampler
    else:
        raise ValueError(f"Invalid causal model type: {args.experiment}. Can only choose between arithmetic or simple.")

    for low_rank_dimension in [4,8,16,32]:
        for layer in range(model_config.n_layer):
        # for layer in [0,1,2,3,4,5,6,7,8,9]:
        # for layer in [10,11]:

            for cm_id, _ in arithmetic_family.causal_models.items():
                if cm_id == 2 or cm_id == 3:
                    continue

                intervenable_model_path = os.path.join(args.results_path, f'intervenable_models/cm_{cm_id}/intervenable_{low_rank_dimension}_{layer}')
                intervenable = IntervenableModel.load(intervenable_model_path, model=model)
                intervenable.set_device("cuda")
                intervenable.disable_model_gradients()

                for test_id, test_model_info in arithmetic_family.causal_models.items():

                    if args.causal_model_type == 'simple' or args.experiment == 'original' or args.experiment == 'ruled':
                        if test_id != cm_id:
                            continue

                    testing_counterfactual_data = test_model_info['causal_model'].generate_counterfactual_dataset(
                        args.n_testing,
                        intervention_id,
                        args.batch_size,
                        device="cuda:0",
                        sampler=sampler,
                        inputFunction=lambda x: tokenized_cache[tuple(x.values())]
                    )

                    report = eval_intervenable(intervenable, testing_counterfactual_data, args.batch_size, low_rank_dimension)
                    save_results(args.results_path, report, layer, low_rank_dimension, cm_id, test_id)
        
if __name__ =="__main__":
    main()