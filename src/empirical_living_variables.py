import sys, os
sys.path.append(os.path.join('..', '..'))

from sklearn.metrics import classification_report
from pyvene import set_seed
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from causal_models import SimpleArithmeticCausalModels
from utils import arithmetic_input_sampler, save_results, visualize_simple_per_token

from transformers import (GPT2Tokenizer,
                          GPT2Config,
                          GPT2ForSequenceClassification)

from pyvene import (
    IntervenableModel,
    RepresentationConfig,
    IntervenableConfig,
    VanillaIntervention
)

def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path)
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def intervention_id(intervention):
    if "P" in intervention:
        return 0

def tokenizePrompt(prompt):
    tokenizer = load_tokenizer("gpt2")
    prompt = f"{prompt['X']}+{prompt['Y']}+{prompt['Z']}="
    return tokenizer.encode(prompt, padding=True, return_tensors='pt')

def main():

    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--model_path', type=str, help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--results_path', type=str, default='results/', help='path to the results folder')
    parser.add_argument('--n_examples', type=int, default=256, help='number of training samples')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise argparse.ArgumentTypeError("Invalid model_path. Path does not exist.")
    
    os.makedirs(args.results_path, exist_ok=True)
    save_dir_path = os.path.join(args.results_path, 'plots')
    os.makedirs(save_dir_path, exist_ok=True)
    
    set_seed(args.seed)

    # fixed parameters
    min_class_value = 3 # summing 3 numbers from 1 to 10 results in values from 3 -..--> 30 => 28 labels in total => subtract 3 to get the predicted label

    # load the trained model
    tokenizer = load_tokenizer('gpt2')
    model_config = GPT2Config.from_pretrained(args.model_path)
    model_config.pad_token_id = tokenizer.pad_token_id # key in making it work?
    model = GPT2ForSequenceClassification.from_pretrained(args.model_path, config=model_config)
    model.resize_token_embeddings(len(tokenizer))

    # get different causal models
    simple_arithmetic_family = SimpleArithmeticCausalModels()

    for low_rank_dimension in [32, 64, 128, 256]:
    
        for id, model_info in simple_arithmetic_family.causal_models.items():

            causal_model = model_info['causal_model']

            # generate counterfactual data
            print('generating data for DAS...')

            counterfactual_data = causal_model.generate_counterfactual_dataset(
                args.n_examples,
                intervention_id,
                args.batch_size,
                device="cuda:0",
                sampler=arithmetic_input_sampler,
                inputFunction=tokenizePrompt
            )

            for token in [0,1,2,3,4,5]:
                for layer in range(model_config.n_layer):

                    # define intervention model
                    intervenable_config = IntervenableConfig(
                        model_type=type(model),
                        representations=[
                            RepresentationConfig(
                                layer,  # layer
                                "block_output",  # intervention type
                                "pos",  # intervention unit is now aligne with tokens; default though
                                1,  # max number of tokens to intervene on
                                # subspace_partition=None,  # binary partition with equal sizes
                                # intervention_link_key=0,
                            )
                        ],
                        intervention_types=VanillaIntervention,
                    )

                    intervenable = IntervenableModel(intervenable_config, model, use_fast=True)
                    intervenable.set_device("cuda")

                    for parameter in intervenable.get_trainable_parameters():
                        parameter.to("cuda:0")

                    # vanilla intervention happening
                    print('vanilla intervening on gpt2')

                    eval_labels = []
                    eval_preds = []
                    with torch.no_grad():
                        epoch_iterator = tqdm(DataLoader(counterfactual_data, args.batch_size), desc=f"Test")
                        for step, batch in enumerate(epoch_iterator):
                            for k, v in batch.items():
                                if v is not None and isinstance(v, torch.Tensor):
                                    batch[k] = v.to("cuda")
                            batch["input_ids"] = batch["input_ids"].squeeze()
                            batch["source_input_ids"] = batch["source_input_ids"].squeeze(2)

                            if batch["intervention_id"][0] == 0:

                                _, counterfactual_outputs = intervenable(
                                    {"input_ids": batch["input_ids"]}, # base
                                    [{"input_ids": batch["source_input_ids"][:, 0]}], # source, selecting all rows and only the values from the first column
                                    unit_locations={
                                        "sources->base": token
                                    },
                                    subspaces=[
                                        [[_ for _ in range(low_rank_dimension)]] * args.batch_size # taking half of the repr. and rotating it
                                    ]
                                )
                            
                            eval_labels += [batch["labels"].type(torch.long).squeeze() - min_class_value]
                            eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1)]
                    report = classification_report(torch.cat(eval_labels).cpu(), torch.cat(eval_preds).cpu(), output_dict=True) # get the IIA
                    save_results(args.results_path, report, layer, low_rank_dimension, token, id)

        for token in [0,1,2,3,4,5]:
            visualize_simple_per_token(args.results_path, save_dir_path, model_config.n_layer, token, low_rank_dimension, simple_arithmetic_family)

if __name__ =="__main__":
    main()