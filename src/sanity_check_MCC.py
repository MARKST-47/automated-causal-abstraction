import sys, os
sys.path.append(os.path.join('..', '..'))

from sklearn.metrics import classification_report
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import random
from tqdm import tqdm, trange
from pyvene import count_parameters, set_seed
import argparse
from causal_models import ArithmeticCausalModels
import numpy as np
from utils import arithmetic_input_sampler, visualize_graph

from transformers import (GPT2Tokenizer,
                          GPT2Config,
                          GPT2ForSequenceClassification)

# from pyvene import (
#     IntervenableModel,
#     IntervenableConfig,
#     LowRankRotatedSpaceIntervention
# )

# temporary import
from my_pyvene.models.intervenable_base import IntervenableModel
from my_pyvene.models.configuration_intervenable_model import IntervenableConfig
from my_pyvene.models.interventions import LowRankRotatedSpaceIntervention

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
    
def tokenizePrompt(input):
    tokenizer = load_tokenizer("gpt2")
    prompt = f"{input['X']}+{input['Y']}+{input['Z']}="
    return tokenizer.encode(prompt, padding=True, return_tensors='pt')

def main():

    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--model_path', type=str, help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--results_path', type=str, default='results/', help='path to the results folder')
    parser.add_argument('--n_training', type=int, default=2560, help='number of training samples')
    parser.add_argument('--n_testing', type=int, default=256, help='number of testing samples')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='number of steps to accumulate before optimization step')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise argparse.ArgumentTypeError("Invalid model_path. Path does not exist.")
    
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

    arithmetic_family = ArithmeticCausalModels()
    bases = []
    for _ in range(args.n_training):
        bases.append(arithmetic_input_sampler())

    test_bases = []
    for _ in range(args.n_testing):
        test_bases.append(arithmetic_input_sampler())

    model_accs = {}
    
    for cm_id, model_info in arithmetic_family.causal_models.items():

        print('generating data for DAS...')

        training_counterfactual_data = model_info['causal_model'].generate_counterfactual_dataset(
            args.n_training,
            intervention_id,
            args.batch_size,
            device="cuda:0",
            sampler=arithmetic_input_sampler,
            inputFunction=tokenizePrompt
        )

        # testing_counterfactual_data = model_info['causal_model'].generate_counterfactual_dataset_on_bases(
        #     args.n_testing,
        #     intervention_id,
        #     args.batch_size,
        #     test_bases,
        #     device="cuda:0",
        #     sampler=arithmetic_input_sampler,
        #     inputFunction=tokenizePrompt
        # )

        layer = 0
        low_rank_dimension = 64

        intervenable_config = IntervenableConfig({
                "layer": layer,
                "component": "block_output",
                "low_rank_dimension": low_rank_dimension,
                "unit":"pos",
                "max_number_of_units": 6
            },
            intervention_types=LowRankRotatedSpaceIntervention,
            model_type=type(model)
        )

        G = []

        for _ in range(10):

            graph_encoding = {}

            for i, x in enumerate(training_counterfactual_data):
                for j, y in enumerate(training_counterfactual_data):

                    if i == j:
                        continue

                    train_data = np.array([x,y])

                    intervenable = IntervenableModel(intervenable_config, model, use_fast=True)
                    intervenable.set_device("cuda")
                    intervenable.disable_model_gradients()

                    optimizer_params = []
                    for k, v in intervenable.interventions.items():
                        optimizer_params += [{"params": v[0].rotate_layer.parameters()}]

                    optimizer = torch.optim.Adam(optimizer_params, lr=0.01)

                    print('DAS training...')

                    intervenable.model.train()
                    print("intervention trainable parameters: ", intervenable.count_parameters())
                    print("gpt2 trainable parameters: ", count_parameters(intervenable.model))
                    train_iterator = trange(0, int(args.epochs), desc="Epoch")

                    # training with two examples
                    for epoch in train_iterator:
                        torch.cuda.empty_cache()
                        epoch_iterator = tqdm(
                            DataLoader(
                                train_data,
                                batch_size=args.batch_size,
                                sampler=batched_random_sampler(train_data, args.batch_size),
                            ),
                            desc=f"Epoch: {epoch}",
                            position=0,
                            leave=True,
                        )
                        
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
                                    [[_ for _ in range(low_rank_dimension)]] * args.batch_size # taking half of the repr. and rotating it
                                ]
                            )

                            eval_metrics = compute_metrics(
                                counterfactual_outputs[0].argmax(1), inputs["labels"].squeeze() - min_class_value
                            )

                            # loss and backprop
                            loss = calculate_loss(counterfactual_outputs.logits, inputs["labels"] - min_class_value)
                            loss_str = round(loss.item(), 2)
                            epoch_iterator.set_postfix({"loss": loss_str, "acc": eval_metrics["accuracy"]})

                            if args.gradient_accumulation_steps > 1:
                                loss = loss / args.gradient_accumulation_steps
                            loss.backward()

                            if total_step % args.gradient_accumulation_steps == 0:
                                optimizer.step()
                                intervenable.set_zero_grad()
                            total_step += 1

                # generate testing counterfactual data
                print('testing...')

                best_acc = 0
                best_model = 0

                for test_id, test_model_info in arithmetic_family.causal_models.items():

                    if test_id not in graph_encoding:
                        graph_encoding[test_id] = torch.zeros(args.n_training, args.n_training)

                    testing_counterfactual_data = test_model_info['causal_model'].generate_counterfactual_dataset(
                        args.n_testing,
                        intervention_id,
                        args.batch_size,
                        device="cuda:0",
                        sampler=arithmetic_input_sampler,
                        inputFunction=tokenizePrompt
                    )

                    eval_labels = []
                    eval_preds = []
                    with torch.no_grad():
                        epoch_iterator = tqdm(DataLoader(testing_counterfactual_data, args.batch_size), desc=f"Test")
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
                                    [[_ for _ in range(low_rank_dimension)]] * args.batch_size
                                ]
                            )

                            eval_labels += [inputs["labels"].type(torch.long).squeeze() - min_class_value]
                            eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1)]
                    report = classification_report(torch.cat(eval_labels).cpu(), torch.cat(eval_preds).cpu(), output_dict=True) # get the IIA

                    # if report['accuracy'] > best_acc:
                    #     best_acc = report['accuracy']
                    #     best_model = test_id
                    graph_encoding[test_id][i][j] = report['accuracy']
            G.append(graph_encoding)

        model_accs = {}
        final_vars = {}
        for cm_id, model_info in arithmetic_family.causal_models.items():
            summing = []
            for m in G:
                summing.append(m[cm_id])
            stacked_tensors = torch.stack(summing)
            average_tensor = stacked_tensors.mean(dim=0)
            variance_tensor = stacked_tensors.var(dim=0)

            model_accs[cm_id] = average_tensor
            final_vars[cm_id] = variance_tensor

        print()
        print(model_accs)
        print()
        print(final_vars)

        graph_1 = torch.zeros(args.n_training, args.n_training)
        graph_2 = torch.zeros(args.n_training, args.n_training)

        for i in range(args.n_training):
            for j in range(args.n_training):
                if i > j:
                    best_acc = 0
                    best_model = 0
                    for id, cm_accs in model_accs.items():

                        if cm_accs[i][j] > best_acc:
                            best_acc = cm_accs[i][j]
                            best_model = id

                    graph_1[i][j] = best_model
                
                if i < j:
                    best_acc = 0
                    best_model = 0
                    for id, cm_accs in model_accs.items():

                        if cm_accs[i][j] > best_acc:
                            best_acc = cm_accs[i][j]
                            best_model = id

                    graph_2[i][j] = best_model

        tag = model_info['label']
        label = f'lower_triangle_{tag}'
        print(graph_1)
        visualize_graph(graph_1, label)

        label = f'upper_triangle_{tag}'
        visualize_graph(graph_2, label)

if __name__ =="__main__":
    main()