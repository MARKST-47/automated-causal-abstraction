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
from causal_models import ArithmeticCausalModels, SimpleSummingCausalModels
import numpy as np
import json
import matplotlib.pyplot as plt
from itertools import product
from utils import arithmetic_input_sampler, construct_arithmetic_input

from transformers import (GPT2Tokenizer,
                          GPT2Config,
                          GPT2ForSequenceClassification)

from pyvene import (
    IntervenableModel,
    IntervenableConfig,
    LowRankRotatedSpaceIntervention
)

from my_pyvene.analyses.visualization import rotation_token_heatmap

# temporary import
# from my_pyvene.models.intervenable_base import IntervenableModel
# from my_pyvene.models.configuration_intervenable_model import IntervenableConfig
# from my_pyvene.models.interventions import LowRankRotatedSpaceIntervention

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
    return report['accuracy']

def eval_one_point(intervenable, data, low_rank_dimension, min_class_value=3):

    base_source_inputs = data['base_source']
    source_base_inputs = data['source_base']

    eval_labels = []
    eval_preds = []
    with torch.no_grad():
        # for k, v in base_source_inputs.items():
        #     if v is not None and isinstance(v, torch.Tensor):
        #         base_source_inputs[k] = v.to("cuda:0")

        # for k, v in source_base_inputs.items():
        #     if v is not None and isinstance(v, torch.Tensor):
        #         source_base_inputs[k] = v.to("cuda:0")

        # base_source_inputs["source_input_ids"] = base_source_inputs["source_input_ids"].squeeze(1)
        # b_s = inputs["input_ids"].shape[0]
        _, counterfactual_outputs = intervenable(
            {"input_ids": base_source_inputs["input_ids"]},
            [{"input_ids": base_source_inputs["source_input_ids"][:, 0]}],
            {"sources->base": [0,1,2,3,4,5]},
            subspaces=[
                [[_ for _ in range(low_rank_dimension)]] # * b_s
            ]
        )

        eval_labels += [base_source_inputs["labels"].type(torch.long).squeeze() - min_class_value]
        eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1).squeeze()]

        # source_base_inputs["source_input_ids"] = source_base_inputs["source_input_ids"].squeeze(1)

        _, counterfactual_outputs = intervenable(
            {"input_ids": source_base_inputs["input_ids"]},
            [{"input_ids": source_base_inputs["source_input_ids"][:, 0]}],
            {"sources->base": [0,1,2,3,4,5]},
            subspaces=[
                [[_ for _ in range(low_rank_dimension)]] # * b_s
            ]
        )

        eval_labels += [source_base_inputs["labels"].type(torch.long).squeeze() - min_class_value]
        eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1).squeeze()]
    
    report = classification_report(torch.tensor(eval_labels).cpu(), torch.tensor(eval_preds).cpu(), output_dict=True) # get the IIA
    return report['accuracy']

def decode_tensor(tensor, vocabulary):
    token_ids = tensor.squeeze().tolist()
    return {'X': int(vocabulary[token_ids[0]]), 'Y': int(vocabulary[token_ids[2]]), 'Z': int(vocabulary[token_ids[4]])}

def main():

    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--model_path', type=str, help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--results_path', type=str, default='disentangling_results/', help='path to the results folder')
    parser.add_argument('--causal_model_type', type=str, choices=['arithmetic', 'simple'], default='arithmetic', help='choose between arithmetic or simple')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    parser.add_argument('--low_rank_dim', type=int, default=256, help='low rank dimension for rotation intervention')
    parser.add_argument('--n_runs', type=int, default=1, help='number of runs before obtaining the graph')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise argparse.ArgumentTypeError("Invalid model_path. Path does not exist.")
    
    os.makedirs(args.results_path, exist_ok=True)

    save_plots_path = os.path.join(args.results_path, 'plots')
    os.makedirs(save_plots_path, exist_ok=True)

    save_graphs_path = os.path.join(args.results_path, 'graphs')
    os.makedirs(save_graphs_path, exist_ok=True)

    args.results_path = os.path.join(args.results_path, args.causal_model_type)
    
    set_seed(args.seed)
    
    min_class_value = 3
    n_arrangements = 1000
    
    tokenizer = load_tokenizer('gpt2')
    # hard-coded vocabulary
    vocabulary = {'1': 16, '2': 17, '3': 18, '4': 19, '5': 20, '6': 21, '7': 22, '8': 23, '9': 24, '10': 940, '+': 10, '=': 28}
    inv_vocabulary = {v: k for k, v in vocabulary.items()}
    # vocabulary = tokenizer.get_vocab()
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

    G = {}

    low_rank_dimension = args.low_rank_dim
    numbers = range(1, 11)
    repeat = 3
    graph_size = len(numbers) ** repeat
    
    # loop through the family of causal models
    for cm_id, model_info in arithmetic_family.causal_models.items():

        G[cm_id] = {}

        for layer in range(model_config.n_layer):
        
            intervenable_model_path = os.path.join(args.results_path, f'intervenable_models/cm_{cm_id}/intervenable_{low_rank_dimension}_{layer}')
            intervenable = IntervenableModel.load(intervenable_model_path, model=model)
            intervenable.set_device("cuda")
            intervenable.disable_model_gradients()
            intervenable.model.eval()

            numbers = range(1, 3)
            arrangements_x = product(numbers, repeat=repeat)

            # initialize graph weighted by accuracies
            graph_encoding = torch.zeros(graph_size, graph_size)

            for i, base in enumerate(arrangements_x):
                base = construct_arithmetic_input(base)
                arrangements_y = product(numbers, repeat=repeat)
                for j, source in enumerate(arrangements_y):
                    
                    if i <= j:
                        continue
                    
                    source = construct_arithmetic_input(source)
                    data = model_info['causal_model'].generate_fixed_counterfactuals(
                        base,
                        source,
                        intervention_id,
                        device="cuda:0",
                        inputFunction=tokenizePrompt
                    )

                    iia = eval_one_point(intervenable, data, low_rank_dimension)
                    graph_encoding[i][j] = iia
                    graph_encoding[j][i] = iia
                    
            G[cm_id][layer] = graph_encoding

    best_graphs = {}
    for layer in range(model_config.n_layer):
        best_graphs[layer] = torch.zeros(graph_size, graph_size)

        for i in range(graph_size):
            for j in range(graph_size):
                    best_acc = 0
                    best_model = 0
                    for id, cm_accs in G.items():
                        if cm_accs[layer][i][j] > best_acc:
                            best_acc = cm_accs[layer][i][j]
                            best_model = id

                    best_graphs[layer][i][j] = best_model
    
        # save graph
        graph_path = os.path.join(save_graphs_path, f'graph_{layer}.pt')
        torch.save(best_graphs[layer], graph_path)

    print(best_graphs)

if __name__ =="__main__":
    main()