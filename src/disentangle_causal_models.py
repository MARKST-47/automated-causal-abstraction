import sys, os
sys.path.append(os.path.join('..', '..'))
from sklearn.metrics import classification_report
import torch
from pyvene import set_seed
import argparse
from causal_models import ArithmeticCausalModels, SimpleSummingCausalModels
from itertools import product
from utils import construct_arithmetic_input
from transformers import (GPT2Tokenizer,
                          GPT2Config,
                          GPT2ForSequenceClassification)
from pyvene import IntervenableModel
from my_pyvene.analyses.visualization import rotation_token_heatmap

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
    
def tokenizePrompt(input, tokenizer):
    prompt = f"{input['X']}+{input['Y']}+{input['Z']}="
    return tokenizer.encode(prompt, padding=True, return_tensors='pt')

def eval_one_point(intervenable, data, low_rank_dimension, min_class_value=3):
    eval_labels = []
    eval_preds = []

    with torch.no_grad():
        for input_key in ["base_source", "source_base"]:
            inputs = data[input_key]

            _, counterfactual_outputs = intervenable(
                {"input_ids": inputs["input_ids"]},
                [{"input_ids": inputs["source_input_ids"][:, 0]}],
                {"sources->base": [0, 1, 2, 3, 4, 5]},
                subspaces=[
                    [[_ for _ in range(low_rank_dimension)]]
                ]
            )

            eval_labels.append(inputs["labels"].type(torch.long).squeeze() - min_class_value)
            eval_preds.append(torch.argmax(counterfactual_outputs[0], dim=1).squeeze())

    report = classification_report(torch.tensor(eval_labels).cpu(), torch.tensor(eval_preds).cpu(), output_dict=True)
    return report['accuracy']

def main():

    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--model_path', type=str, help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--results_path', type=str, default='results/', help='path to the results folder')
    parser.add_argument('--causal_model_type', type=str, choices=['arithmetic', 'simple'], default='arithmetic', help='choose between arithmetic or simple')
    parser.add_argument('--low_rank_dim', type=int, default=256, help='low rank dimension for rotation intervention')
    parser.add_argument('--layer', type=int, default=0, help='layer on which to evaluate')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    parser.add_argument('--n_runs', type=int, default=1, help='number of runs before obtaining the graph')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise argparse.ArgumentTypeError("Invalid model_path. Path does not exist.")
    
    os.makedirs(args.results_path, exist_ok=True)

    save_plots_path = os.path.join(args.results_path, 'plots')
    os.makedirs(save_plots_path, exist_ok=True)

    args.results_path = os.path.join(args.results_path, args.causal_model_type)

    save_graphs_path = os.path.join(args.results_path, 'graphs')
    os.makedirs(save_graphs_path, exist_ok=True)
    
    set_seed(args.seed)
    
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

    low_rank_dimension = args.low_rank_dim
    numbers = range(1, 11)
    # numbers = range(1,4)
    repeat = 3
    graph_size = len(numbers) ** repeat
    arrangements = list(product(numbers, repeat=repeat))

    print('Tokenizing and caching...')

    tokenized_cache = {}
    for arrangement in arrangements:
        tokenized_cache[arrangement] = tokenizePrompt(construct_arithmetic_input(arrangement), tokenizer)

    print('Constructing the graphs..')
    # loop through the family of causal models
    for cm_id, model_info in arithmetic_family.causal_models.items():

        # to only get the graphs for targetting X or X+Y+Z
        if model_info['label'] == 'X+(Y)+Z' or model_info['label'] == 'X+Y+(Z)':
            continue

        print('loading intervenable model')
        intervenable_model_path = os.path.join(args.results_path, f'intervenable_models/cm_{cm_id}/intervenable_{low_rank_dimension}_{args.layer}')
        intervenable = IntervenableModel.load(intervenable_model_path, model=model)
        intervenable.set_device("cuda")
        intervenable.disable_model_gradients()

        graph_encoding = torch.zeros(graph_size, graph_size)

        print(f'..constructing graph {cm_id}..')

        for i, base in enumerate(arrangements):
            base = construct_arithmetic_input(base)
            for j, source in enumerate(arrangements[i + 1:]):
                source = construct_arithmetic_input(source)
                
                data = model_info['causal_model'].generate_counterfactual_pairs(
                    base,
                    source,
                    intervention_id,
                    device="cuda:0",
                    inputFunction=lambda x: tokenized_cache[tuple(x.values())]
                )
                
                iia = eval_one_point(intervenable, data, low_rank_dimension)
                graph_encoding[i][j] = iia
                graph_encoding[j][i] = iia

        graph_path = os.path.join(save_graphs_path, f'cm_{cm_id}')
        os.makedirs(graph_path, exist_ok=True)
        graph_path = os.path.join(graph_path, f'graph_{low_rank_dimension}_{args.layer}.pt')
        torch.save(graph_encoding, graph_path)

if __name__ =="__main__":
    main()