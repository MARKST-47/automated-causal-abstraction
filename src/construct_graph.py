import sys, os
sys.path.append(os.path.join('..', '..'))
from sklearn.metrics import classification_report
import torch
from pyvene import set_seed
import argparse
from causal_models import ArithmeticCausalModels, SimpleSummingCausalModels
from itertools import product
from utils import construct_arithmetic_input
from transformers import (AutoTokenizer,
                          GPT2Config,
                          GPT2ForSequenceClassification)
from my_pyvene.models.intervenable_base import IntervenableModel
from torch.utils.data import DataLoader
from tqdm import tqdm

def intervention_id(intervention):
    if "P" in intervention:
        return 0
    
def tokenizePrompt(input, tokenizer):
    prompt = f"{input['X']}+{input['Y']}+{input['Z']}="
    return tokenizer.encode(prompt, padding=True, return_tensors='pt')

def eval_one_point(intervenable, eval_data, low_rank_dimension, batch_size = 2, min_class_value=3):
    # eval on all data
    eval_labels = []
    eval_preds = []
    with torch.no_grad():
        epoch_iterator = tqdm(DataLoader(eval_data, batch_size), desc=f"Test")
        for _, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")
            inputs["input_ids"] = inputs["input_ids"].squeeze()
            inputs["source_input_ids"] = inputs["source_input_ids"].squeeze(2)

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

def main():

    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--model_path', type=str, default='mara589/arithmetic-gpt2', help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--cm_id', type=int, choices=[1,2,3,4,5,6,7], default='1', help='choose which causal model you want to construct the graph on')
    parser.add_argument('--layer', type=int, default=0, help='layer on which to evaluate')
    parser.add_argument('--results_path', type=str, default='results/', help='path to the results folder')
    parser.add_argument('--low_rank_dimension', type=int, default=256, help='low rank dimension for rotation intervention')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    os.makedirs(args.results_path, exist_ok=True)

    save_graphs_path = os.path.join(args.results_path, 'graphs')
    os.makedirs(save_graphs_path, exist_ok=True)
    
    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_config = GPT2Config.from_pretrained(args.model_path)
    model = GPT2ForSequenceClassification.from_pretrained(args.model_path, config=model_config)

    numbers = range(1, 11)
    repeat = 3
    graph_size = len(numbers) ** repeat
    arrangements = list(product(numbers, repeat=repeat))

    print('Tokenizing and caching...')

    tokenized_cache = {}
    for arrangement in arrangements:
        tokenized_cache[arrangement] = tokenizePrompt(construct_arithmetic_input(arrangement), tokenizer)

    arithmetic_family = ArithmeticCausalModels()
    simple_family = SimpleSummingCausalModels()

    # merge all causal models solving the summing of three numbers task
    for cm_id, model_info in simple_family.causal_models.items():
        arithmetic_family.add_model(model_info['causal_model'], model_info['label'])
    
    causal_model = arithmetic_family.get_model_by_id(args.cm_id)
    label = arithmetic_family.get_label_by_id(args.cm_id)

    print(f'loading intervenable model {label} on layer {args.layer}, lrd {args.low_rank_dimension}') 
    intervenable_model_path = 'mara589/intervenable-models'
    subfolder = f'M_{args.cm_id}/intervenable_{args.low_rank_dimension}_{args.layer}'
    
    intervenable = IntervenableModel.load(intervenable_model_path, model=model, subfolder=subfolder)
    intervenable.set_device("cuda")
    intervenable.disable_model_gradients()

    graph_encoding = torch.zeros(graph_size, graph_size)

    print(f'..constructing graph {label} on layer {args.layer}, lrd {args.low_rank_dimension}..')

    for i, base in enumerate(arrangements):
        base = construct_arithmetic_input(base)
        for j, source in enumerate(arrangements):

            if i <= j:
                break

            source = construct_arithmetic_input(source)
            
            data = causal_model.generate_counterfactual_pairs(
                base,
                source,
                intervention_id,
                device="cuda:0",
                inputFunction=lambda x: tokenized_cache[tuple(x.values())]
            )
            
            iia = eval_one_point(intervenable, list(data.values()), args.low_rank_dimension)
            
            graph_encoding[i][j] = iia
            graph_encoding[j][i] = iia

    graph_path = os.path.join(save_graphs_path, f'{label}_graph_{args.low_rank_dimension}_{args.layer}.pt')
    torch.save(graph_encoding, graph_path)

if __name__ =="__main__":
    main()