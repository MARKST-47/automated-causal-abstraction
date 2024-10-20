import sys, os
sys.path.append(os.path.join('..', '..'))
from sklearn.metrics import classification_report
import torch
from pyvene import set_seed
import argparse
from causal_models import DeMorgansLawCausalModels
from utils import generate_all_combinations_de_morgan, construct_de_morgan_input
from transformers import (AutoTokenizer,
                          GPT2Config,
                          GPT2ForSequenceClassification)
from torch.utils.data import DataLoader
from tqdm import tqdm

from pyvene import (
    IntervenableModel
)

def intervention_id(intervention):
    if "P" in intervention:
        return 0
    
def tokenizePrompt(prompt, tokenizer):
    prompt = f"{prompt['Op1']}({prompt['Op2']}({prompt['X']}) {prompt['B']} {prompt['Op3']}({prompt['Y']}))"
    return tokenizer.encode(prompt, return_tensors='pt')

def eval_one_point(intervenable, eval_data, low_rank_dimension, device, batch_size = 2):
    # eval on all data
    eval_labels = []
    eval_preds = []
    with torch.no_grad():
        epoch_iterator = tqdm(DataLoader(eval_data, batch_size), desc=f"Test")
        for _, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            inputs["input_ids"] = inputs["input_ids"].squeeze().long()
            inputs["source_input_ids"] = inputs["source_input_ids"].squeeze(2).long()

            _, counterfactual_outputs = intervenable(
                {"input_ids": inputs["input_ids"]},
                [{"input_ids": inputs["source_input_ids"][:, 0]}],
                {
                    "sources->base": [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
                },
                subspaces=[
                    [[_ for _ in range(low_rank_dimension)]] * batch_size
                ]
            )

            eval_labels += [inputs["labels"].type(torch.long).squeeze()]
            eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1)]
            
    report = classification_report(torch.cat(eval_labels).cpu(), torch.cat(eval_preds).cpu(), output_dict=True) # get the IIA
    return report['accuracy']

def main():

    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--model_path', type=str, default='mara589/binary-gpt2', help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--train_id', type=int, default=1, help='id of the model to train')
    parser.add_argument('--layer', type=int, default=8, help='layer on which to evaluate')
    parser.add_argument('--results_path', type=str, default='results/binary', help='path to the results folder')
    parser.add_argument('--low_rank_dimension', type=int, default=256, help='low rank dimension for rotation intervention')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') 

    os.makedirs(args.results_path, exist_ok=True)

    save_graphs_path = os.path.join(args.results_path, 'graphs')
    os.makedirs(save_graphs_path, exist_ok=True)
    
    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_config = GPT2Config.from_pretrained(args.model_path)
    model = GPT2ForSequenceClassification.from_pretrained(args.model_path, config=model_config)

    causal_model_family = DeMorgansLawCausalModels()
    train_id = args.train_id
    label = causal_model_family.get_label_by_id(train_id)
    causal_model = causal_model_family.get_model_by_id(train_id)

    all_comb = generate_all_combinations_de_morgan()

    tokenized_cache = {}
    for comb in all_comb:
        tokenized_cache[comb] = tokenizePrompt(construct_de_morgan_input(comb), tokenizer)

    print(f'loading intervenable model {label} on layer {args.layer}, lrd {args.low_rank_dimension}') 

    # intervenable_model_path = 'mara589/intervenable-models'
    # subfolder = f'{label}/intervenable_{args.low_rank_dimension}_{args.layer}'

    intervenable_model_path = os.path.join(args.results_path, f'intervenable_models/{label}/intervenable_{args.low_rank_dimension}_{args.layer}')
    intervenable = IntervenableModel.load(intervenable_model_path, model=model)
    
    # intervenable = IntervenableModel.load(intervenable_model_path, model=model, subfolder=subfolder)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()

    graph_size = len(all_comb)

    graph_encoding = torch.zeros(graph_size, graph_size)

    print(f'..constructing graph {label} on layer {args.layer}, lrd {args.low_rank_dimension}..')

    for i, base in enumerate(all_comb):
        base = construct_de_morgan_input(base)
        print(base)
        for j, source in enumerate(all_comb):

            if i <= j:
                break

            source = construct_de_morgan_input(source)

            print(source)
            return
            
            data = causal_model.generate_counterfactual_pairs(
                base,
                source,
                intervention_id,
                device=device,
                inputFunction=lambda x: tokenized_cache[tuple(x.values())]
            )
            
            iia = eval_one_point(intervenable, list(data.values()), args.low_rank_dimension, device)
            
            graph_encoding[i][j] = iia
            graph_encoding[j][i] = iia

    graph_path = os.path.join(save_graphs_path, f'{label}_graph_{args.low_rank_dimension}_{args.layer}.pt')
    torch.save(graph_encoding, graph_path)

if __name__ =="__main__":
    main()