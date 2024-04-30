import sys, os
sys.path.append(os.path.join('..', '..'))

import argparse
from pyvene import set_seed
import torch
import numpy as np
from causal_models import ArithmeticCausalModels
import matplotlib.pyplot as plt
import networkx as nx
import itertools
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from utils import arithmetic_input_sampler

from transformers import (GPT2Tokenizer,
                          GPT2Config,
                          GPT2ForSequenceClassification)

from my_pyvene.models.intervenable_base import IntervenableModel

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

def main():
    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--model_path', type=str, help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--results_path', type=str, default='results/', help='path to the results folder')
    parser.add_argument('--low_rank_dim', type=int, default=64, help='low rank dimension for rotation intervention')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise argparse.ArgumentTypeError("Invalid model_path. Path does not exist.")
    
    if not os.path.exists(args.results_path):
        raise argparse.ArgumentTypeError("Invalid results_path. Path does not exist.")

    # load labelled graph
    graph_path = os.path.join(args.results_path, 'graph.pt')
    graph = torch.load(graph_path)

    # load subset of bases
    subset_bases_path = os.path.join(args.results_path, 'testing_bases.npy')
    T = np.load(subset_bases_path, allow_pickle=True)

    # load iia_s, the iia on the whole subset data T
    iias_path = os.path.join(args.results_path, 'iia_s.json')
    with open(iias_path, 'r') as f:
        iia_s = json.load(f)

    D = []
    low_rank_dimension = args.low_rank_dim
    
    set_seed(args.seed)

    tokenizer = load_tokenizer('gpt2')
    model_config = GPT2Config.from_pretrained(args.model_path)
    model_config.pad_token_id = tokenizer.pad_token_id
    model = GPT2ForSequenceClassification.from_pretrained(args.model_path, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    
    arithmetic_family = ArithmeticCausalModels()

    iia_d = {}

    for cm_id, model_info in arithmetic_family.causal_models.items():
        best_combo_path = os.path.join(args.results_path, f'class_data_{cm_id}.npy')
        data_ids = np.load(best_combo_path, allow_pickle=True)
        D = [T[index] for index in data_ids]
        intervenable_path = os.path.join(args.results_path, f'intervenable_{cm_id}/')
        
        intervenable = IntervenableModel.load(intervenable_path, model)
        intervenable.set_device("cuda")

        if len(D) % 2 == 0:
            testing_batch_size = 2
        else:
            testing_batch_size = len(D)
        n_testing = len(D)

        clique_counterfactual_data = model_info['causal_model'].generate_counterfactual_dataset_on_bases(
            n_testing,
            intervention_id,
            testing_batch_size,
            D,
            device="cuda:0",
            sampler=arithmetic_input_sampler,
            inputFunction=tokenizePrompt
        )

        iia = eval_intervenable(intervenable, clique_counterfactual_data, testing_batch_size, low_rank_dimension)
        iia_d[cm_id] = iia
        print(f'IIA on entire subset is: {iia_s[str(cm_id)]}')
        print(f'IIA on clique is: {iia}')

    data = {
        'Clique Data': iia_d, 
        'All Data': iia_s
    }

    classes = list(data['Clique Data'].keys())
    situation_names = list(data.keys())

    width = 0.35

    for i, situation_name in enumerate(situation_names):
        accuracies = list(data[situation_name].values())
        x_positions = [x + i * width for x in classes]
        plt.bar(x_positions, accuracies, width, label=situation_name)

    plt.xlabel('Class')
    plt.ylabel('IIA')
    plt.title('IIA comparison when evaluation on all data and only on clique data')
    plt.xticks(classes)
    plt.legend()
    plt.savefig('compared_IIA.png')
    plt.close()

if __name__ =="__main__":
    main()