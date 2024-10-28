import sys, os
sys.path.append(os.path.join('..', '..'))
from sklearn.metrics import classification_report
import torch
from pyvene import set_seed
import argparse
from causal_models import DeMorgansLawCausalModels
from utils import binary_evaluation_visualization
from transformers import (AutoTokenizer,
                          GPT2Config,
                          GPT2ForSequenceClassification)
import json
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    # parser.add_argument('--model_path', type=str, default='mara589/binary-gpt2', help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--results_path', type=str, default='results/binary', help='path to the results folder')
    # parser.add_argument('--low_rank_dimension', type=int, default=128, help='low rank dimension for rotation intervention')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    os.makedirs(args.results_path, exist_ok=True)

    save_dir_path = os.path.join(args.results_path, 'plots')
    os.makedirs(save_dir_path, exist_ok=True)

    set_seed(args.seed)
    
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # model_config = GPT2Config.from_pretrained(args.model_path)
    # model = GPT2ForSequenceClassification.from_pretrained(args.model_path, config=model_config)

    causal_model_family = DeMorgansLawCausalModels()
    n_layers = 3
    _, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10(range(13)) 

    lrd = 128

    for cm_id, model_info in causal_model_family.causal_models.items():
        label = model_info['label']
        accuracies = []
        for layer in range(n_layers):
            file_path = os.path.join(args.results_path, f'results_{label}/{label}_report_layer_{layer}_tkn_128.json')

            with open(file_path, 'r') as json_file:
                accuracies.append(json.load(json_file)['accuracy'])
        
        ax.plot(range(n_layers), accuracies, marker='o', linestyle='-', 
                linewidth=1.5, color=colors[cm_id - 1], label=label, alpha=0.8)
        
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("IIA", fontsize=10)
    ax.set_xticks(range(n_layers))
    ax.set_xlim([-0.5, n_layers - 0.5])
    ax.grid(axis='y', linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=10)

    save_file_name = f'binary_IIA_per_layer_{lrd}.pdf'
    # save_file_name = f'solving_arithmetic_task_{experiment_id}.pdf'
    file_path = os.path.join(save_dir_path, save_file_name)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    
if __name__ =="__main__":
    main()