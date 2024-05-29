import os
from causal_models import ArithmeticCausalModels, SimpleSummingCausalModels
import argparse
from pyvene import set_seed
from utils import sanity_check_visualization, empirical_visualization
from transformers import (GPT2Config)

def main():

    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--model_path', type=str, help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--results_path', type=str, default='disentangling_results/', help='path to the results folder')
    parser.add_argument('--causal_model_type', type=str, choices=['arithmetic', 'simple', 'show_causal_models'], default='arithmetic', help='choose between arithmetic or simple')
    parser.add_argument('--experiment', type=str, choices=['sanity_check', 'empirical'])
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    parser.add_argument('--low_rank_dim', type=int, default=256, help='low rank dimension for rotation intervention')
    # parser.add_argument('--n_runs', type=int, default=1, help='number of runs before obtaining the graph')
    args = parser.parse_args()

    save_dir_path = os.path.join(args.results_path, 'plots_2')
    os.makedirs(save_dir_path, exist_ok=True)

    args.results_path = os.path.join(args.results_path, args.causal_model_type)
    if not os.path.exists(args.results_path):
        raise argparse.ArgumentTypeError("Invalid results_path. You also need to have your intervenable models trained and saved. Refer to README.md file.")

    set_seed(args.seed)

    if not os.path.exists(args.model_path):
        raise argparse.ArgumentTypeError("Invalid model_path. Path does not exist.")
    
    model_config = GPT2Config.from_pretrained(args.model_path)

    if args.causal_model_type == 'arithmetic':
        arithmetic_family = ArithmeticCausalModels()
    elif args.causal_model_type == 'simple':
        arithmetic_family = SimpleSummingCausalModels()
    else:
        raise ValueError(f"Invalid causal model type: {args.causal_model_type}. Can only choose between arithmetic or simple.")

    if args.experiment == 'show_causal_models':
        for cm_id, model_info in arithmetic_family.causal_models.items():
            model_info['causal_model'].print_structure(fig_name=model_info['label'])
    elif args.experiment == 'sanity_check':
        for cm_id, model_info in arithmetic_family.causal_models.items():
            for experiment_id in [64, 128, 256, 768, 4608]:
                sanity_check_visualization(args.results_path, save_dir_path, model_config.n_layer, cm_id, experiment_id, arithmetic_family)
    elif args.experiment == 'empirical':
        for cm_id, model_info in arithmetic_family.causal_models.items():
            for experiment_id in [64, 128, 256, 768, 4608]:
                empirical_visualization(args.results_path, save_dir_path, model_config.n_layer, cm_id, experiment_id, model_info['label'])

if __name__ =="__main__":
    main()