import os
from causal_models import ArithmeticCausalModels, SimpleSummingCausalModels
import argparse
from pyvene import set_seed
from utils import sanity_check_visualization, empirical_visualization, evaluation_visualization, compare_intermediate_vs_simple, evaluation_visualization_combined
from transformers import (AutoTokenizer,
                          GPT2Tokenizer,
                          GPT2Config,
                          GPT2ForSequenceClassification)
from my_pyvene.analyses.visualization import rotation_token_heatmap, rotation_token_layers_heatmap

def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path)
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def main():

    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--model_path', type=str, default='mara589/arithmetic-gpt2', help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--results_path', type=str, default='results/', help='path to the results folder')
    parser.add_argument('--causal_model_type', type=str, choices=['arithmetic', 'simple'], default='arithmetic', help='choose between arithmetic or simple')
    parser.add_argument('--experiment', type=str, choices=['sanity_check', 'empirical', 'attention_weights', 'show_causal_models', 'evaluation', 'compare'])
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()


    simple_path = os.path.join(args.results_path, 'simple')
    args.results_path = os.path.join(args.results_path, args.causal_model_type)
    save_dir_path = os.path.join(args.results_path, 'plots')
    os.makedirs(save_dir_path, exist_ok=True)
    if not os.path.exists(args.results_path):
        raise argparse.ArgumentTypeError("Invalid results_path. You also need to have your intervenable models trained and saved. Refer to README.md file.")

    set_seed(args.seed)
    
    # tokenizer = load_tokenizer('gpt2')
    # model_config = GPT2Config.from_pretrained(args.model_path)
    # model_config.pad_token_id = tokenizer.pad_token_id
    # model = GPT2ForSequenceClassification.from_pretrained(args.model_path, config=model_config)
    # model.resize_token_embeddings(len(tokenizer))

    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_config = GPT2Config.from_pretrained(args.model_path)
    model = GPT2ForSequenceClassification.from_pretrained(args.model_path, config=model_config)

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
            for experiment_id in [64, 128, 256]:
                sanity_check_visualization(args.results_path, save_dir_path, model_config.n_layer, cm_id, experiment_id, arithmetic_family)
    elif args.experiment == 'empirical':
        for cm_id, model_info in arithmetic_family.causal_models.items():
            if cm_id == 4:
                continue
            for experiment_id in [1]:
                empirical_visualization(args.results_path, save_dir_path, model_config.n_layer, cm_id, experiment_id, model_info['label'])
    elif args.experiment == 'attention_weights':

        dir_path = os.path.join(args.results_path, 'intervenable_models_plots')
        os.makedirs(dir_path, exist_ok=True)

        # hardcode_labels = ['X+Y', 'X+Z', 'Y+Z'] # arithmetic
        hardcode_labels = ['X', 'Y', 'Z', 'X+Y+Z'] # simple

        for low_rank_dimension in [256]:

            lrd_path = os.path.join(dir_path, f'{low_rank_dimension}')
            os.makedirs(lrd_path, exist_ok=True)

            
            for cm_id, model_info in arithmetic_family.causal_models.items():

                # if cm_id == 2 or cm_id == 3:
                #     continue

                path = os.path.join(lrd_path, f'{model_info["label"]}_{low_rank_dimension}_attention_weights.pdf')
                rotation_token_layers_heatmap(args.results_path, 
                                cm_id,
                                hardcode_labels[cm_id - 1],
                                low_rank_dimension,
                                model,
                                tokens = ['X', '+', 'Y', '+', 'Z', '='],
                                token_size = 6,
                                variables = [hardcode_labels[cm_id - 1]],
                                intervention_size = 1,
                                fig_path=path)
                    
    elif args.experiment == 'evaluation':
        dir_path = os.path.join(args.results_path, 'evals')
        for cm_id, model_info in arithmetic_family.causal_models.items():
            for experiment_id in [256]:
                evaluation_visualization(dir_path, save_dir_path, 8, cm_id, experiment_id)

        evaluation_visualization_combined(dir_path, save_dir_path, 8, arithmetic_family, 256)

    elif args.experiment == 'compare':
        arithmetic_family = ArithmeticCausalModels()
        compare_intermediate_vs_simple(args.results_path, simple_path, save_dir_path,model_config.n_layer, arithmetic_family, experiment_id=256)

if __name__ =="__main__":
    main()