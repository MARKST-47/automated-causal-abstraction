import os
from causal_models import ArithmeticCausalModels, SimpleSummingCausalModels
import argparse
from pyvene import set_seed
from utils import sanity_check_visualization, empirical_visualization
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

def main():

    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--model_path', type=str, help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--results_path', type=str, default='results/', help='path to the results folder')
    parser.add_argument('--causal_model_type', type=str, choices=['arithmetic', 'simple'], default='arithmetic', help='choose between arithmetic or simple')
    parser.add_argument('--experiment', type=str, choices=['sanity_check', 'empirical', 'attention_weights', 'show_causal_models'])
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    save_dir_path = os.path.join(args.results_path, 'plots')
    os.makedirs(save_dir_path, exist_ok=True)

    args.results_path = os.path.join(args.results_path, args.causal_model_type)
    if not os.path.exists(args.results_path):
        raise argparse.ArgumentTypeError("Invalid results_path. You also need to have your intervenable models trained and saved. Refer to README.md file.")

    set_seed(args.seed)

    if not os.path.exists(args.model_path):
        raise argparse.ArgumentTypeError("Invalid model_path. Path does not exist.")
    
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

    if args.experiment == 'show_causal_models':
        for cm_id, model_info in arithmetic_family.causal_models.items():
            model_info['causal_model'].print_structure(fig_name=model_info['label'])
    elif args.experiment == 'sanity_check':
        for cm_id, model_info in arithmetic_family.causal_models.items():
            for experiment_id in [64, 128, 256, 768, 4608]:
                sanity_check_visualization(args.results_path, save_dir_path, model_config.n_layer, cm_id, experiment_id, arithmetic_family)
    elif args.experiment == 'empirical':
        for cm_id, model_info in arithmetic_family.causal_models.items():
            if cm_id == 1:
                continue
            for experiment_id in [64, 128, 256]:
                empirical_visualization(args.results_path, save_dir_path, model_config.n_layer, cm_id, experiment_id, model_info['label'])
    elif args.experiment == 'attention_weights':

        dir_path = os.path.join(args.results_path, 'intervenable_models_plots')
        os.makedirs(dir_path, exist_ok=True)

        hardcode_labels = ['X+Y', 'X+Z', 'Y+Z']
        
        for low_rank_dimension in [64, 128, 256]:

            lrd_path = os.path.join(dir_path, f'{low_rank_dimension}')
            os.makedirs(lrd_path, exist_ok=True)

            for layer in range(12):
                for cm_id, model_info in arithmetic_family.causal_models.items():

                        intervenable_model_path = os.path.join(args.results_path, f'intervenable_models/cm_{cm_id}/intervenable_{low_rank_dimension}_{layer}')
                        intervenable = IntervenableModel.load(intervenable_model_path, model=model)
                        intervenable.set_device("cuda")

                        for k, v in intervenable.interventions.items():
                            path = os.path.join(lrd_path, f'{model_info["label"]}_{low_rank_dimension}_{k}.png')
                            rotation_token_heatmap(v[0].rotate_layer.cpu(),
                                                tokens = ['X', '+', 'Y', '+', 'Z', '='], 
                                                token_size = 6, 
                                                variables = [hardcode_labels[cm_id - 1]], 
                                                intervention_size = 1,
                                                fig_path=path)

if __name__ =="__main__":
    main()