import sys, os
sys.path.append(os.path.join('..', '..'))

import torch
import random
from sklearn.metrics import classification_report
from pyvene import CausalModel
from datetime import datetime
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import argparse
import pickle
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          GPT2Config,
                          AutoTokenizer,
                          GPT2ForSequenceClassification)
from causal_models import ArithmeticCausalModels
from utils import arithmetic_input_sampler, construct_arithmetic_input
from itertools import product
from sklearn.metrics import accuracy_score

def tokenizePrompt(prompt, tokenizer):
    prompt = f"{prompt['X']}+{prompt['Y']}+{prompt['Z']}=" # prompt for numerical causal model
    return tokenizer.encode(prompt, return_tensors='pt')

def evaluate(dataloader, model, device_):

    model.eval()

    predictions_labels = []
    true_labels = []

    for batch in tqdm(dataloader, total=len(dataloader)):

        true_labels += batch['labels'][0].type(torch.long).tolist() # for eval

        batch['input_ids'] = torch.stack(batch['input_ids'][0]).T.to(device_)
        batch['labels'] = batch['labels'][0].type(torch.long).to(device_)

        # batch = {k:torch.tensor(v).type(torch.long).to(device_) for k,v in batch.items()} # move to device

        with torch.no_grad():        

            outputs = model(**batch)
            _, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content

    return true_labels, predictions_labels

def main():

    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--model_path', type=str, default='mara589/arithmetic-gpt2', help='model to finetune on the task')
    parser.add_argument('--results_path', type=str, default='evaluate_llm/', help='path to the results folder')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--n_testing', type=int, default=2560, help='number of training samples')
    parser.add_argument('--seed', type=int, default=123, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    os.makedirs(args.results_path, exist_ok=True)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    offset = 3

    # Sequence Classification with GPT2 n_labels=28
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_config = GPT2Config.from_pretrained(args.model_path)
    model = GPT2ForSequenceClassification.from_pretrained(args.model_path, config=model_config)

    numbers = range(1, 11)
    repeat = 3
    arrangements = list(product(numbers, repeat=repeat))

    tokenized_cache = {}
    for arrangement in arrangements:
        tokenized_cache[arrangement] = tokenizePrompt(construct_arithmetic_input(arrangement), tokenizer)

    # generate testing data
    causal_model_family = ArithmeticCausalModels()
    causal_model = causal_model_family.get_model_by_id(1) # doesn't matter which causal model you choose to generate the factual data for training gpt2
    
    data_path=f'/home/mara/workspace/LLM_causal_model_learning/low_iia_data.pkl'
    with open(data_path, 'rb') as file:
        data_ids = pickle.load(file)

    def sampler():
        random_id = random.choice(data_ids)
        return construct_arithmetic_input(arrangements[random_id])
    
    test_inputs, test_labels = causal_model.generate_factual_dataset(args.n_testing, sampler, inputFunction=lambda x: tokenized_cache[tuple(x.values())])

    # load model to defined device.
    model.to(device)
    print('Model loaded to `%s`'%device)

    # test model
    test_ds = Dataset.from_dict(
        {
            "labels": test_labels - offset,
            "input_ids": test_inputs
        }
    )

    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    true_labels, predictions_labels = evaluate(test_dataloader, model, device)
    evaluation_report = classification_report(true_labels, predictions_labels, labels=list(test_labels.squeeze()))
    print(evaluation_report)

    accuracy = accuracy_score(true_labels, predictions_labels)
    print(f"Accuracy: {accuracy:.2%}") 

if __name__ =="__main__":
    main()