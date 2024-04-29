import sys, os
sys.path.append(os.path.join('..', '..'))
import torch
import argparse
import numpy as np
from causal_models import ArithmeticCausalModels
from pyvene import set_seed
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def main():
    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    # parser.add_argument('--model_path', type=str, help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--results_path', type=str, default='results/', help='path to the results folder')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    if not os.path.exists(args.results_path):
        raise argparse.ArgumentTypeError("Invalid results_path. Path does not exist.")

    # load labelled graph
    graph_path = os.path.join(args.results_path, 'graph.pt')
    graph = torch.load(graph_path)

    # load subset of bases
    subset_bases_path = os.path.join(args.results_path, 'testing_bases.npy')
    T = np.load(subset_bases_path, allow_pickle=True)
    
    set_seed(args.seed)

    arithmetic_family = ArithmeticCausalModels()
    D = []

    for cm_id, model_info in arithmetic_family.causal_models.items():
        best_combo_path = os.path.join(args.results_path, f'class_data_{cm_id}.npy')
        loaded_arr = np.load(best_combo_path, allow_pickle=True)
        for x in loaded_arr:
            T[x]['class'] = cm_id
            D.append(T[x])
    
    # construct dataset
    df = pd.DataFrame(D)
    features = df[['X', 'Y', 'Z']]
    labels = df['class']
    
    # train the classification
    model = DecisionTreeClassifier()
    model.fit(features, labels) 

    # predict
    new_sample = {'X': 2, 'Y': 1, 'Z': 2}
    prediction = model.predict(np.array(list(new_sample.values())).reshape(1, -1)) 
    print(prediction)

if __name__ =="__main__":
    main()