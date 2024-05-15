import sys, os
sys.path.append(os.path.join('..', '..'))
import torch
import argparse
import numpy as np
from causal_models import ArithmeticCausalModels
from pyvene import set_seed
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from utils import biased_sampler_1, biased_sampler_2, biased_sampler_3

def main():
    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    # parser.add_argument('--model_path', type=str, help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--results_path', type=str, default='results/', help='path to the results folder')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    parser.add_argument('--n_testing', type=int, default=100, help='number of samples used for predicton during testing')
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

    # predict for class 1 aka (A+B)+C
    for _ in args.n_testing:
        sample = biased_sampler_1()
        prediction = model.predict(np.array(list(sample.values())).reshape(1, -1))
        print(sample, prediction)

    # prediction for class 2 aka (A+C)+B
    for _ in args.n_testing:
        sample = biased_sampler_2()
        prediction = model.predict(np.array(list(sample.values())).reshape(1, -1))
        print(sample, prediction)

    # prediction for class 3 aka A+(B+C)
    for _ in args.n_testing:
        sample = biased_sampler_3()
        prediction = model.predict(np.array(list(sample.values())).reshape(1, -1))
        print(sample, prediction)

if __name__ =="__main__":
    main()