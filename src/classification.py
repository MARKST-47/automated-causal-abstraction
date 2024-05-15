import sys, os
sys.path.append(os.path.join('..', '..'))
import torch
import argparse
import numpy as np
from causal_models import ArithmeticCausalModels
from pyvene import set_seed
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from itertools import product
from utils import construct_input
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

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

    # get all the arrangements
    numbers = range(1, 11)
    arrangements = product(numbers, repeat=3)
    arr = []
    for x,y,z in arrangements:
        inp = construct_input(x,y,z)
        arr.append(inp)

    # save arrangements
    arr_saved = np.array(arr)
    arr_path = os.path.join(args.results_path, 'arrangements.npy')
    np.save(arr_path, arr_saved)

    set_seed(args.seed)

    arithmetic_family = ArithmeticCausalModels()
    D = []

    # construct dataset
    for cm_id, model_info in arithmetic_family.causal_models.items():
        best_combo_path = os.path.join(args.results_path, f'class_data_{cm_id}.npy')
        loaded_arr = np.load(best_combo_path, allow_pickle=True)
        for x in loaded_arr:
            arr[x]['class'] = cm_id
            D.append(arr[x])
    
    df = pd.DataFrame(D)
    features = df[['X', 'Y', 'Z']]
    features['X_f1'] = features['X'] < 3
    features['X_f2'] = (features['X'] >= 3) & (features['X'] <= 6)
    features['X_f3'] = features['X'] >= 7
    features['Y_f1'] = features['Y'] < 3
    features['Y_f2'] = (features['Y'] >= 3) & (features['Y'] <= 6)
    features['Y_f3'] = features['Y'] >= 7
    features['Z_f1'] = features['Z'] < 3
    features['Z_f2'] = (features['Z'] >= 3) & (features['Z'] <= 6)
    features['Z_f3'] = features['Z'] >= 7
    labels = df['class']

    # split data intro training and testing data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    print(f'Training size: {len(y_train)}')
    print(f'Testing size: {len(y_test)}')
    
    # hyperparameter search
    param_grid = {
        'criterion': ['gini', 'entropy'], # measures node purity, the thing to minimize
        'max_depth': [2, 3, 4, 5, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2'],
        'ccp_alpha': [0.0, 0.015, 0.1]
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
    # train the classification
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    # testing
    prediction = model.predict(X_test)
    testing_df = pd.DataFrame(X_test[['X', 'Y', 'Z']])
    testing_df['true_class'] = y_test
    testing_df['predicted_class'] = prediction
    print(testing_df)

    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=features.columns, class_names=["1", "2", "3"])
    plt.title("Decision Tree Visualization")
    file_path = os.path.join(args.results_path, "decision_tree.png")
    plt.savefig(file_path)
    plt.close()

if __name__ =="__main__":
    main()