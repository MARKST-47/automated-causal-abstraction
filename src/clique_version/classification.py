import sys, os
sys.path.append(os.path.join('..', '..'))
import argparse
from causal_models import ArithmeticCausalModels, SimpleSummingCausalModels
from pyvene import set_seed
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle
from utils import construct_arithmetic_input
import random
from itertools import product

def main():
    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--results_path', type=str, default='results/', help='path to the results folder')
    parser.add_argument('--causal_model_type', type=str, choices=['arithmetic', 'simple'], default='arithmetic', help='choose between arithmetic or simple')
    parser.add_argument('--layer', type=int, default=0, help='layer corresponding to the graphs obtained that you want to analyse')
    parser.add_argument('--low_rank_dimension', type=int, default=256, help='low_rank_dimension corresponding to the graphs obtained that you want to analyse')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    if not os.path.exists(args.results_path):
        raise argparse.ArgumentTypeError("Invalid results_path. Path does not exist.")

    if args.causal_model_type == 'arithmetic':
        arithmetic_family = ArithmeticCausalModels()
    elif args.causal_model_type == 'simple':
        arithmetic_family = SimpleSummingCausalModels()
    else:
        raise ValueError(f"Invalid causal model type: {args.causal_model_type}. Can only choose between arithmetic or simple.")

    # args.results_path = os.path.join(args.results_path, args.causal_model_type)
    save_plots_path = os.path.join(args.results_path, 'classification_plots')
    os.makedirs(save_plots_path, exist_ok=True)
    save_models_path = os.path.join(args.results_path, 'classification_models')
    os.makedirs(save_models_path, exist_ok=True)

    # get all the arrangements
    numbers = range(1, 11)
    arrangements = list(product(numbers, repeat=3))

    labels = ['(X)+Y+Z', '(X+Y)+Z', '(X+Y+Z)']

    set_seed(args.seed)

    for label in labels:
        merged_data = set()
        data_path = os.path.join(args.results_path, f'classification_data/{args.low_rank_dimension}/data_{label}_{args.layer}.pkl')
        with open(data_path, 'rb') as file:
            data_ids = pickle.load(file)
            merged_data.update(data_ids)
        
        D = []

        # construct positive inputs
        for id in merged_data:
            input = construct_arithmetic_input(arrangements[id])
            input['class'] = 1
            D.append(input)
    
        # construct negative inputs
        for _ in range(len(merged_data)):
            id = random.choice(range(len(arrangements)))
            while id in merged_data:
                id = random.choice(range(len(arrangements)))
            input = construct_arithmetic_input(arrangements[id])
            input['class'] = 0
            D.append(input)
    
        df = pd.DataFrame(D)

        features = df[['X', 'Y', 'Z']]
    
        # parity features
        features['X_even'] = features['X'] % 2 == 0
        features['Y_even'] = features['Y'] % 2 == 0
        features['Z_even'] = features['Z'] % 2 == 0

        # sum features
        features['XY_sum'] = features['X'] + features['Y']
        features['XZ_sum'] = features['X'] + features['Z']
        features['YZ_sum'] = features['Y'] + features['Z']
        features['XYZ_sum'] = features['X'] + features['Y'] + features['Z']

        # divisible features --> they don't really make sense here honestly
        divisors = [3, 5, 7]
        for divisor in divisors:
            features[f'X_divisible_by_{divisor}'] = features['X'] % divisor == 0
            features[f'Y_divisible_by_{divisor}'] = features['Y'] % divisor == 0
            features[f'Z_divisible_by_{divisor}'] = features['Z'] % divisor == 0

        labels = df['class']

        # split data intro training and testing data
        X_train, y_train = features, labels

        print(f'Training size: {len(y_train)}')

        # hyperparameter search
        param_grid = {
            'criterion': ['gini', 'entropy'], 
            'max_depth': [1, 2, 3, 4],
            'min_samples_split': [6, 7, 8, 9, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5, 6],
            'max_features': [None, 'sqrt', 'log2'],
            'ccp_alpha': [0.0, 0.001, 0.005]
        }

        grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
        
        # train the classification
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

        print(f"Best parameters for layer {args.layer} and model {label}:", grid_search.best_params_)

        # check accuracy on training data
        prediction = model.predict(X_train)
        testing_df = pd.DataFrame(X_train[['X', 'Y', 'Z']])
        testing_df['true_class'] = y_train
        testing_df['predicted_class'] = prediction
        accuracy = (testing_df['true_class'] == testing_df['predicted_class']).mean()
        print(f"Accuracy on layer {args.layer}: {accuracy:.2%}")

        plt.figure(figsize=(12, 8))
        # plot_tree(model, filled=True, feature_names=features.columns, class_names=["Non-Clique", "Clique"])
        plot_tree(model, 
            filled=True,
            feature_names=features.columns,
            class_names=["Non-Clique", "Clique"],
            rounded=True,
            fontsize=12,
            max_depth=5,
            impurity=False,
            node_ids=False,
            proportion=True,
            precision=2
        )  
        # plt.title(f"Decision Tree - layer {args.layer}, lrd {args.low_rank_dimension}")
        plt.tight_layout()
        file_path = os.path.join(save_plots_path, f'{args.low_rank_dimension}')
        os.makedirs(file_path, exist_ok=True)
        file_path = os.path.join(file_path, f"decision_tree_{label}_{args.layer}.pdf")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

        file_path = os.path.join(save_models_path, f'{args.low_rank_dimension}')
        os.makedirs(file_path, exist_ok=True)
        file_path = os.path.join(file_path, f"decision_tree_{label}_{args.layer}.pkl")

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

if __name__ =="__main__":
    main()