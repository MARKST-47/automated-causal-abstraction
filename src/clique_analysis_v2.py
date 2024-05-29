import sys, os
sys.path.append(os.path.join('..', '..'))

import argparse
from pyvene import set_seed
import torch
import numpy as np
from causal_models import ArithmeticCausalModels
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product
from utils import filter_by_max_length, construct_arithmetic_input

def main():
    
    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    # parser.add_argument('--model_path', type=str, help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--results_path', type=str, help='path to the results folder where you have saved the graphs')
    parser.add_argument('--causal_model_type', type=str, choices=['arithmetic', 'simple'], default='arithmetic', help='choose between arithmetic or simple')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    args.results_path = os.path.join(args.results_path, args.causal_model_type)
    if not os.path.exists(args.results_path):
        raise argparse.ArgumentTypeError("Invalid results_path. Path does not exist.")
    
    set_seed(args.seed)
    
    arithmetic_family = ArithmeticCausalModels()
    layer = 0
    maximal_cliques = {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["skyblue", "lightcoral", "lightgreen"]

    numbers = range(1, 3)
    arrangements = product(numbers, repeat=3)
    arr = []
    for i, data in enumerate(arrangements):
        # arr.append(construct_arithmetic_input(data))
        print(f'{i}: {construct_arithmetic_input(data)}')
    
    i = 0

    for cm_id, model_info in arithmetic_family.causal_models.items():
        graph_path = os.path.join(args.results_path, f'graphs/graph_{cm_id}_{layer}.pt')
        graph = torch.load(graph_path)
        mask = (graph == 1).float()
        graph = graph * mask

        G = nx.from_numpy_array(graph.numpy())
        pos = nx.spring_layout(G)
        node_colors = [colors[i] for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_color=node_colors, ax=axes[i])
        axes[i].set_title(model_info['label'])

        cliques = list(nx.find_cliques(G))
        maximal_cliques[cm_id] = filter_by_max_length(cliques)
        print(f"Maximal cliques for {model_info['label']}: {maximal_cliques[cm_id]}")

        i += 1

    plt.tight_layout()
    plt.savefig('graphs.png')
    plt.close()

if __name__ =="__main__":
    main()