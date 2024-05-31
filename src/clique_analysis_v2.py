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
    parser.add_argument('--results_path', type=str, default='disentangling_results/', help='path to the results folder where you have saved the graphs')
    parser.add_argument('--causal_model_type', type=str, choices=['arithmetic', 'simple'], default='arithmetic', help='choose between arithmetic or simple')
    parser.add_argument('--layer', type=int, default=0, help='layer corresponding to the graphs obtained that you want to analyse')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    args.results_path = os.path.join(args.results_path, args.causal_model_type)
    if not os.path.exists(args.results_path):
        raise argparse.ArgumentTypeError("Invalid results_path. Path does not exist.")
    
    set_seed(args.seed)
    
    arithmetic_family = ArithmeticCausalModels()
    maximal_cliques = {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["skyblue", "lightcoral", "lightgreen"]

    # numbers = range(1, 3)
    # repeat = 3
    # arrangements = list(product(numbers, repeat=repeat))
    i = 0
    # limit = 100

    for cm_id, model_info in arithmetic_family.causal_models.items():

        print(f'Loading graph {cm_id}..')
        graph_path = os.path.join(args.results_path, f'graphs/graph_{cm_id}_{args.layer}.pt')
        graph = torch.load(graph_path)
        mask = (graph == 1).float()
        graph = graph * mask

        print('Constructing graph..')
        G = nx.from_numpy_array(graph.numpy())
        pos = nx.spring_layout(G)
        node_colors = [colors[i] for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_color=node_colors, ax=axes[i])
        axes[i].set_title(model_info['label'])

        print('Finding cliques..')
        cliques = list(nx.find_cliques(G))
        maximal_cliques[cm_id] = filter_by_max_length(cliques)

        print(len(maximal_cliques[cm_id][0]))
        max_clique_nodes = set(node for clique in maximal_cliques[cm_id] for node in clique)
        print(max_clique_nodes)

        # print(f"Maximal cliques for {model_info['label']}: {maximal_cliques[cm_id]}")
        print(len(max_clique_nodes))
        print(f"Perfecntage of nodes in maximal cliques: {len(max_clique_nodes)/len(graph)}")

        nodes_in_all_max_cliques = set.intersection(*map(set, cliques))
        print("Nodes in all maximal cliques:", nodes_in_all_max_cliques)

        i += 1

    plt.savefig('graphs.png')
    plt.close()

if __name__ =="__main__":
    main()