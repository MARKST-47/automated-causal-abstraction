import sys, os
sys.path.append(os.path.join('..', '..'))

import argparse
from pyvene import set_seed
import torch
import numpy as np
from causal_models import ArithmeticCausalModels
import matplotlib.pyplot as plt
import networkx as nx
import itertools

def get_all_combinations(cliques):
    all_combinations = list(itertools.product(*cliques.values()))
    return all_combinations

def filter_by_max_length(list_of_lists):
    max_len = max(len(sublist) for sublist in list_of_lists)
    return [sublist for sublist in list_of_lists if len(sublist) == max_len]

def calculate_overlap(tuple_):
    total_count = 0
    overlapping_percentage = 0
    merged_set = set()
    for sublist in tuple_:
        total_count += len(sublist)
        merged_set = merged_set.union(set(sublist)) # merge sets
    overlap = total_count - len(merged_set)
    if total_count != 0:
        overlapping_percentage = (overlap / total_count)
    return overlapping_percentage, overlap

def find_least_overlap_tuple(list_of_tuples):
    min_overlap = float('inf')
    best_tuple = None

    for tuple_ in list_of_tuples:
        _, overlap = calculate_overlap(tuple_)
        if overlap < min_overlap:
            min_overlap = overlap
            best_tuple = tuple_

    return best_tuple

def main():
    
    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    # parser.add_argument('--model_path', type=str, help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--results_path', type=str, default='results/', help='path to the results folder')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    # if not os.path.exists(args.model_path):
    #     raise argparse.ArgumentTypeError("Invalid model_path. Path does not exist.")
    
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

    # construct entire graph
    G = nx.Graph()

    for i in range(graph.shape[0]):
        G.add_node(i)

    # undirected graph edge-label construction
    for i in range(graph.shape[0]):
        for j in range(i+1, graph.shape[0]):
            if graph[i, j] != 0:
                G.add_edge(i, j, label=graph[i, j].item()) # label means the causal model

    # saving the graph with all nodes
    pos = nx.spring_layout(G, k=0.1, iterations=100)
    color_map = {1: 'red', 2: 'green', 3: 'blue'}
    edge_colors = [color_map[w['label']] for (u, v, w) in G.edges(data=True)]

    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color=edge_colors, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.savefig(f'graph.png')
    plt.close()

    # construct subgraphs based on label
    subgraphs = {}
    for label in set(nx.get_edge_attributes(G, 'label').values()):
        subgraph = nx.Graph()

        for u, v, data in G.edges(data=True):
            if data['label'] == label:

                if u not in subgraph.nodes(): 
                    subgraph.add_node(u)
                if v not in subgraph.nodes():
                    subgraph.add_node(v)

                if not subgraph.has_edge(u, v):
                    subgraph.add_edge(u, v, label=label)

        subgraphs[arithmetic_family.get_label_by_id(label)] = subgraph

    # find optimal positions for each subgraph
    positions = {}
    for label, subgraph in subgraphs.items():
        # spring layout is an algorithm for showing the graph in a pretty way
        positions[label] = nx.spring_layout(subgraph, k=0.3, iterations=50)

    num_subgraphs = len(subgraphs.keys())
    fig, axes = plt.subplots(nrows=1, ncols=num_subgraphs, figsize=(12, 5))

    color_map = {1: 'red', 2: 'green', 3: 'blue'}

    i = 0 # axes id
    maximal_cliques = {}
    for label, subgraph in subgraphs.items():

        cliques = list(nx.find_cliques(subgraph))
        # biggest_clique = max(cliques, key=len)
        # print(cliques)
        # print(biggest_clique)
        maximal_cliques[label] = filter_by_max_length(cliques)
        pos = positions[label]

        # visualizing the subgraphs
        nx.draw_networkx(
            subgraph, 
            pos=pos, 
            node_size=40, 
            node_color='lightblue', 
            edge_color=[color_map[subgraph[u][v]['label']] for u, v in subgraph.edges()],
            with_labels=True, 
            font_size=8,
            ax=axes[i]
        )

        # highlight cliques with these colors
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'lime'] # len is 7
        
        for j, clique in enumerate(cliques):
            nx.draw_networkx_nodes(
                subgraph,
                pos,
                nodelist=clique, 
                node_color=colors[j % len(colors)], # so we kinda have different colors
                ax=axes[i]
            )

        axes[i].axis('off')
        axes[i].set_title(f"All connected edges with label {label}")
        i += 1

    plt.tight_layout()
    plt.savefig('connected_component_visualization.png')
    plt.close()

    all_combinations = get_all_combinations(maximal_cliques)
    best_combo = find_least_overlap_tuple(all_combinations)
    overlap_percentage, _ = calculate_overlap(best_combo)
    print(overlap_percentage)
    i = 0
    for data in best_combo:
        print(data)
        best_combo_path = os.path.join(args.results_path, f'class_data_{i+1}.npy')
        np.save(best_combo_path, data)
        # loaded_arr = np.load(best_combo_path, allow_pickle=True)
        i += 1
    
if __name__ =="__main__":
    main()