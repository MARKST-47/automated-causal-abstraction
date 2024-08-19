import sys, os
sys.path.append(os.path.join('..', '..'))
import argparse
import pickle
import json
from pyvene import set_seed
import torch
import networkx as nx
from networkx.readwrite import json_graph
from itertools import product
from utils import construct_arithmetic_input

def pairwise_node_overlap(G1, G2):
    """Calculates the number of overlapping nodes between two NetworkX graphs."""
    return len(set(G1.nodes()) & set(G2.nodes()))

def main():
    
    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--results_path', type=str, default='results/', help='path to the results folder where you have saved the graphs')
    parser.add_argument('--layer', type=int, default=0, help='layer corresponding to the graphs obtained that you want to analyse')
    parser.add_argument('--low_rank_dimension', type=int, default=256, help='low_rank_dimension corresponding to the graphs obtained that you want to analyse')
    parser.add_argument('--top_k', type=int, default=100, help='get the top 100 nodes representing the best each intervenable model data')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    set_seed(args.seed)

    args.results_path = os.path.join(args.results_path)
    if not os.path.exists(args.results_path):
        raise argparse.ArgumentTypeError("Invalid results_path. Path does not exist.")
    
    classification_path = os.path.join(args.results_path, 'classification_data')
    os.makedirs(classification_path, exist_ok=True)

    cliques_info_path = os.path.join(args.results_path, 'cliques_info')
    os.makedirs(cliques_info_path, exist_ok=True)

    labels = ['(X)+Y+Z', '(X+Y)+Z', '(X+Y+Z)']
    exclude_list = set()

    for label in labels:

        print(f'Loading graph for lrd {args.low_rank_dimension}, layer {args.layer}, model {label}')
        graph_path = os.path.join(args.results_path, f'graphs/{label}_graph_{args.low_rank_dimension}_{args.layer}.pt')
        graph = torch.load(graph_path)
        # mask = (graph == 1).float()
        # graph = graph * mask
        graph.fill_diagonal_(0)

        print('Constructing graph..')
        G = nx.from_numpy_array(graph.numpy())

        # obtaining the list of nodes sorted by degree
        node_degrees = dict(G.degree())
        sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)
        top_k_nodes = [node for node in sorted_nodes if node not in exclude_list][:args.top_k]
        exclude_list.update(top_k_nodes)

        subgraph = G.subgraph(top_k_nodes)

        iia = sum(data['weight'] for _, _, data in subgraph.edges(data=True))
        num_edges = subgraph.number_of_edges()
        print(iia/num_edges)

        graph_density = nx.density(subgraph)
        avg_degree = sum(dict(subgraph.degree()).values()) / subgraph.number_of_nodes()

        print(f'Graph density: {graph_density}')
        print(f'Average node degree: {avg_degree}')

        save_data_path = os.path.join(classification_path, f'{args.low_rank_dimension}')
        os.makedirs(save_data_path, exist_ok=True)
        data_path = os.path.join(save_data_path, f'data_{label}_{args.layer}.pkl')
        
        with open(data_path, 'wb') as f:
            pickle.dump(top_k_nodes, f)


if __name__ =="__main__":
    main()