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
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from utils import visualize_graph

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
    
    classification_path = os.path.join(args.results_path, 'classification_data_2')
    os.makedirs(classification_path, exist_ok=True)

    cliques_info_path = os.path.join(args.results_path, 'cliques_info')
    os.makedirs(cliques_info_path, exist_ok=True)

    # labels = ['(X)+Y+Z', '(X+Y)+Z', '(X+Y+Z)']
    labels = ['(X+Y+Z)']

    # limits = [num for num in range(10, 1001, 20)]
    # limits.append(1000)
    # print(len(limits))

    # all_accs = {}
    
    exclude_list = set()

    for label in labels:
        accs = []

        print(f'Loading graph for lrd {args.low_rank_dimension}, layer {args.layer}, model {label}')
        graph_path = os.path.join(args.results_path, f'graphs_3/{label}_graph_{args.low_rank_dimension}_{args.layer}.pt')
        graph = torch.load(graph_path)
        # mask = (graph == 1).float()
        # graph = graph * mask
        graph.fill_diagonal_(0)

        print('Constructing graph..')
        G = nx.from_numpy_array(graph.numpy())
        # G = G.subgraph([990, 991, 992, 993, 994, 995, 996, 997, 998, 999])
        # visualize_graph(G, label=label)
        # print(G.nodes())

        node_degrees = dict(G.degree())
        sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)

        if label == '(X)+Y+Z':
            limits = [10]
            # limits = [num for num in range(10, 501, 20)]
            # limits.append(500)
        elif label == '(X+Y)+Z':
            # limits = [num for num in range(10, 501, 20)]
            # limits.append(500)
            limits = [10]
        else:
            # limits = [500]
            limits = [num for num in range(10, 1001, 20)]
            limits.append(1000)
            # limits = [20]

        # limits = [num for num in range(10, 1001, 20)]
        # limits.append(1000)
        print(limits)

        for top_k in limits:

            args.top_k = top_k

            # exclude_list = set(top_k_nodes)
            temp_exclude_list = exclude_list.copy()
            top_k_nodes = []

            iia = 0
            temp_iia = 0
            margin_of_error = 0

            while len(top_k_nodes) < args.top_k:
                iia_before = iia
                for node in sorted_nodes:
                    if node in temp_exclude_list:
                        continue
                    subgraph = G.subgraph([n for n in top_k_nodes + [node]])
                    temp_iia = sum(data['weight'] for _, _, data in subgraph.edges(data=True))
                    num_nodes = subgraph.number_of_nodes()
                    if (num_nodes*(num_nodes - 1)/2) > 0:
                        temp_iia = temp_iia/(num_nodes*(num_nodes - 1)/2)
                    if temp_iia >= iia - margin_of_error:
                        iia = temp_iia
                        top_k_nodes.append(node)
                    
                    if len(top_k_nodes) >= args.top_k:
                        break
            
                temp_exclude_list.update(top_k_nodes)

                if iia_before == iia:
                    margin_of_error += 0.01

            subgraph = G.subgraph(top_k_nodes)
            num_nodes = subgraph.number_of_nodes()
            iia = sum(data['weight'] for _, _, data in subgraph.edges(data=True))/(num_nodes*(num_nodes - 1)/2)
            accs.append((top_k, iia))
            print(top_k, iia)

            # check IIA on the rest of the nodes not selected in top_k

            # all_nodes_in_G = set(G.nodes())
            # nodes_not_in_subgraph = all_nodes_in_G - set(top_k_nodes)
            # subgraph = G.subgraph(nodes_not_in_subgraph)
            # num_nodes = subgraph.number_of_nodes()
            # iia = sum(data['weight'] for _, _, data in subgraph.edges(data=True))/(num_nodes*(num_nodes - 1)/2)
            # print(top_k, iia)
            
            # if label == '(X)+Y+Z' or label == '(X+Y)+Z':
            #     exclude_list.update(top_k_nodes)
            
            # visualize_graph(subgraph, label=label)
            
            # save_data_path = os.path.join(classification_path, f'{args.low_rank_dimension}')
            # os.makedirs(save_data_path, exist_ok=True)
            # data_path = os.path.join(save_data_path, f'data_{label}_{args.layer}.pkl')
            
            # with open(data_path, 'wb') as f:
            #     pickle.dump(top_k_nodes, f)

        file_name = f'{label}_iias_top_k.txt'
        if os.path.isfile(file_name):
            os.remove(file_name)
        
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['top_k', 'iia'])
            for top_k, iia in accs:
                writer.writerow([top_k, iia])
                
if __name__ =="__main__":
    main()