import sys, os
sys.path.append(os.path.join('..', '..'))
import argparse
from pyvene import set_seed
import torch
import networkx as nx
import csv
import pickle
import shutil
from multiprocessing import Pool
import json

def calculate_iia(graph):
    num_nodes = graph.number_of_nodes()
    total_weight = sum(data['weight'] for _, _, data in graph.edges(data=True))
    return total_weight / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0

def copy_files(top_k_path, data_division_path, l, cm_id, label, args):

    source_file = os.path.join(top_k_path, f'exp_{l}_{label}_faithfulness_{args.faithfulness}_layer_{args.layer}.txt')
    destination_file = os.path.join(top_k_path, f'exp_{cm_id}_{label}_faithfulness_{args.faithfulness}_layer_{args.layer}.txt')

    shutil.copyfile(source_file, destination_file)

    source_file = os.path.join(data_division_path, f'exp_{l}_{label}_faithfulness_{args.faithfulness}_layer_{args.layer}.pkl')
    destination_file = os.path.join(data_division_path, f'exp_{cm_id}_{label}_faithfulness_{args.faithfulness}_layer_{args.layer}.pkl')

    shutil.copyfile(source_file, destination_file)

def greedy_selection_of_cm(labels, exclude_list, low_rank_dimension, layer, results_path):
    max_iia = 0
    best_label = ''
    best_num_nodes = 0

    for label in labels:

        print(f'Loading graph for lrd {low_rank_dimension}, layer {layer}, model {label}')
        graph_path = os.path.join(results_path, f'graphs/{label}_graph_{low_rank_dimension}_{layer}.pt')
        graph = torch.load(graph_path)
        graph.fill_diagonal_(0)

        G = nx.from_numpy_array(graph.numpy())

        subgraph = G.subgraph(set(G.nodes()) - exclude_list)
        num_nodes = subgraph.number_of_nodes()
        iia = sum(data['weight'] for _, _, data in subgraph.edges(data=True))/(num_nodes*(num_nodes - 1)/2)

        if iia > max_iia:
            max_iia = iia
            best_label = label
            best_num_nodes = num_nodes
    
    print(best_label, iia, best_num_nodes)
    return best_label, iia

def run_process(all_labels, args, data_division_path, top_k_path, n_nodes=1000):
    label_enc = {}
    for cm_id, labels in all_labels.items():

        print(labels)

        exclude_list = set()
        visited_labels = []

        # greedy sort based on IIA
        best_label, iia = greedy_selection_of_cm(list(set(labels) - set(['O'])), exclude_list, args.low_rank_dimension, args.layer, args.results_path)
        visited_labels.append(best_label)

        # if len(labels) >= 3:

        #     if labels[0] == '(X)+Y+Z':
        #         l = '1'
        #     elif labels[0] == 'X+(Y)+Z':
        #         l = '2'
        #     elif labels[0] == 'X+Y+(Z)':
        #         l = '3'

        #     source_file = os.path.join(top_k_path, f'exp_{l}_{labels[0]}_faithfulness_{args.faithfulness}_layer_{args.layer}.txt')
        #     destination_file = os.path.join(top_k_path, f'exp_{cm_id}_{labels[0]}_faithfulness_{args.faithfulness}_layer_{args.layer}.txt')

        #     shutil.copyfile(source_file, destination_file)

        #     source_file = os.path.join(data_division_path, f'exp_{l}_{labels[0]}_faithfulness_{args.faithfulness}_layer_{args.layer}.pkl')
        #     destination_file = os.path.join(data_division_path, f'exp_{cm_id}_{labels[0]}_faithfulness_{args.faithfulness}_layer_{args.layer}.pkl')

        #     shutil.copyfile(source_file, destination_file)

        #     with open(destination_file, 'rb') as file:
        #         array = pickle.load(file)
        #         exclude_list.update(array)

        #     labels = labels[1:]
        #     print(labels)

        label = best_label

        while len(visited_labels) < len(labels):

            print(label, len(exclude_list))

            accs = []

            print(f'Loading graph for lrd {args.low_rank_dimension}, layer {args.layer}, model {label}')
            graph_path = os.path.join(args.results_path, f'graphs/{label}_graph_{args.low_rank_dimension}_{args.layer}.pt')
            graph = torch.load(graph_path)
            graph.fill_diagonal_(0)

            print('Constructing graph..')
            G = nx.from_numpy_array(graph.numpy())

            node_degrees = dict(G.degree())
            sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)

            limits = [num for num in range(2, n_nodes + 1 - len(exclude_list), 1)]
            limits.append(n_nodes - len(exclude_list))

            if label == '(X+Y+Z)' or label == 'O':
                limits = [n_nodes - len(exclude_list)]

            for top_k in limits:

                if len(exclude_list) == n_nodes:
                    break

                args.top_k = top_k

                if label == '(X+Y+Z)' or label == 'O':

                    subgraph = G.subgraph(set(G.nodes()) - exclude_list)
                    num_nodes = subgraph.number_of_nodes()
                    iia = sum(data['weight'] for _, _, data in subgraph.edges(data=True))/(num_nodes*(num_nodes - 1)/2)
                    accs.append((top_k, iia))
                    print(top_k, iia)

                    data_path = os.path.join(data_division_path, f'exp_{cm_id}_{label}_faithfulness_{args.faithfulness}_layer_{args.layer}.pkl')
                
                    with open(data_path, 'wb') as f:
                        pickle.dump(subgraph.nodes(), f)
                    
                    continue

                temp_exclude_list = exclude_list.copy()
                top_k_nodes = []

                iia = 0
                temp_iia = 0
                margin_of_error = 0

                while len(top_k_nodes) < args.top_k and len(top_k_nodes) != n_nodes:
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

                if args.faithfulness <= iia or label == '(X+Y+Z)' or label == 'O':
                    temp_top_k_nodes = top_k_nodes

                    subgraph = G.subgraph(top_k_nodes)
                    num_nodes = subgraph.number_of_nodes()
                    iia = sum(data['weight'] for _, _, data in subgraph.edges(data=True))/(num_nodes*(num_nodes - 1)/2)
                    accs.append((top_k, iia))
                    print(top_k, iia)
                
                if args.faithfulness > iia or (top_k + len(exclude_list) == n_nodes):
                    exclude_list.update(temp_top_k_nodes)
                    data_path = os.path.join(data_division_path, f'exp_{cm_id}_{label}_faithfulness_{args.faithfulness}_layer_{args.layer}.pkl')
                
                    with open(data_path, 'wb') as f:
                        pickle.dump(temp_top_k_nodes, f)
                    
                    if len(exclude_list) != n_nodes:
                        temp_labels = list(set(labels) - set(visited_labels)-set(['O']))
                        if len(temp_labels) == 0:
                            temp_labels = ['O']
                        best_label, iia = greedy_selection_of_cm(temp_labels, exclude_list, args.low_rank_dimension, args.layer, args.results_path)
                        visited_labels.append(best_label)
                        label = best_label

                        print(visited_labels)
                        print(best_label, iia)

                    else:
                        temp_labels = list(set(labels) - set(visited_labels))
                        visited_labels = visited_labels + temp_labels
                        print(visited_labels)

                    break

            data_path = os.path.join(top_k_path, f'exp_{cm_id}_{label}_faithfulness_{args.faithfulness}_layer_{args.layer}.txt')

            with open(data_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['top_k', 'iia'])
                for top_k, iia in accs:
                    writer.writerow([top_k, iia])
            
            label_enc[str(cm_id)] = visited_labels
    
    with open(f"label_enc_binary_faithfulness_{args.faithfulness}_lrd_{args.low_rank_dimension}_layer_{args.layer}.json", "w") as f:
        json.dump(label_enc, f, indent=2)


def main():
    
    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--results_path', type=str, default='results/', help='path to the results folder where you have saved the graphs')
    parser.add_argument('--layer', type=int, default=0, help='layer corresponding to the graphs obtained that you want to analyse')
    parser.add_argument('--faithfulness', type=lambda x: (float(x) if 0.0 <= float(x) <= 1.0 else parser.error(f"--faithfulness {x} not in range [0.0, 1.0]")))
    parser.add_argument('--low_rank_dimension', type=int, default=256, help='low_rank_dimension corresponding to the graphs obtained that you want to analyse')
    parser.add_argument('--top_k', type=int, default=100, help='get the top 100 nodes representing the best each intervenable model data')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    set_seed(args.seed)

    args.results_path = os.path.join(args.results_path)
    if not os.path.exists(args.results_path):
        raise argparse.ArgumentTypeError("Invalid results_path. Path does not exist.")
    
    data_division_path = os.path.join(args.results_path, 'divided_data')
    os.makedirs(data_division_path, exist_ok=True)

    top_k_path = os.path.join(args.results_path, 'top_k_results')
    os.makedirs(top_k_path, exist_ok=True)

    cliques_info_path = os.path.join(args.results_path, 'cliques_info')
    os.makedirs(cliques_info_path, exist_ok=True)

    '''
    all_labels_list = [
        {
            '1': ['(X)+Y+Z', '(X+Y+Z)'],
            '7': ['(X)+Y+Z', '(X+Y)+Z', '(X+Y+Z)'],
            '9': ['(X)+Y+Z', '(X+Z)+Y', '(X+Y+Z)'],
            '13': ['(X)+Y+Z', 'X+(Y)+Z', 'X+Y+(Z)', '(X+Y)+Z', '(X+Z)+Y', 'X+(Y+Z)', '(X+Y+Z)']
        },
        {
            '2': ['X+(Y)+Z', '(X+Y+Z)'], 
            '8': ['X+(Y)+Z', '(X+Y)+Z', '(X+Y+Z)'],
            '11': ['X+(Y)+Z', 'X+(Y+Z)', '(X+Y+Z)']
        },
        {
            '3': ['X+Y+(Z)', '(X+Y+Z)'],
            '10': ['X+Y+(Z)', '(X+Z)+Y', '(X+Y+Z)'],
            '12': ['X+Y+(Z)', 'X+(Y+Z)', '(X+Y+Z)']
        },
        {
            '4': ['(X+Y)+Z', '(X+Y+Z)'],
        },
        {
            '5': ['(X+Z)+Y', '(X+Y+Z)'],
        },
        {
            '6': ['X+(Y+Z)', '(X+Y+Z)']
        }
    ]
    '''

    all_labels_list = [
        {
            '1': ['OP1', 'O'],
            '2': ['OP2', 'O'],
            '3': ['OP3', 'O'],
            '4': ['X', 'O'],
            '5': ['Y', 'O'],
            '6': ['B', 'O'],

            '7': ["X'", 'O'],
            '8': ["Y'", 'O'],
            '9': ['Q', 'O'],
            '10': ['V', 'O'],
            '11': ['W', 'O'],
            '12': ["B'", 'O'],

            '13': ["X", "X'", "O"],
            '14': ["OP2", "X'", "O"],

            '15': ["Y", "Y'", "O"],
            '16': ["OP3", "Y'", "O"],

            '17': ["OP1", "B'", "O"],
            '18': ["B", "B'", "O"],

            '19': ["OP1", 'V', 'O'],
            '20': ["OP2", 'V', 'O'],
            '21': ["X", 'V', 'O'],

            '22': ["OP1", 'W', 'O'],
            '23': ["OP3", 'W', 'O'],
            '24': ["Y", 'W', 'O'],

            '25': ["OP2", 'Q', 'O'],
            '26': ["X", 'Q', 'O'],
            '27': ["B", 'Q', 'O'],
            '28': ["OP3", 'Q', 'O'],
            '29': ["Y", 'Q', 'O'],
            '30': ["OP1", "OP2", "X", "B", "OP3", "Y", "X'", "Y'", "B'", "Q", "V", "W", "O"]
        }
    ]

    # all_faithfulness = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6]

    n_nodes = 64

    with Pool(processes=os.cpu_count()) as pool:
        pool.starmap(run_process, [(all_labels, args, data_division_path, top_k_path, n_nodes) for all_labels in all_labels_list])
                
if __name__ =="__main__":
    main()