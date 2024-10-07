import sys, os
sys.path.append(os.path.join('..', '..'))
import argparse
from pyvene import set_seed
import torch
import networkx as nx
import csv
import pickle

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

    all_labers = [
                  ['(X)+Y+Z', '(X+Y+Z)'], ['X+(Y)+Z', '(X+Y+Z)'], ['X+Y+(Z)', '(X+Y+Z)'],
                  ['(X+Y)+Z', '(X+Y+Z)'], ['(X+Z)+Y', '(X+Y+Z)'], ['X+(Y+Z)', '(X+Y+Z)'],
                  ['(X)+Y+Z', '(X+Y)+Z', '(X+Y+Z)'],
                  ['X+(Y)+Z', '(X+Y)+Z', '(X+Y+Z)'],
                  ['(X)+Y+Z', '(X+Z)+Y', '(X+Y+Z)'],
                  ['X+Y+(Z)', '(X+Z)+Y', '(X+Y+Z)'],
                  ['X+(Y)+Z', 'X+(Y+Z)', '(X+Y+Z)'],
                  ['X+Y+(Z)', 'X+(Y+Z)', '(X+Y+Z)'],
                  ['(X)+Y+Z', 'X+(Y)+Z', 'X+Y+(Z)', '(X+Y)+Z', '(X+Z)+Y', 'X+(Y+Z)', '(X+Y+Z)']]
    
    # all_faithfulness = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6]

    for cm_id, labels in enumerate(all_labers):

        print(labels)

        exclude_list = set()

        for label in labels:
            accs = []

            print(f'Loading graph for lrd {args.low_rank_dimension}, layer {args.layer}, model {label}')
            graph_path = os.path.join(args.results_path, f'graphs/{label}_graph_{args.low_rank_dimension}_{args.layer}.pt')
            graph = torch.load(graph_path)
            graph.fill_diagonal_(0)

            print('Constructing graph..')
            G = nx.from_numpy_array(graph.numpy())

            node_degrees = dict(G.degree())
            sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)

            limits = [num for num in range(10, 1001 - len(exclude_list), 20)]

            if label == '(X+Y+Z)':
                limits = [1000 - len(exclude_list)]

            for top_k in limits:

                args.top_k = top_k

                if label == '(X+Y+Z)':

                    subgraph = G.subgraph(set(G.nodes()) - exclude_list)
                    num_nodes = subgraph.number_of_nodes()
                    iia = sum(data['weight'] for _, _, data in subgraph.edges(data=True))/(num_nodes*(num_nodes - 1)/2)
                    accs.append((top_k, iia))
                    print(top_k, iia)

                    data_path = os.path.join(data_division_path, f'exp_{cm_id}_{label}_faithfulness_{args.faithfulness}_layer_{args.layer}.pkl')
                
                    with open(data_path, 'wb') as f:
                        pickle.dump(temp_top_k_nodes, f)
                    
                    continue

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

                if args.faithfulness <= iia or label == '(X+Y+Z)':
                    temp_top_k_nodes = top_k_nodes

                    subgraph = G.subgraph(top_k_nodes)
                    num_nodes = subgraph.number_of_nodes()
                    iia = sum(data['weight'] for _, _, data in subgraph.edges(data=True))/(num_nodes*(num_nodes - 1)/2)
                    accs.append((top_k, iia))
                    print(top_k, iia)
                
                if args.faithfulness > iia:
                    exclude_list.update(temp_top_k_nodes)

                    data_path = os.path.join(data_division_path, f'exp_{cm_id}_{label}_faithfulness_{args.faithfulness}_layer_{args.layer}.pkl')
                
                    with open(data_path, 'wb') as f:
                        pickle.dump(temp_top_k_nodes, f)

                    break

            data_path = os.path.join(top_k_path, f'exp_{cm_id}_{label}_faithfulness_{args.faithfulness}_layer_{args.layer}.txt')
            
            with open(data_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['top_k', 'iia'])
                for top_k, iia in accs:
                    writer.writerow([top_k, iia])
                
if __name__ =="__main__":
    main()