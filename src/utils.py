import random
import json
import os
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.cm import ScalarMappable
import torch

def randNum(lower=1, upper=10):
    number = random.randint(lower, upper)
    return number

def construct_arithmetic_input(data):
    x,y,z = data
    return {"X":x, "Y":y, "Z":z}

def arithmetic_input_sampler():
    A = randNum()
    B = randNum()
    C = randNum()
    return {"X":A, "Y":B, "Z":C}

def redundancy_input_sampler():
    A = randNum()
    B = randNum()
    C = randNum()
    return {"X1":A, "X2":A, "X3":A, "Y":B, "Z":C}

def filter_by_max_length(list_of_lists):
    max_len = max(len(sublist) for sublist in list_of_lists)
    return [sublist for sublist in list_of_lists if len(sublist) == max_len]

def save_results(results_path, report, layer, exp_id, train_id, test_id):
    file_name = f'{train_id}_report_layer_{layer}_tkn_{exp_id}.json'
    directory = os.path.join(results_path, f'results_{test_id}')
    os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, file_name)
    with open(full_path, 'w') as json_file:
        json.dump(report, json_file)

def sanity_check_visualization(results_path, save_dir_path, n_layers, train_id, experiment_id, arithmetic_family):
    data = {}
    for test_id, model_info in arithmetic_family.causal_models.items():
        label = model_info['label']
        accuracies = []
        for layer in range(n_layers):
            file_name = f'{train_id}_report_layer_{layer}_tkn_{experiment_id}.json'
            directory = os.path.join(results_path, f'results_{test_id}')
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as json_file:
                accuracies.append(json.load(json_file)['accuracy'])
        data[label] = accuracies

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10(range(len(data))) 

    for i, (label, accuracies) in enumerate(data.items()):
        ax.plot(range(n_layers), accuracies, marker='o', linestyle='-', 
                linewidth=1.5, color=colors[i], label=label, alpha=0.8)  

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("IIA", fontsize=10)
    ax.set_xticks(range(n_layers))
    ax.set_xlim([-0.5, n_layers - 0.5])
    ax.grid(axis='y', linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=10)

    save_file_name = f'{train_id}_IIA_per_layer_targeting_[0,1,2,3,4,5]_{experiment_id}.png'
    file_path = os.path.join(save_dir_path, save_file_name)
    plt.savefig(file_path, dpi=300)
    plt.close()



def empirical_visualization(results_path, save_dir_path, n_layers, train_id, experiment_id, label):

    # Data Collection and Preparation
    accuracies = []  # Changed variable name for clarity
    for layer in range(n_layers):
        file_name = f'{train_id}_report_layer_{layer}_tkn_{experiment_id}.json'
        directory = os.path.join(results_path, f'results_{train_id}')
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r') as json_file:
            accuracies.append(json.load(json_file)['accuracy'])

    # Plotting (Improved for research papers)
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted figure size for better fit in papers
    ax.plot(range(n_layers), accuracies, marker='o', linestyle='-', linewidth=1.5, color='blue', alpha=0.8)

    # Enhanced Plot Configuration
    ax.set_title(f"IIA when Targeting Tokens [0, 1, 2, 3, 4, 5], Exp. {experiment_id}\nTrained on {label}", fontsize=12)  # Reduced font size for space
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("IIA (Accuracy)", fontsize=10)
    ax.set_xticks(range(n_layers))
    ax.set_xlim([-0.5, n_layers - 0.5])
    ax.grid(axis='y', linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Saving (High resolution for publication)
    save_file_name = f'{train_id}_IIA_per_layer_targeting_[0,1,2,3,4,5]_{experiment_id}.png'
    file_path = os.path.join(save_dir_path, save_file_name)
    plt.savefig(file_path, dpi=300)  # Higher DPI for clearer images
    plt.close()  # Ensure plot is closed to avoid display issues


def visualize_graph(graph_encoding, label=''):
    G = nx.from_numpy_matrix(graph_encoding.numpy(), create_using=nx.DiGraph)

    edge_weights = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    edge_colors = [d['weight'] for u, v, d in G.edges(data=True) if d['weight'] > 0]

    if edge_colors == []:
        edge_colors = [0]

    cmap = plt.cm.get_cmap('viridis')
    norm = plt.Normalize(min(edge_colors), max(edge_colors))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}

    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos,
                    node_color='lightblue',
                    with_labels=True,
                    edgelist=edge_weights,
                    edge_color=edge_colors,
                    width=2,
                    edge_cmap=cmap)
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.colorbar(sm, ticks=range(1, 4), label='Different Models encodings') 
    plt.title(f'Datapoints and their edges')
    plt.savefig(f'directed_graph_{label}.png')
    plt.close()

def get_average_iia_per_low_rank_dimension(n_layers, cm_id, task_results_path):
    best_lrd = 0
    best_average = 0

    for lrd in [64, 128, 256, 768, 4608]:
        sum_iia = 0
        for layer in range(n_layers):
            file_name = f'{cm_id}_report_layer_{layer}_tkn_{lrd}.json'
            directory = os.path.join(task_results_path, f'results_{cm_id}')
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as json_file:
                report_dict = json.load(json_file)
                sum_iia += report_dict['accuracy']
        average_iia = sum_iia / n_layers
        if best_average < average_iia:
            best_average = average_iia
            best_lrd = lrd
    
    print(f'Best low_rank_dimension for model {cm_id} is {best_lrd} with an iia average of {best_average}')


def merge_iia_graphs(n_layers, graph_size, save_graphs_path):
    best_graphs = {}
    for layer in range(n_layers):
    # for layer in [0]:
        best_graphs[layer] = torch.zeros(graph_size, graph_size)

        for i in range(graph_size):
            for j in range(graph_size):
                    best_acc = 0
                    best_model = 0
                    for id, cm_accs in G.items():
                        if cm_accs[layer][i][j] > best_acc:
                            best_acc = cm_accs[layer][i][j]
                            best_model = id

                    best_graphs[layer][i][j] = best_model
    
        # save graph
        graph_path = os.path.join(save_graphs_path, f'graph_{layer}.pt')
        torch.save(best_graphs[layer], graph_path)