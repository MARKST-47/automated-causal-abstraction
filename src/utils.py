import random
import json
import os
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.cm import ScalarMappable
import torch
import pandas as pd
import pickle

def randNum(lower=1, upper=10):
    number = random.randint(lower, upper)
    return number

def randBool():
    return random.choice([True, False])

def construct_arithmetic_input(data):
    x,y,z = data
    return {"X":x, "Y":y, "Z":z}

def de_morgan_sampler():
    X = randBool()
    Y = randBool()
    return {"X": X, "Y": Y}

def arithmetic_input_sampler():
    A = randNum()
    B = randNum()
    C = randNum()
    return {"X":A, "Y":B, "Z":C}

def ruled_arithmetic_input_sampler():

    while True:
        A = randNum()
        B = randNum()
        C = randNum()
        if C > 7:
            return {"X":A, "Y":B, "Z":C}

def iia_based_sampler(data_path, arrangements):
    with open(data_path, 'rb') as file:
        data_ids = pickle.load(file)
    
    random_id = random.choice(data_ids)
    print(construct_arithmetic_input(arrangements[random_id]))
    return construct_arithmetic_input(arrangements[random_id])

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
                linewidth=1.5, color=colors[i], label=f'$M_{{{i+1}}}$', alpha=0.8)  

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("IIA", fontsize=10)
    ax.set_xticks(range(n_layers))
    ax.set_xlim([-0.5, n_layers - 0.5])
    ax.grid(axis='y', linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=10)

    save_file_name = f'{train_id}_IIA_per_layer_targeting_[0,1,2,3,4,5]_{experiment_id}.pdf'
    file_path = os.path.join(save_dir_path, save_file_name)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

def evaluation_visualization(results_path, save_dir_path, n_layers, cm_id, experiment_id):
    data = {}

    # evals = ['original', 'iia_based', 'iia_based_new']
    evals = ['original', 'iia_based_(X+Y)+Z', 'iia_based_(X)+Y+Z', 'iia_based_(X+Y+Z)']

    for name in evals:
        dir_path = os.path.join(results_path, name)

        if name == 'sanity_check':
            name = 'original'

        accuracies = []
        for layer in range(n_layers):
            file_name = f'{cm_id}_report_layer_{layer}_tkn_{experiment_id}.json'
            directory = os.path.join(dir_path, f'results_{cm_id}')
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as json_file:
                accuracies.append(json.load(json_file)['accuracy'])
        data[name] = accuracies

    fig, ax = plt.subplots(figsize=(10, 7))
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

    save_file_name = f'{cm_id}_iia_based_{experiment_id}.pdf'
    file_path = os.path.join(save_dir_path, save_file_name)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

def evaluation_visualization_combined(results_path, save_dir_path, n_layers, arithmetic_family, experiment_id):
    
    cm_data = {}
    # evals = ['original', 'iia_based', 'iia_based_new']
    evals = ['original', 'iia_based_(X+Y)+Z', 'iia_based_(X)+Y+Z', 'iia_based_(X+Y+Z)']

    for cm_id, model_info in arithmetic_family.causal_models.items():
        
        if cm_id == 4:
            continue
        
        data = {}
        for name in evals:
            dir_path = os.path.join(results_path, name)

            if name == 'sanity_check':
                name = 'original'

            
            accuracies = []
            for layer in range(n_layers):
                file_name = f'{cm_id}_report_layer_{layer}_tkn_{experiment_id}.json'
                directory = os.path.join(dir_path, f'results_{cm_id}')
                file_path = os.path.join(directory, file_name)
                with open(file_path, 'r') as json_file:
                    accuracies.append(json.load(json_file)['accuracy'])
            data[name] = accuracies
        
        cm_data[cm_id] = data

    # Aggregate with model IDs
    max_accuracy_dict = {}  # Initialize the result dictionary
    for name in evals:
        max_accuracy_dict[name] = {}  # Create a dictionary for each condition
        for layer in range(n_layers):
            max_accuracy = 0
            best_model_id = None
            for cm_id, model_data in cm_data.items():
                if model_data[name][layer] >= max_accuracy:
                    max_accuracy = model_data[name][layer]
                    best_model_id = cm_id
            max_accuracy_dict[name][layer] = (max_accuracy, best_model_id)

    # Create a new figure and axes object
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot data for each condition
    for name, layer_data in max_accuracy_dict.items():
        layers = list(layer_data.keys())
        accuracies = [acc for acc, _ in layer_data.values()]
        model_ids = [model_id for _, model_id in layer_data.values()]

        ax.plot(layers, accuracies, marker='o', linestyle='-', label=name)

        # Add text labels with model IDs next to the points
        for layer, accuracy, model_id in zip(layers, accuracies, model_ids):
            ax.text(layer, accuracy, f'$M_{{{model_id}}}$', va='bottom', ha='left')

    # Customize the plot
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("IIA", fontsize=10)
    ax.set_xticks(range(n_layers))
    ax.set_xlim([-0.5, n_layers - 0.5])
    ax.grid(axis='y', linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=10)

    save_file_name = f'iia_based_{experiment_id}.pdf'
    file_path = os.path.join(save_dir_path, save_file_name)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

def empirical_visualization(results_path, save_dir_path, n_layers, train_id, experiment_id, label):

    data = {}
    for experiment_id in [64,128,256]:
        # Data Collection and Preparation
        accuracies = []  # Changed variable name for clarity
        for layer in range(n_layers):
            file_name = f'{train_id}_report_layer_{layer}_tkn_{experiment_id}.json'
            directory = os.path.join(results_path, f'results_{train_id}')
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as json_file:
                accuracies.append(json.load(json_file)['accuracy'])
            data[experiment_id] = accuracies
    # Plotting (Improved for research papers)
    # fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted figure size for better fit in papers
    # ax.plot(range(n_layers), accuracies, marker='o', linestyle='-', linewidth=1.5, color='blue', label=experiment_id, alpha=0.8)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10(range(len(data))) 

    for i, (label, accuracies) in enumerate(data.items()):
        ax.plot(range(n_layers), accuracies, marker='o', linestyle='-', 
                linewidth=1.5, color=colors[i], label=label, alpha=0.8)  

    # Enhanced Plot Configuration
    # ax.set_title(f"IIA when Targeting Tokens [0, 1, 2, 3, 4, 5], Exp. {experiment_id}\nTrained on {label}", fontsize=12)  # Reduced font size for space
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("IIA", fontsize=10)
    ax.set_xticks(range(n_layers))
    ax.set_xlim([-0.5, n_layers - 0.5])
    ax.grid(axis='y', linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=8)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, frameon=False, title="Low Rank Dimension")
    ax.legend(fontsize=10)
    # Saving (High resolution for publication)
    save_file_name = f'{train_id}_IIA_per_layer_targeting_[0,1,2,3,4,5].pdf'
    file_path = os.path.join(save_dir_path, save_file_name)
    
    plt.savefig(file_path, dpi=300, bbox_inches="tight")  # Higher DPI for clearer images
    plt.close()  # Ensure plot is closed to avoid display issues

def compare_intermediate_vs_simple(arithmetic_results_path, simple_results_path, save_dir_path, n_layers, arithmetic_family, experiment_id):
    options = [[1,2], [1,3], [2,3]]
    # options = [1,2,3,4]
    # data = {}
    for cm_id, model_info in arithmetic_family.causal_models.items():
        if cm_id == 1:
            add_label = 'X+Y'
        elif cm_id == 2:
            add_label = 'X+Z'
        else:
            add_label = 'Y+Z'
        label = f'$M_{{{cm_id}}}$, P={add_label}'
        data = {}
        simple_ids = options[cm_id-1]
        # simple_ids = options
        accuracies = []
        for layer in range(n_layers):
            file_name = f'{cm_id}_report_layer_{layer}_tkn_{experiment_id}.json'
            directory = os.path.join(arithmetic_results_path, f'results_{cm_id}')
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as json_file:
                accuracies.append(json.load(json_file)['accuracy'])
        data[label] = accuracies

        # if cm_id == 3:
        for id in simple_ids:
            if id == 1:
                add_label = 'X'
            elif id == 2:
                add_label = 'Y'
            else:
                add_label = 'Z'
            label = f'$M_{{{id+3}}}$, P={add_label}'
            accuracies = []
            for layer in range(n_layers):
                file_name = f'{id}_report_layer_{layer}_tkn_{experiment_id}.json'
                directory = os.path.join(simple_results_path, f'results_{id}')
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

        save_file_name = f'{cm_id}_vs_simple_comapring_IIA_per_layer_{experiment_id}.pdf'
        # save_file_name = f'solving_arithmetic_task_{experiment_id}.pdf'
        file_path = os.path.join(save_dir_path, save_file_name)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

def visualize_graph(G, label=''):
    
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