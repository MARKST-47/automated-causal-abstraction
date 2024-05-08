import random
import json
import os
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.cm import ScalarMappable

def randNum(lower=1, upper=10):
    number = random.randint(lower, upper)
    return number

def arithmetic_input_sampler():
    A = randNum()
    B = randNum()
    C = randNum()
    return {"X":A, "Y":B, "Z":C}

def biased_sampler_1():
    A = randNum(lower=1, upper=3)
    B = randNum(lower=1, upper=3)
    C = randNum()
    return {"X":A, "Y":B, "Z":C}

def biased_sampler_2():
    A = randNum(lower=7, upper=10)
    B = randNum()
    C = randNum(lower=7, upper=10)
    return {"X":A, "Y":B, "Z":C}

def biased_sampler_3():
    A = randNum()
    B = randNum(lower=4, upper=6)
    C = randNum(lower=4, upper=6)
    return {"X":A, "Y":B, "Z":C}

def redundancy_input_sampler():
    A = randNum()
    B = randNum()
    C = randNum()
    return {"X1":A, "X2":A, "X3":A, "Y":B, "Z":C}

def save_results(results_path, report, layer, exp_id, train_id, test_id):
    file_name = f'{train_id}_report_layer_{layer}_tkn_{exp_id}.json'
    directory = os.path.join(results_path, f'results_{test_id}')
    os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, file_name)
    with open(full_path, 'w') as json_file:
        json.dump(report, json_file)


def visualize_per_trained_model(results_path, save_dir_path, n_layers, train_id, experiment_id, arithmetic_family, causal_model_type='arithmetic'):
            
    for test_id, model_info in arithmetic_family.causal_models.items():

        if causal_model_type == 'simple':
            if test_id != train_id:
                continue
        
        label = model_info['label']

        cm = []
        report_dicts = []

        for layer in range(n_layers):
            file_name = f'{train_id}_report_layer_{layer}_tkn_{experiment_id}.json'
            directory = os.path.join(results_path, f'results_{test_id}')
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as json_file:
                report_dict = json.load(json_file)
                report_dicts.append(report_dict)

        for layer, report_dict in enumerate(report_dicts, start=1):
            cm.append(report_dict['accuracy'])
        
        plt.scatter(range(n_layers), cm)
        plt.plot(range(n_layers), cm, label=label)
        plt.xticks(range(int(min(plt.xticks()[0])), int(max(plt.xticks()[0])) + 1))
        plt.xlabel('layer')
        plt.ylabel('IIA')

    plt.title(f'IIA when targeting tokens [0,1,2,3,4,5], {experiment_id}, trained on {arithmetic_family.get_label_by_id(train_id)}')
    plt.rcParams.update({'figure.autolayout': True})
    plt.legend()
    plt.tight_layout()
    
    save_file_name = f'{train_id}_IIA_per_layer_targeting_[0,1,2,3,4,5]_{experiment_id}.png'
    file_path = os.path.join(save_dir_path, save_file_name)
    plt.savefig(file_path)
    plt.close()

def visualize_per_model(results_path, save_dir_path, n_layers, train_id, experiment_id, label):

    cm = []
    report_dicts = []

    for layer in range(n_layers):
        file_name = f'{train_id}_report_layer_{layer}_tkn_{experiment_id}.json'
        directory = os.path.join(results_path, f'results_{train_id}')
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r') as json_file:
            report_dict = json.load(json_file)
            report_dicts.append(report_dict)

    for layer, report_dict in enumerate(report_dicts, start=1):
        cm.append(report_dict['accuracy'])
    
    plt.scatter(range(n_layers), cm)
    plt.plot(range(n_layers), cm, label=label)
    plt.xticks(range(int(min(plt.xticks()[0])), int(max(plt.xticks()[0])) + 1))
    plt.xlabel('layer')
    plt.ylabel('IIA')

    plt.title(f'IIA when targeting tokens [0,1,2,3,4,5], {experiment_id}, {label}')
    plt.rcParams.update({'figure.autolayout': True})
    plt.legend()
    plt.tight_layout()
    
    save_file_name = f'{train_id}_IIA_per_layer_targeting_[0,1,2,3,4,5]_{experiment_id}.png'
    file_path = os.path.join(save_dir_path, save_file_name)
    plt.savefig(file_path)
    plt.close()

def visualize_model_all_tokens(results_path, save_dir_path, n_layers, train_id, label):

    for token in [0,1,2,3,4,5]:
        cm = []
        report_dicts = []
        for layer in range(n_layers):
            file_name = f'{train_id}_report_layer_{layer}_tkn_{token}.json'
            directory = os.path.join(results_path, f'results_{train_id}')
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as json_file:
                report_dict = json.load(json_file)
                report_dicts.append(report_dict)

        for layer, report_dict in enumerate(report_dicts, start=1):
            cm.append(report_dict['accuracy'])
        
        plt.scatter(range(n_layers), cm)
        plt.plot(range(n_layers), cm, label=token)
        plt.xticks(range(int(min(plt.xticks()[0])), int(max(plt.xticks()[0])) + 1))
        plt.xlabel('layer')
        plt.ylabel('IIA')
    
    plt.title(f'IIA when targeting tokens one by one, causal model {label}')
    plt.rcParams.update({'figure.autolayout': True})
    plt.legend()
    plt.tight_layout()
    
    save_file_name = f'{train_id}_IIA_per_layer_targeting_tokens_{label}.png'
    file_path = os.path.join(save_dir_path, save_file_name)
    plt.savefig(file_path)
    plt.close()

def visualize_simple_per_token(results_path, save_dir_path, n_layers, token, subspace, causal_model_family):
            
    for test_id, model_info in causal_model_family.causal_models.items():
        
        label = model_info['label']

        cm = []
        report_dicts = []

        for layer in range(n_layers):
            file_name = f'{token}_report_layer_{layer}_tkn_{subspace}.json'
            directory = os.path.join(results_path, f'results_{test_id}')
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as json_file:
                report_dict = json.load(json_file)
                report_dicts.append(report_dict)

        for layer, report_dict in enumerate(report_dicts, start=1):
            cm.append(report_dict['accuracy'])
        
        plt.scatter(range(n_layers), cm)
        plt.plot(range(n_layers), cm, label=label)
        plt.xticks(range(int(min(plt.xticks()[0])), int(max(plt.xticks()[0])) + 1))
        plt.xlabel('layer')
        plt.ylabel('IIA')

    plt.title(f'IIA when targeting token {token}, {subspace}, trained on {causal_model_family.get_label_by_id(test_id)}')
    plt.rcParams.update({'figure.autolayout': True})
    plt.legend()
    plt.tight_layout()
    
    save_file_name = f'{token}_targeted_IIA_per_layer_{subspace}.png'
    file_path = os.path.join(save_dir_path, save_file_name)
    plt.savefig(file_path)
    plt.close()

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


def visualize_connected_components(matrix, causal_model_family, title_label=''):
    G = nx.Graph()

    for i in range(matrix.shape[0]):
        G.add_node(i)

    # undirected graph edge-label construction
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[0]):
            if matrix[i, j] != 0:
                G.add_edge(i, j, label=matrix[i, j].item()) # label means the causal model

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

        subgraphs[causal_model_family.get_label_by_id(label)] = subgraph

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
        maximal_cliques[label] = cliques
        pos = positions[label]

        # visualizing the subgraph
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
        colors = ['r', 'g', 'b', 'c', 'm', 'y'] # len is 7
        
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
    plt.savefig(f'connected_component_visualization_{title_label}.png')
    plt.close()
    return maximal_cliques