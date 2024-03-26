import random
import json
import os
import matplotlib.pyplot as plt

def randNum(lower=1, upper=10):
    number = random.randint(lower, upper)
    return number

def arithmetic_input_sampler():
    A = randNum()
    B = randNum()
    C = randNum()
    return {"X":A, "Y":B, "Z":C}

def save_results(results_path, report, layer, exp_id, train_id, test_id):
    file_name = f'{train_id}_report_layer_{layer}_tkn_{exp_id}.json'
    directory = os.path.join(results_path, f'results_{test_id}')
    os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, file_name)
    with open(full_path, 'w') as json_file:
        json.dump(report, json_file)


def visualize_per_trained_model(results_path, save_dir_path, n_layers, train_id, experiment_id, arithmetic_family):
            
    for test_id, model_info in arithmetic_family.causal_models.items():
        
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