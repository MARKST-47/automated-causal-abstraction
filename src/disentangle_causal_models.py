import sys, os
sys.path.append(os.path.join('..', '..'))

from sklearn.metrics import classification_report
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import random
from tqdm import tqdm, trange
from pyvene import count_parameters, set_seed
import argparse
from causal_models import ArithmeticCausalModels
import numpy as np
import json
from utils import arithmetic_input_sampler, visualize_connected_components

from transformers import (GPT2Tokenizer,
                          GPT2Config,
                          GPT2ForSequenceClassification)

from pyvene import (
    IntervenableModel,
    IntervenableConfig,
    LowRankRotatedSpaceIntervention
)

# temporary import
# from my_pyvene.models.intervenable_base import IntervenableModel
# from my_pyvene.models.configuration_intervenable_model import IntervenableConfig
# from my_pyvene.models.interventions import LowRankRotatedSpaceIntervention

def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path)
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def batched_random_sampler(data, batch_size):
    batch_indices = [_ for _ in range(int(len(data) / batch_size))]
    random.shuffle(batch_indices)
    for b_i in batch_indices:
        for i in range(b_i * batch_size, (b_i + 1) * batch_size):
            yield i

def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        total_count += 1
        correct_count += eval_pred == eval_label
    accuracy = float(correct_count) / float(total_count)
    return {"accuracy": accuracy}
    
def calculate_loss(logits, labels):
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, 28) # 28 is the number of classes
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device).long()
    loss = loss_fct(shift_logits, shift_labels)

    return loss

def intervention_id(intervention):
    if "P" in intervention:
        return 0
    
def tokenizePrompt(input):
    tokenizer = load_tokenizer("gpt2")
    prompt = f"{input['X']}+{input['Y']}+{input['Z']}="
    return tokenizer.encode(prompt, padding=True, return_tensors='pt')

def eval_intervenable(intervenable, eval_data, batch_size, low_rank_dimension, min_class_value=3):
    # eval on all data
    eval_labels = []
    eval_preds = []
    with torch.no_grad():
        epoch_iterator = tqdm(DataLoader(eval_data, batch_size), desc=f"Test")
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")
            inputs["input_ids"] = inputs["input_ids"].squeeze()
            inputs["source_input_ids"] = inputs["source_input_ids"].squeeze(2)
            b_s = inputs["input_ids"].shape[0]
            _, counterfactual_outputs = intervenable(
                {"input_ids": inputs["input_ids"]},
                [{"input_ids": inputs["source_input_ids"][:, 0]}],
                {"sources->base": [0,1,2,3,4,5]},
                subspaces=[
                    [[_ for _ in range(low_rank_dimension)]] * batch_size
                ]
            )

            eval_labels += [inputs["labels"].type(torch.long).squeeze() - min_class_value]
            eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1)]
    report = classification_report(torch.cat(eval_labels).cpu(), torch.cat(eval_preds).cpu(), output_dict=True) # get the IIA
    return report['accuracy']

def eval_one_point(intervenable, inputs, batch_size, low_rank_dimension, min_class_value=3):
    # eval on all data
    eval_labels = []
    eval_preds = []
    with torch.no_grad():
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to("cuda")
        
        inputs["input_ids"] = inputs["input_ids"]
        inputs["source_input_ids"] = inputs["source_input_ids"].squeeze(2)
        
        b_s = inputs["input_ids"].shape[0]
        _, counterfactual_outputs = intervenable(
            {"input_ids": inputs["input_ids"]},
            [{"input_ids": inputs["source_input_ids"][:, 0]}],
            {"sources->base": [0,1,2,3,4,5]},
            subspaces=[
                [[_ for _ in range(low_rank_dimension)]] * batch_size
            ]
        )

        eval_labels += [inputs["labels"].type(torch.long).squeeze() - min_class_value]
        eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1).squeeze()]

        _, counterfactual_outputs = intervenable(
            {"input_ids": inputs["input_ids"]},
            [{"input_ids": inputs["input_ids"]}],
            {"sources->base": [0,1,2,3,4,5]},
            subspaces=[
                [[_ for _ in range(low_rank_dimension)]] * batch_size
            ]
        )

        eval_labels += [inputs["labels"].type(torch.long).squeeze() - min_class_value]
        eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1).squeeze()]
    
    report = classification_report(torch.tensor(eval_labels).cpu(), torch.tensor(eval_preds).cpu(), output_dict=True) # get the IIA
    return report['accuracy']

def train_intervenable(counterfactual_data, model, layer, low_rank_dimension, batch_size, epochs, gradient_accumulation_steps, min_class_value=3):
    
    total_step = 0

    intervenable_config = IntervenableConfig({
            "layer": layer,
            "component": "block_output",
            "low_rank_dimension": low_rank_dimension,
            "unit":"pos",
            "max_number_of_units": 6
        },
        intervention_types=LowRankRotatedSpaceIntervention,
        model_type=type(model)
    )

    intervenable = IntervenableModel(intervenable_config, model, use_fast=True)
    intervenable.set_device("cuda")
    intervenable.disable_model_gradients()

    optimizer_params = []
    for k, v in intervenable.interventions.items():
        optimizer_params += [{"params": v[0].rotate_layer.parameters()}]

    optimizer = torch.optim.Adam(optimizer_params, lr=0.01)

    print('DAS training...')

    intervenable.model.train()
    print("intervention trainable parameters: ", intervenable.count_parameters())
    print("gpt2 trainable parameters: ", count_parameters(intervenable.model))
    train_iterator = trange(0, int(epochs), desc="Epoch")

    for epoch in train_iterator:
        torch.cuda.empty_cache()
        epoch_iterator = tqdm(
            DataLoader(
                counterfactual_data,
                batch_size=batch_size,
                sampler=batched_random_sampler(counterfactual_data, batch_size),
            ),
            desc=f"Epoch: {epoch}",
            position=0,
            leave=True,
        )
        
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")
            inputs["input_ids"] = inputs["input_ids"].squeeze()
            inputs["source_input_ids"] = inputs["source_input_ids"].squeeze(2)
            b_s = inputs["input_ids"].shape[0]
            _, counterfactual_outputs = intervenable(
                {"input_ids": inputs["input_ids"]},
                [{"input_ids": inputs["source_input_ids"][:, 0]}],
                {"sources->base": [0,1,2,3,4,5]},
                subspaces=[
                    [[_ for _ in range(low_rank_dimension)]] * batch_size # taking half of the repr. and rotating it
                ]
            )

            eval_metrics = compute_metrics(
                counterfactual_outputs[0].argmax(1), inputs["labels"].squeeze() - min_class_value
            )

            # loss and backprop
            loss = calculate_loss(counterfactual_outputs.logits, inputs["labels"] - min_class_value)
            loss_str = round(loss.item(), 2)
            epoch_iterator.set_postfix({"loss": loss_str, "acc": eval_metrics["accuracy"]})

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()

            if total_step % gradient_accumulation_steps == 0:
                optimizer.step()
                intervenable.set_zero_grad()
            total_step += 1
    
    return intervenable

def decode_tensor(tensor, vocabulary):
    token_ids = tensor.squeeze().tolist()
    return {'X': int(vocabulary[token_ids[0]]), 'Y': int(vocabulary[token_ids[2]]), 'Z': int(vocabulary[token_ids[4]])}

def main():

    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--model_path', type=str, help='path to the finetuned GPT2ForSequenceClassification on the arithmetic task')
    parser.add_argument('--results_path', type=str, default='disentangling_results/', help='path to the results folder')
    parser.add_argument('--n_training', type=int, default=2560, help='number of training samples')
    parser.add_argument('--n_testing', type=int, default=256, help='number of testing samples')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='number of steps to accumulate before optimization step')
    parser.add_argument('--seed', type=int, default=43, help='experiment seed to be able to reproduce the results')
    parser.add_argument('--layer', type=int, default=0, help='layer to intervene at')
    parser.add_argument('--low_rank_dim', type=int, default=64, help='low rank dimension for rotation intervention')
    parser.add_argument('--n_runs', type=int, default=1, help='number of runs before obtaining the graph')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise argparse.ArgumentTypeError("Invalid model_path. Path does not exist.")
    
    os.makedirs(args.results_path, exist_ok=True)

    save_dir_path = os.path.join(args.results_path, 'plots')
    os.makedirs(save_dir_path, exist_ok=True)
    
    set_seed(args.seed)
    
    min_class_value = 3
    
    tokenizer = load_tokenizer('gpt2')
    # hard-coded vocabulary
    vocabulary = {'1': 16, '2': 17, '3': 18, '4': 19, '5': 20, '6': 21, '7': 22, '8': 23, '9': 24, '10': 940, '+': 10, '=': 28}
    inv_vocabulary = {v: k for k, v in vocabulary.items()}
    # vocabulary = tokenizer.get_vocab()
    model_config = GPT2Config.from_pretrained(args.model_path)
    model_config.pad_token_id = tokenizer.pad_token_id
    model = GPT2ForSequenceClassification.from_pretrained(args.model_path, config=model_config)
    model.resize_token_embeddings(len(tokenizer))

    arithmetic_family = ArithmeticCausalModels()
    
    D = [] # bases
    for _ in range(args.n_training):
        D.append(arithmetic_input_sampler())

    sources = []
    for _ in range(args.n_training):
        sources.append(arithmetic_input_sampler())

    sampled_indices = random.sample(range(len(D)), args.n_testing)
    pairs = [(D[i], sources[i]) for i in sampled_indices]
    T, test_sources = zip(*pairs) 

    print(T)
    print(test_sources)
    
    # random subset of bases
    # T = random.sample(D, args.n_testing)
    T_saved = np.array(T)
    T_path = os.path.join(args.results_path, 'testing_bases.npy')
    np.save(T_path, T_saved)

    T_sources_saved = np.array(test_sources)
    T_sources_path = os.path.join(args.results_path, 'testing_sources.npy')
    np.save(T_sources_path, T_sources_saved)

    # array of runs to average over multiple runs --> obtain run average and variance
    # R = []

    # runs to average over
    # for _ in range(args.n_runs):

    G = {}
    iia_s = {}
    iia_c = {}
    intervenable_models = {}
    
    # loop through the family of causal models
    for cm_id, model_info in arithmetic_family.causal_models.items():

        # initiate graph weighted by accuracies
        graph_encoding = torch.zeros(args.n_testing, args.n_testing)

        print('generating data for DAS...')

        # generate counterfactual data C_i
        training_counterfactual_data = model_info['causal_model'].generate_counterfactual_dataset_on_bases(
            args.n_training,
            intervention_id,
            args.batch_size,
            D,
            sources,
            device="cuda:0",
            sampler=arithmetic_input_sampler,
            inputFunction=tokenizePrompt
        )

        # training intervenable model
        layer = args.layer
        low_rank_dimension = args.low_rank_dim

        intervenable = train_intervenable(training_counterfactual_data, 
                                            model, layer, low_rank_dimension, 
                                            args.batch_size, args.epochs, 
                                            args.gradient_accumulation_steps, 
                                            min_class_value)
        
        intervenable_path = os.path.join(args.results_path, f'intervenable_{cm_id}')
        intervenable.save(intervenable_path)

        # eval on all counterfactual data C_i
        iia_c[cm_id] = eval_intervenable(intervenable, training_counterfactual_data, args.batch_size, low_rank_dimension)
        print(f"Accuracy when evaluating on the entire data: {iia_c[cm_id]}")

        # generate counterfactual data S_i
        testing_counterfactual_data = model_info['causal_model'].generate_counterfactual_dataset_on_bases(
            args.n_testing,
            intervention_id,
            args.n_testing, # batch size when testing
            T, # random subset of bases samples
            test_sources,
            device="cuda:0",
            sampler=arithmetic_input_sampler,
            inputFunction=tokenizePrompt
        )

        # eval on the whole dataset
        iia_s[cm_id] = eval_intervenable(intervenable, testing_counterfactual_data, args.n_testing, low_rank_dimension)
        intervenable_models[cm_id] = intervenable # save trained intervenable model

        # evaluate per pair of data in training data

        for i, x in enumerate(testing_counterfactual_data):
            for j, source in enumerate(test_sources):

                x['source_input_ids'] = tokenizePrompt(source).unsqueeze(0).to("cuda")
                iia = eval_one_point(intervenable, x, 1, low_rank_dimension)
                graph_encoding[i][j] = iia
                # source_input = decode_tensor(x['source_input_ids'], inv_vocabulary)
                # base_input = T[i] # equivalent to base_input = decode_tensor(x['input_ids'], inv_vocabulary)
        
        # save graph
        G[cm_id] = graph_encoding
    
    print(G)

    # R.append(G)

    # model_accs = {}
    # model_vars = {}
    # for cm_id, model_info in arithmetic_family.causal_models.items():
    #     summing = []
    #     for m in R:
    #         summing.append(m[cm_id])
    #     stacked_tensors = torch.stack(summing)
    #     average_tensor = stacked_tensors.mean(dim=0)
    #     variance_tensor = stacked_tensors.var(dim=0)

    #     model_accs[cm_id] = average_tensor
    #     model_vars[cm_id] = variance_tensor
    
    # # averaged model accuracies per runs for each causal model
    # print(model_accs)

    # variance of iias whe running n_runs times
    # print(model_vars)

    graph = torch.zeros(args.n_testing, args.n_testing)

    for i in range(args.n_testing):
        for j in range(args.n_testing):
                best_acc = 0
                best_model = 0
                for id, cm_accs in G.items():
                    if cm_accs[i][j] > best_acc:
                        best_acc = cm_accs[i][j]
                        best_model = id

                graph[i][j] = best_model
    
    # save graph
    graph_path = os.path.join(args.results_path, 'graph.pt')
    torch.save(graph, graph_path)

    # save iia_s
    iias_path = os.path.join(args.results_path, 'iia_s.json')
    with open(iias_path, 'w') as file:
        json.dump(iia_s, file)

    print(graph)
    print(iia_s)
    print(iia_c)

if __name__ =="__main__":
    main()