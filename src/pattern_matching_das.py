import sys, os
sys.path.append(os.path.join('..', '..'))

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
import random
import copy
import itertools
import numpy as np
from tqdm import tqdm, trange
import json

from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from pyvene import CausalModel
from pyvene.models.mlp.modelings_mlp import MLPConfig
from pyvene import create_mlp_classifier
from pyvene import (
    IntervenableModel,
    VanillaIntervention,
    RotatedSpaceIntervention,
    LowRankRotatedSpaceIntervention,
    IntervenableRepresentationConfig,
    IntervenableConfig,
)
from transformers import TrainingArguments, Trainer

def intervention_id(intervention):
    if "P" in intervention and "Q" in intervention:
        return 2
    if "P" in intervention:
        return 0
    if "Q" in intervention:
        return 1

def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        total_count += 1
        correct_count += (eval_pred== eval_label)
    accuracy = float(correct_count)/float(total_count)
    return {"accuracy" : accuracy}

def compute_loss(outputs, labels):
    CE = torch.nn.CrossEntropyLoss()
    return CE(outputs, labels)

def batched_random_sampler(data, batch_size):
    batch_indices = [ _ for _ in range(int(len(data)/batch_size))]
    random.shuffle(batch_indices)
    for b_i in batch_indices:
        for i in range(b_i*batch_size, (b_i + 1)*batch_size):
            yield i

def randvec(n=50, lower=-1, upper=1):
    return np.array([round(random.uniform(lower, upper), 2) for i in range(n)])

# def input_sampler():
#     A = randvec(4)
#     B = randvec(4)
#     C = randvec(4)
#     D = randvec(4)
#     x = random.randint(1,4)
#     if x == 1:
#         return {"W":A, "X":B, "Y":C, "Z":D}
#     elif x == 2:
#         return {"W":A,"X":B, "Y":A, "Z":B}
#     elif x == 3:
#         return {"W":A ,"X":B, "Y":A, "Z":C}
#     elif x == 4:
#         return {"W":A ,"X":B, "Y":C, "Z":B}

def input_sampler():
    A = randvec(4)
    B = randvec(4)
    C = randvec(4)
    D = randvec(4)
    x = random.randint(1,2)
    if x == 1:
        return {"W":A, "X":B, "Y":C, "Z":D}
    elif x == 2:
        return {"W":A,"X":B, "Y":A, "Z":B}

def train_alignable_model(alignable, train_dataset, optimizer, embedding_dim, batch_size = 640, epochs = 6):
    total_step = 0
    gradient_accumulation_steps = 1
    alignable.model.train() # train enables drop-off but no grads
    print("intervention trainable parameters: ", alignable.count_parameters())
    train_iterator = trange(
        0, int(epochs), desc="Epoch"
    )

    for epoch in train_iterator:
        epoch_iterator = tqdm(
            DataLoader(train_dataset,
                    batch_size=batch_size,
                    sampler=batched_random_sampler(train_dataset, batch_size)),
            desc=f"Epoch: {epoch}", 
            position=0, 
            leave=True
        )
        for batch in epoch_iterator:
            batch["input_ids"] = batch["input_ids"].unsqueeze(1)    
            batch["source_input_ids"] = batch["source_input_ids"].unsqueeze(2) 
            batch_size = batch["input_ids"].shape[0]
            if batch["intervention_id"][0] == 2:
                _, counterfactual_outputs = alignable(
                        {"inputs_embeds":batch["input_ids"]},
                        [{"inputs_embeds":batch["source_input_ids"][:, 0]}, 
                        {"inputs_embeds":batch["source_input_ids"][:,1]}],
                        {"sources->base": ([[[0]]*batch_size, [[0]]*batch_size], [[[0]]*batch_size, [[0]]*batch_size])},
                    subspaces=[[[_ for _ in range(0,embedding_dim*2)]]*batch_size, 
                            [[_ for _ in range(embedding_dim*2, embedding_dim*4)]]*batch_size]
                    )
            elif batch["intervention_id"][0] == 0:
                _, counterfactual_outputs = alignable(
                        {"inputs_embeds":batch["input_ids"]},
                        [{"inputs_embeds":batch["source_input_ids"][:,0]}, None],
                        {"sources->base": ([[[0]]*batch_size, None], [[[0]]*batch_size, None])},
                        subspaces=[[[_ for _ in range(0,embedding_dim*2)]]*batch_size, 
                                None]
                    )
            elif batch["intervention_id"][0] == 1:
                _, counterfactual_outputs = alignable(
                        {"inputs_embeds":batch["input_ids"]},
                        [None, {"inputs_embeds":batch["source_input_ids"][:,0]}],
                        {"sources->base": ([None, [[0]]*batch_size], [None, [[0]]*batch_size])},
                        subspaces=[None, 
                                [[_ for _ in range(embedding_dim*2, embedding_dim*4)]]*batch_size]
                    )
            eval_metrics=compute_metrics(counterfactual_outputs[0].argmax(1), batch['labels'].squeeze())

            # loss and backprop
            loss = compute_loss(
                counterfactual_outputs[0], batch["labels"].squeeze().to(torch.long)
            )

            epoch_iterator.set_postfix({'loss': loss, 'acc': eval_metrics["accuracy"]})
            
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            if total_step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    alignable.set_zero_grad()
            total_step += 1

def test_alignable_model(alignable, test_dataset, embedding_dim, batch_size = 640):
    eval_labels = []
    eval_preds = []
    with torch.no_grad():
        epoch_iterator = tqdm(DataLoader(test_dataset, batch_size), desc=f"Test")
        for step, batch in enumerate(epoch_iterator):
            for k, v in batch.items():
                if v is not None and isinstance(v, torch.Tensor):
                    batch[k] = v.to("cuda")
            batch["input_ids"] = batch["input_ids"].unsqueeze(1)    
            batch["source_input_ids"] = batch["source_input_ids"].unsqueeze(2)    
            if batch["intervention_id"][0] == 2:
                _, counterfactual_outputs = alignable(
                        {"inputs_embeds":batch["input_ids"]},
                        [{"inputs_embeds":batch["source_input_ids"][:, 0]}, 
                        {"inputs_embeds":batch["source_input_ids"][:,1]}],
                        {"sources->base": ([[[0]]*batch_size, [[0]]*batch_size], [[[0]]*batch_size, [[0]]*batch_size])},
                        subspaces=[[[_ for _ in range(0,embedding_dim*2)]]*batch_size, 
                            [[_ for _ in range(embedding_dim*2, embedding_dim*4)]]*batch_size]
                    )
            elif batch["intervention_id"][0] == 0:
                _, counterfactual_outputs = alignable(
                        {"inputs_embeds":batch["input_ids"]},
                        [{"inputs_embeds":batch["source_input_ids"][:,0]}, None],
                        {"sources->base": ([[[0]]*batch_size, None], [[[0]]*batch_size, None])},
                        subspaces=[[[_ for _ in range(0,embedding_dim*2)]]*batch_size, 
                                None]
                    )
            elif batch["intervention_id"][0] == 1:
                _, counterfactual_outputs = alignable(
                        {"inputs_embeds":batch["input_ids"]},
                        [None, {"inputs_embeds":batch["source_input_ids"][:,0]}],
                        {"sources->base": ([None, [[0]]*batch_size], [None, [[0]]*batch_size])},
                        subspaces=[None, 
                                [[_ for _ in range(embedding_dim*2, embedding_dim*4)]]*batch_size]
                    )
            eval_labels += [batch['labels']]
            eval_preds += [torch.argmax(counterfactual_outputs[0],dim=1)]

    report = classification_report(torch.cat(eval_labels).cpu(), torch.cat(eval_preds).cpu(), output_dict=True)
    print(report)
    return report

def DAS_per_layer(model, train_dataset, test_dataset, layer = 0, batch_size = 640, lr = 0.001, embedding_dim = 4, epochs=1):
    alignable_config = IntervenableConfig(
        intervenable_model_type=type(model),
        intervenable_representations=[
            IntervenableRepresentationConfig(
                0,  # layer
                "block_output",  # intervention type
                "pos",  # intervention unit is now aligne with tokens
                1,  # max number of unit
                subspace_partition=None,  # binary partition with equal sizes
                intervention_link_key=0,
            ),
            IntervenableRepresentationConfig(
                0,  # layer
                "block_output",  # intervention type
                "pos",  # intervention unit is now aligne with tokens
                1,  # max number of unit
                subspace_partition=None,  # binary partition with equal sizes,
                intervention_link_key=0,
            ),
        ],
        intervenable_interventions_type=RotatedSpaceIntervention,
    )

    alignable = IntervenableModel(alignable_config, model, use_fast=True)
    alignable.set_device("cuda")
    alignable.disable_model_gradients()

    optimizer_params = []
    for k, v in alignable.interventions.items():
        optimizer_params += [{'params': v[0].rotate_layer.parameters()}]
        break
    optimizer = torch.optim.Adam(
        optimizer_params,
        lr=lr
    )
    
    train_alignable_model(alignable, train_dataset, optimizer, embedding_dim, batch_size=batch_size, epochs=epochs)
    report = test_alignable_model(alignable, test_dataset, embedding_dim, batch_size=batch_size)
    save_results(layer, report)

def get_causal_model(embedding_dim = 2, number_of_entities = 20):

    variables =  ["W", "X", "Y", "Z", "P", "Q", "O"]

    reps = [randvec(embedding_dim, lower=-1, upper=1) for _ in range(number_of_entities)]
    values = {variable:reps for variable in ["W","X","Y","Z"]}
    values["P"] = [True, False]
    values["Q"] = [True, False]
    values["O"] = [True, False]

    parents = {"W":[],"X":[], "Y":[], "Z":[], 
            "P":["W", "Y"], "Q":["X", "Z"], 
            "O":["P", "Q"]}

    def FILLER():
        return reps[0]

    functions = {"W":FILLER,"X":FILLER, "Y":FILLER, "Z":FILLER, 
                "P": lambda x,y: np.array_equal(x,y), 
                "Q":lambda x,y: np.array_equal(x,y), 
                "O": lambda x,y: x and y}

    pos = {"W":(0.2,0),"X":(1,0.1), "Y":(2,0.2), "Z":(2.8,0), 
            "P":(1,2), "Q":(2,2), 
            "O":(1.5,3)}

    return CausalModel(variables, values, parents, functions, pos = pos)

def save_results(layer, report):
    with open(f'report_{layer}.json', 'w') as json_file:
        json.dump(report, json_file)

def main():
    torch.cuda.empty_cache()
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    epochs=10

    # define causal model
    pattern_model = get_causal_model(embedding_dim=2, number_of_entities=20)

    # generate data
    n_examples = 1048576
    batch_size = 1024
    embedding_dim = 4

    X, y = pattern_model.generate_factual_dataset(n_examples,input_sampler)

    X = X.unsqueeze(1)

    # define MLP
    config = MLPConfig(h_dim=embedding_dim*4,
            activation_function = "relu",
            n_layer = 3,
            n_labels = 2,
            pdrop = 0.0
            )
    config, tokenizer, trained = create_mlp_classifier(config)
    trained.train()

    train_ds = Dataset.from_dict({
        "labels":[torch.FloatTensor([0,1]) if i == 1 else torch.FloatTensor([1,0]) for i in y],
        "inputs_embeds":X})

    training_args = TrainingArguments(
                    output_dir="test_trainer", 
                    evaluation_strategy="epoch",
                    learning_rate=0.001,
                    num_train_epochs=3,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size
                    )

    trainer = Trainer(
        model=trained,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=train_ds,
        compute_metrics=lambda x: {"accuracy":classification_report(x[0].argmax(1),x[1].argmax(1), output_dict=True)["accuracy"]},
    )

    _ = trainer.train()

    # define testing causal model
    test_pattern_model = get_causal_model(embedding_dim=4, number_of_entities=100)
    X_test, y_test = test_pattern_model.generate_factual_dataset(10000,input_sampler)

    test_ds = Dataset.from_dict({
        "labels":[torch.FloatTensor([0,1]) if i == 1 else torch.FloatTensor([1,0]) for i in y_test],
        "inputs_embeds":X_test})
    
    test_preds = trainer.predict(test_ds)

    print(classification_report(y_test, test_preds[0].argmax(1)))

    # apply DAS for every layer
    n_examples = 12800
    batch_size = 64

    torch.cuda.empty_cache()

    train_dataset = pattern_model.generate_counterfactual_dataset(n_examples,
                                                            intervention_id,
                                                            batch_size,
                                                            device = "cuda:0",
                                                            sampler=input_sampler)

    test_dataset = test_pattern_model.generate_counterfactual_dataset(10000,
                                                        intervention_id,
                                                        batch_size,
                                                        device = "cuda:0",
                                                        sampler=input_sampler)

    for layer in range(config.n_layer):
        DAS_per_layer(trained, train_dataset, test_dataset, layer, batch_size, epochs=epochs)

    # visualize results
    report_dicts = []

    for layer in range(config.n_layer):
        file_path = f'/gpfs/home1/mpislar/report_{layer}.json'
        with open(file_path, 'r') as json_file:
            report_dict = json.load(json_file)
            report_dicts.append(report_dict)

    values = []
    for layer, report_dict in enumerate(report_dicts, start=1):
        values.append(report_dict['accuracy'])

    plt.scatter(range(config.n_layer), values)
    plt.plot(range(config.n_layer), values,color='r')
    plt.xticks(range(int(min(plt.xticks()[0])), int(max(plt.xticks()[0])) + 1))
    plt.xlabel('layer')
    plt.ylabel('accuracy')
    plt.title('Accuracy for alignining the causal model with neural representations per layer')

    plt.tight_layout()
    plt.savefig('accuracy_per_layer.png')

if __name__ =="__main__":
    main()