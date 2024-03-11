import sys, os
sys.path.append(os.path.join('..', '..'))

from sklearn.metrics import classification_report
from pyvene import CausalModel
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
import random
import json

from transformers import (GPT2Tokenizer,
                          GPT2Config,
                          GPT2ForSequenceClassification)

from pyvene import (
    IntervenableModel,
    RotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig
)

# sample such numbers to be fed into the task
def input_sampler():
    A = randNum()
    B = randNum()
    C = randNum()
    return {"X":A, "Y":B, "Z":C}

def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path)
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def randNum(lower=1, upper=10):
    number = random.randint(lower, upper)
    return number

def get_causal_model(id=1):

    variables =  ["X", "Y", "Z", "P", "O"]
    number_of_entities = 20

    reps = [randNum() for _ in range(number_of_entities)]
    values = {variable:reps for variable in ["X", "Y", "Z"]}
    values["P"] = list(range(2, 21))
    values["O"] = list(range(3, 31))

    if id == 1:
        parents = {"X":[], "Y":[], "Z":[], 
                "P":["X", "Y"],
                "O":["P", "Z"]}
    elif id == 2:
        parents = {"X":[], "Y":[], "Z":[], 
            "P":["X", "Z"],
            "O":["P", "Y"]}
    else:
        parents = {"X":[], "Y":[], "Z":[], 
                "P":["Y", "Z"],
                "O":["P", "X"]}

    def FILLER():
        return reps[0]
    
    functions = {"X":FILLER, "Y":FILLER, "Z":FILLER,
                "P": lambda x, y: x + y,
                "O": lambda x, y: x + y}

    pos = {"X":(1,0.1), "Y":(2,0.2), "Z":(2.8,0), 
            "P":(1,2),
            "O":(1.5,3)}

    return CausalModel(variables, values, parents, functions, pos = pos)

def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        total_count += 1
        correct_count += eval_pred == eval_label
    accuracy = float(correct_count) / float(total_count)
    return {"accuracy": accuracy}


def compute_loss(outputs, labels):
    CE = torch.nn.CrossEntropyLoss()
    labels = labels.long()
    return CE(outputs, labels)


def batched_random_sampler(data, batch_size):
    batch_indices = [_ for _ in range(int(len(data) / batch_size))]
    random.shuffle(batch_indices)
    for b_i in batch_indices:
        for i in range(b_i * batch_size, (b_i + 1) * batch_size):
            yield i

def intervention_id(intervention):
    if "P" in intervention:
        return 0

def tokenizePrompt(prompt):
    tokenizer = load_tokenizer("gpt2")
    prompt = f"{prompt['X']}+{prompt['Y']}+{prompt['Z']}="
    return tokenizer.encode(prompt, padding=True, return_tensors='pt')

def save_results(layer, token, id, report):
    with open(f'/home/mpislar/align-transformers/my_results/rotation_intervention/results_{id}/report_layer_{layer}_tkn_{token}.json', 'w') as json_file:
        json.dump(report, json_file)

def main():

    n_training = 2560
    n_testing = round(0.1 * n_training)
    batch_size = 32
    epochs = 10
    gradient_accumulation_steps = 1
    total_step = 0
    min_class_value = 3
    
    # load the trained model
    model_path = "/home/mpislar/align-transformers/my_experiments/trained_gpt2forseq"
    tokenizer = load_tokenizer('gpt2')
    model_config = GPT2Config.from_pretrained(model_path)
    model_config.pad_token_id = tokenizer.pad_token_id
    model = GPT2ForSequenceClassification.from_pretrained(model_path, config=model_config)

    model.resize_token_embeddings(len(tokenizer))

    for id in [1,2,3]:

        # define causal models
        causal_model = get_causal_model(id)
        test_causal_model = get_causal_model(id)

        # generate counterfactual training data
        print('generating data for DAS...')

        train_dataset = causal_model.generate_counterfactual_dataset(
            n_training,
            intervention_id,
            batch_size,
            sampler=input_sampler,
            inputFunction=tokenizePrompt
        )

        for token in range(6):
            for layer in range(model_config.n_layer):

                # define intervention model
                intervenable_config = IntervenableConfig(
                    model_type=type(model),
                    representations=[
                        RepresentationConfig(
                            0,  # layer
                            "block_output",  # intervention type
                            # "pos",  # intervention unit is now aligne with tokens; default though
                            # 1,  # max number of tokens to intervene on
                            # subspace_partition=None,  # binary partition with equal sizes
                            # intervention_link_key=0,
                        )
                    ],
                    intervention_types=RotatedSpaceIntervention
                )

                intervenable = IntervenableModel(intervenable_config, model, use_fast=True)
                intervenable.set_device("cuda")
                intervenable.disable_model_gradients()

                

                # activate parameters and define optimizer

                # t_total = int(len(train_dataset) * epochs)
                optimizer_params = []
                for k, v in intervenable.interventions.items():
                    optimizer_params += [{"params": v[0].rotate_layer.parameters()}]
                    break

                optimizer = torch.optim.Adam(intervenable.get_trainable_parameters(), lr=0.001)

                # train DAS
                print('training DAS...')

                # embedding_dim = 768

                intervenable.model.train()  # train enables drop-off but no grads
                print("intervention trainable parameters: ", intervenable.count_parameters())
                train_iterator = trange(0, int(epochs), desc="Epoch")

                for epoch in train_iterator:
                    epoch_iterator = tqdm(
                        DataLoader(
                            train_dataset,
                            batch_size=batch_size,
                            sampler=batched_random_sampler(train_dataset, batch_size),
                        ),
                        desc=f"Epoch: {epoch}",
                        position=0,
                        leave=True,
                    )
                    for batch in epoch_iterator:
                        batch["input_ids"] = batch["input_ids"].squeeze()
                        batch["source_input_ids"] = batch["source_input_ids"].squeeze(2)
                        batch_size = batch["input_ids"].shape[0]
                        for k, v in batch.items():
                            if v is not None and isinstance(v, torch.Tensor):
                                batch[k] = v.to("cuda")

                        if batch["intervention_id"][0] == 0:
                            _, counterfactual_outputs = intervenable(
                                {"input_ids": batch["input_ids"]}, # base
                                [{"input_ids": batch["source_input_ids"][:, 0]}], # source, selecting all rows and only the values from the first column
                                unit_locations={"sources->base": token} # target token 0
                            )

                        eval_metrics = compute_metrics(
                            counterfactual_outputs[0].argmax(1), batch["labels"].squeeze() - min_class_value
                        )

                        # loss and backprop
                        loss = compute_loss(
                            counterfactual_outputs[0], batch["labels"].squeeze() - min_class_value
                        )

                        print(loss)
                        print(eval_metrics)

                        epoch_iterator.set_postfix({"loss": loss, "acc": eval_metrics["accuracy"]})

                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps
                        loss.backward()
                        if total_step % gradient_accumulation_steps == 0:
                            optimizer.step()
                            intervenable.set_zero_grad()
                        total_step += 1
                
                # test DAS

                test_dataset = test_causal_model.generate_counterfactual_dataset(
                    n_testing, intervention_id, batch_size, device="cuda:0", sampler=input_sampler, inputFunction=tokenizePrompt
                )

                eval_labels = []
                eval_preds = []
                with torch.no_grad():
                    epoch_iterator = tqdm(DataLoader(test_dataset, batch_size), desc=f"Test")
                    for step, batch in enumerate(epoch_iterator):
                        for k, v in batch.items():
                            if v is not None and isinstance(v, torch.Tensor):
                                batch[k] = v.to("cuda")
                        batch["input_ids"] = batch["input_ids"].squeeze()
                        batch["source_input_ids"] = batch["source_input_ids"].squeeze(2)

                        if batch["intervention_id"][0] == 0:
                            _, counterfactual_outputs = intervenable(
                                {"input_ids": batch["input_ids"]}, # base
                                [{"input_ids": batch["source_input_ids"][:, 0]}], # source, selecting all rows and only the values from the first column
                                unit_locations={"sources->base": token} # target token 0
                            )
                        
                        eval_labels += [batch["labels"].type(torch.long).squeeze() - min_class_value]
                        eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1)]

                # print(classification_report(torch.cat(eval_labels).cpu(), torch.cat(eval_preds).cpu()))
                report = classification_report(torch.cat(eval_labels).cpu(), torch.cat(eval_preds).cpu(), output_dict=True) # get the IIA
                print(report)
                save_results(layer, token, id, report)

if __name__ =="__main__":
    main()