import sys, os
sys.path.append(os.path.join('..', '..'))

from sklearn.metrics import classification_report
from pyvene import CausalModel
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import random
import json
import matplotlib.pyplot as plt

from transformers import (GPT2Tokenizer,
                          GPT2Config,
                          GPT2ForSequenceClassification)

from pyvene import (
    IntervenableModel,
    RepresentationConfig,
    IntervenableConfig,
    VanillaIntervention
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

def experimental_causal_model():

    variables =  ["X", "Y", "Z", "A", "B", "C", "P", "O"]
    number_of_entities = 20

    reps = [randNum() for _ in range(number_of_entities)]
    values = {variable:reps for variable in ["X", "Y", "Z"]}
    values["A"] = list(range(1,11)) # can possibly take values from 1 to 10
    values["B"] = list(range(1,11))
    values["C"] = list(range(1,11))
    values["P"] = list(range(2, 21))
    values["O"] = list(range(3, 31))

    ##         O 
    ##      /   |
    ##     P    |
    ##   /  \   |
    ##  A   B   C
    ##  ^   ^   ^
    ##  |   |   |
    ##  X   Y   Z
    
    parents = {"X":[], "Y":[], "Z":[],
            "A":["X"],
            "B":["Y"],
            "C":["Z"],
            "P":["A", "B"],
            "O":["P", "C"]}

    def FILLER():
        return reps[0]
    
    functions = {"X":FILLER, "Y":FILLER, "Z":FILLER,
                "A": lambda x: x,
                "B": lambda x: x,
                "C": lambda x: x,
                "P": lambda x, y: x + y,
                "O": lambda x, y: x + y}

    return CausalModel(variables, values, parents, functions)

def intervention_id(intervention):
    if "P" in intervention and "C" in intervention:
        return 1
    if "P" in intervention and "A" in intervention:
        return 0
    if "P" in intervention and "B" in intervention:
        return 0
    if "A" in intervention and "B" in intervention and "C" in intervention:
        return 2
    if "A" in intervention and "C" in intervention and "P" in intervention:
        return 1
    if "A" in intervention and "B" in intervention and "P" in intervention:
        return 1
    if "B" in intervention and "C" in intervention and "P" in intervention:
        return 1
    if "A" in intervention and "C" in intervention:
        return 3
    if "A" in intervention and "B" in intervention:
        return 4
    if "B" in intervention and "C" in intervention:
        return 5
    if "P" in intervention:
        return 0
    if "A" in intervention:
        return 6
    if "B" in intervention:
        return 7
    if "C" in intervention:
        return 8

def tokenizePrompt(prompt):
    tokenizer = load_tokenizer("gpt2")
    prompt = f"{prompt['X']}+{prompt['Y']}+{prompt['Z']}="
    return tokenizer.encode(prompt, padding=True, return_tensors='pt')

def save_results(layer, token, id, report):
    with open(f'/home/mpislar/LLM_causal_model_learning/results/results_{id}/report_layer_{layer}_tkn_{token}.json', 'w') as json_file:
        json.dump(report, json_file)

def main():

    # fixed parameters
    min_class_value = 3 # summing 3 numbers from 1 to 10 results in values from 3 -..--> 30 => 28 labels in total => subtract 3 to get the predicted label
    n_examples = 64
    batch_size = 32
    pretrained_model_path = "/home/mpislar/LLM_causal_model_learning/models/trained_gpt2forseq"

    # load the trained model
    tokenizer = load_tokenizer('gpt2')
    model_config = GPT2Config.from_pretrained(pretrained_model_path)
    model_config.pad_token_id = tokenizer.pad_token_id # key in making it work?
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_path, config=model_config)
    model.resize_token_embeddings(len(tokenizer))

    # get different causal models
    causal_model = experimental_causal_model()

    # generate counterfactual data
    print('generating data for DAS...')

    counterfactual_data = causal_model.generate_counterfactual_dataset(
        n_examples,
        intervention_id,
        batch_size,
        device="cuda:0",
        sampler=input_sampler,
        inputFunction=tokenizePrompt
    )

    # for token in range(6):
    # for layer in range(model_config.n_layer):
    for layer in [0]:

        # define intervention model
        intervenable_config = IntervenableConfig(
            model_type=type(model),
            representations=[
                RepresentationConfig(
                    layer,  # layer
                    "block_output",  # intervention type
                    "pos",  # intervention unit is now aligne with tokens; default though
                    4,  # max number of tokens to intervene on
                    # subspace_partition=None,  # binary partition with equal sizes
                    # intervention_link_key=0,
                )
            ],
            intervention_types=VanillaIntervention,
        )

        intervenable = IntervenableModel(intervenable_config, model, use_fast=True)
        intervenable.set_device("cuda")

        for parameter in intervenable.get_trainable_parameters():
            parameter.to("cuda:0")

        # vanilla intervention happening
        print('vanilla intervening on gpt2')

        eval_labels = []
        eval_preds = []
        with torch.no_grad():
            epoch_iterator = tqdm(DataLoader(counterfactual_data, batch_size), desc=f"Test")
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
                        # the location to intervene at (token 0)
                        unit_locations={
                            "sources->base": [0,1,2]
                        }
                    )
                elif batch["intervention_id"][0] == 1:

                    _, counterfactual_outputs = intervenable(
                        {"input_ids": batch["input_ids"]}, # base
                        [{"input_ids": batch["source_input_ids"][:, 0]}], # source, selecting all rows and only the values from the first column
                        # the location to intervene at (token 0)
                        unit_locations={
                            "sources->base": [0,1,2,4]
                        }
                    )

                elif batch["intervention_id"][0] == 2:

                    _, counterfactual_outputs = intervenable(
                        {"input_ids": batch["input_ids"]}, # base
                        [{"input_ids": batch["source_input_ids"][:, 0]}], # source, selecting all rows and only the values from the first column
                        # the location to intervene at (token 0)
                        unit_locations={
                            "sources->base": [0,2,4]
                        }
                    )
                
                elif batch["intervention_id"][0] == 3:

                    _, counterfactual_outputs = intervenable(
                        {"input_ids": batch["input_ids"]}, # base
                        [{"input_ids": batch["source_input_ids"][:, 0]}], # source, selecting all rows and only the values from the first column
                        # the location to intervene at (token 0)
                        unit_locations={
                            "sources->base": [0,4]
                        }
                    )

                elif batch["intervention_id"][0] == 4:

                    _, counterfactual_outputs = intervenable(
                        {"input_ids": batch["input_ids"]}, # base
                        [{"input_ids": batch["source_input_ids"][:, 0]}], # source, selecting all rows and only the values from the first column
                        # the location to intervene at (token 0)
                        unit_locations={
                            "sources->base": [0,2]
                        }
                    )

                elif batch["intervention_id"][0] == 5:

                    _, counterfactual_outputs = intervenable(
                        {"input_ids": batch["input_ids"]}, # base
                        [{"input_ids": batch["source_input_ids"][:, 0]}], # source, selecting all rows and only the values from the first column
                        # the location to intervene at (token 0)
                        unit_locations={
                            "sources->base": [2,4]
                        }
                    )

                elif batch["intervention_id"][0] == 6:

                    _, counterfactual_outputs = intervenable(
                        {"input_ids": batch["input_ids"]}, # base
                        [{"input_ids": batch["source_input_ids"][:, 0]}], # source, selecting all rows and only the values from the first column
                        # the location to intervene at (token 0)
                        unit_locations={
                            "sources->base": 0
                        }
                    )
                
                elif batch["intervention_id"][0] == 7:

                    _, counterfactual_outputs = intervenable(
                        {"input_ids": batch["input_ids"]}, # base
                        [{"input_ids": batch["source_input_ids"][:, 0]}], # source, selecting all rows and only the values from the first column
                        # the location to intervene at (token 0)
                        unit_locations={
                            "sources->base": 2
                        }
                    )
                
                elif batch["intervention_id"][0] == 8:

                    _, counterfactual_outputs = intervenable(
                        {"input_ids": batch["input_ids"]}, # base
                        [{"input_ids": batch["source_input_ids"][:, 0]}], # source, selecting all rows and only the values from the first column
                        # the location to intervene at (token 0)
                        unit_locations={
                            "sources->base": 4
                        }
                    )
                
                eval_labels += [batch["labels"].type(torch.long).squeeze() - min_class_value]
                eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1)]
        report = classification_report(torch.cat(eval_labels).cpu(), torch.cat(eval_preds).cpu()) # get the IIA
        print(report)
        # save_results(layer, token, id, report)

if __name__ =="__main__":
    main()