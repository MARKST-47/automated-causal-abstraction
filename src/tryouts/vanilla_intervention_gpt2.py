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

def intervention_id(intervention):
    if "P" in intervention:
        return 0

def tokenizePrompt(prompt):
    tokenizer = load_tokenizer("gpt2")
    prompt = f"{prompt['X']}+{prompt['Y']}+{prompt['Z']}="
    return tokenizer.encode(prompt, padding=True, return_tensors='pt')

def save_results(layer, token, id, report):
    with open(f'/home/mpislar/align-transformers/my_experiments/results_{id}/report_layer_{layer}_tkn_{token}.json', 'w') as json_file:
        json.dump(report, json_file)

def main():

    # fixed parameters
    min_class_value = 3 # summing 3 numbers from 1 to 10 results in values from 3 -..--> 30 => 28 labels in total => subtract 3 to get the predicted label
    n_examples = 2560
    batch_size = 32
    pretrained_model_path = "/home/mpislar/align-transformers/my_experiments/trained_gpt2forseq"

    # load the trained model
    tokenizer = load_tokenizer('gpt2')
    model_config = GPT2Config.from_pretrained(pretrained_model_path)
    model_config.pad_token_id = tokenizer.pad_token_id # key in making it work?
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_path, config=model_config)
    model.resize_token_embeddings(len(tokenizer))

    for id in [1,2,3]:

        # get different causal models
        causal_model = get_causal_model(id)

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

        # per token per layer
        for token in range(6):
            for layer in range(model_config.n_layer):

                # define intervention model
                intervenable_config = IntervenableConfig(
                    model_type=type(model),
                    representations=[
                        RepresentationConfig(
                            layer,  # layer
                            "block_output",  # intervention type
                            # "pos",  # intervention unit is now aligne with tokens; default though
                            # 1,  # max number of tokens to intervene on
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
                                unit_locations={"sources->base": token}
                            )
                        
                        eval_labels += [batch["labels"].type(torch.long).squeeze() - min_class_value]
                        eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1)]
                report = classification_report(torch.cat(eval_labels).cpu(), torch.cat(eval_preds).cpu(), output_dict=True) # get the IIA
                print(report)
                save_results(layer, token, id, report)

if __name__ =="__main__":
    main()