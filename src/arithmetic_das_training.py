import sys, os
sys.path.append(os.path.join('..', '..'))

from sklearn.metrics import classification_report
from pyvene import CausalModel
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import random
import json
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
from pyvene import set_seed, count_parameters

from transformers import (GPT2Tokenizer,
                          GPT2Config,
                          GPT2ForSequenceClassification)

from pyvene import (
    IntervenableModel,
    RepresentationConfig,
    IntervenableConfig,
    VanillaIntervention,
    RotatedSpaceIntervention,
    LowRankRotatedSpaceIntervention,
    BoundlessRotatedSpaceIntervention
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

def save_results(layer, token, id, report):
    with open(f'/home/mpislar/LLM_causal_model_learning/results/results_{id}/report_layer_{layer}_tkn_{token}.json', 'w') as json_file:
        json.dump(report, json_file)

def main():

    # fixed parameters
    min_class_value = 3 # summing 3 numbers from 1 to 10 results in values from 3 -..--> 30 => 28 labels in total => subtract 3 to get the predicted label
    n_training = 64
    n_testing = 32
    batch_size = 32
    epochs = 10
    # embedding_dim = 768
    gradient_accumulation_steps = 1
    total_step = 0
    min_class_value = 3
    pretrained_model_path = "/home/mpislar/LLM_causal_model_learning/models/trained_gpt2forseq"

    # load the trained model
    tokenizer = load_tokenizer('gpt2')
    model_config = GPT2Config.from_pretrained(pretrained_model_path)
    model_config.pad_token_id = tokenizer.pad_token_id # key in making it work?
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_path, config=model_config)
    model.resize_token_embeddings(len(tokenizer))

    # get different causal models

    causal_model = get_causal_model(id=1)

    # generate counterfactual data
    print('generating data for DAS...')

    training_counterfactual_data = causal_model.generate_counterfactual_dataset(
        n_training,
        intervention_id,
        batch_size,
        device="cuda:0",
        sampler=input_sampler,
        inputFunction=tokenizePrompt
    )

    low_rank_dimension = 128 # try 64, 256

    for layer in range(model_config.n_layer):
        # for token in [0,1,2,3,4,5]:

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

        # t_total = int(len(training_counterfactual_data) * batch_size)
        # warm_up_steps = 0.1 * t_total

        optimizer_params = []
        for k, v in intervenable.interventions.items():
            optimizer_params += [{"params": v[0].rotate_layer.parameters()}]

        optimizer = torch.optim.Adam(optimizer_params, lr=0.01)
        scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, end_factor=0.1, total_iters=epochs
        )

        # training with boundless intervention

        print('BoundlessDAS training...')

        intervenable.model.train()
        print("intervention trainable parameters: ", intervenable.count_parameters())
        print("gpt2 trainable parameters: ", count_parameters(intervenable.model))
        train_iterator = trange(0, int(epochs), desc="Epoch")

        for epoch in train_iterator:
            torch.cuda.empty_cache()
            epoch_iterator = tqdm(
                DataLoader(
                    training_counterfactual_data,
                    batch_size=batch_size,
                    sampler=batched_random_sampler(training_counterfactual_data, batch_size),
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
                    {"sources->base": [0,1,2,3,4,5]},  # swap 0th token
                    subspaces=[
                        [[_ for _ in range(0, 128)]] * batch_size # taking half of the repr. and rotating it
                    ]
                )
                # eval_metrics = compute_metrics(
                #     [counterfactual_outputs.logits], [inputs["labels"] - min_class_value]
                # )

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

        # generate testing counterfactual data
        print('testing...')

        for id in [1]:

            causal_model = get_causal_model(id=id)

            testing_counterfactual_data = causal_model.generate_counterfactual_dataset(
                n_testing,
                intervention_id,
                batch_size,
                device="cuda:0",
                sampler=input_sampler,
                inputFunction=tokenizePrompt
            )

            eval_labels = []
            eval_preds = []
            with torch.no_grad():
                epoch_iterator = tqdm(DataLoader(testing_counterfactual_data, batch_size), desc=f"Test")
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
                        {"sources->base": [0,1,2,3,4,5]},  # swap 0th token
                        subspaces=[
                            [[_ for _ in range(0, 128)]] * batch_size # taking half of the repr. and rotating it
                        ]
                    )
                    # eval_labels += [inputs["labels"] - min_class_value]
                    # eval_preds += [counterfactual_outputs.logits]

                    eval_labels += [inputs["labels"].type(torch.long).squeeze() - min_class_value]
                    eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1)]
            report = classification_report(torch.cat(eval_labels).cpu(), torch.cat(eval_preds).cpu()) # get the IIA
            print(report)
            # save_results(layer, 0, id, report)
            # eval_metrics = compute_metrics(eval_preds, eval_labels)
            # print(eval_metrics)

if __name__ =="__main__":
    main()