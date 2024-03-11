import sys, os
sys.path.append(os.path.join('..', '..'))

import torch
import torch.nn.functional as F
import random
from torch.utils.data import random_split
from sklearn.metrics import classification_report
from pyvene import CausalModel
from datetime import datetime
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import numpy as np
from ml_things import plot_dict
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

def randNum(lower=1, upper=10):
    number = random.randint(lower, upper)
    return number

def get_causal_model():

    variables =  ["X", "Y", "Z", "P", "O"]
    number_of_entities = 20

    reps = [randNum() for _ in range(number_of_entities)]
    values = {variable:reps for variable in ["X", "Y", "Z"]}
    values["P"] = list(range(2, 21))
    values["O"] = list(range(3, 31))

    parents = {"X":[], "Y":[], "Z":[], 
            "P":["X", "Y"],
            "O":["P", "Z"]}

    def FILLER():
        return reps[0]

    functions = {"X":FILLER, "Y":FILLER, "Z":FILLER,
                "P": lambda x, y: x + y,
                "O": lambda x, y: x + y}

    pos = {"X":(1,0.1), "Y":(2,0.2), "Z":(2.8,0), 
            "P":(1,2),
            "O":(1.5,3)}

    return CausalModel(variables, values, parents, functions, pos = pos)

def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path)
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def input_sampler():
    A = randNum()
    B = randNum()
    C = randNum()
    return {"X":A, "Y":B, "Z":C}

def tokenizePrompt(prompt):
    tokenizer = load_tokenizer("gpt2")
    prompt = f"{prompt['X']}+{prompt['Y']}+{prompt['Z']}=" # prompt for numerical causal model
    return tokenizer.encode(prompt, return_tensors='pt')

def train(model, dataloader, optimizer, scheduler, device):
    model.train()

    predictions_labels = []
    true_labels = []

    # Total loss for this epoch.
    total_loss = 0

    for batch in tqdm(dataloader, total=len(dataloader)):

        true_labels += batch['labels'][0].type(torch.long).tolist() # used later for eval

        # get the values to device
        batch['input_ids'] = torch.stack(batch['input_ids'][0]).T.to(device) # results in a (batch_size)
        batch['labels'] = batch['labels'][0].type(torch.long).to(device)

        # batch = {k:torch.tensor(v).type(torch.long).to(device) for k,v in batch.items()} # move to device
        
        model.zero_grad()
        outputs = model(**batch)
        loss, logits = outputs[:2] # probs has to be n_labels?

        total_loss += loss.item()

        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step() # scheduler for the learnign rate

        logits = logits.detach().cpu().numpy()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediction for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss

def validation(dataloader, model, device_):

    model.eval()

    predictions_labels = []
    true_labels = []
    total_loss = 0

    for batch in tqdm(dataloader, total=len(dataloader)):

        true_labels += batch['labels'][0].type(torch.long).tolist() # for eval

        batch['input_ids'] = torch.stack(batch['input_ids'][0]).T.to(device_)
        batch['labels'] = batch['labels'][0].type(torch.long).to(device_)

        # batch = {k:torch.tensor(v).type(torch.long).to(device_) for k,v in batch.items()} # move to device

        with torch.no_grad():        

            outputs = model(**batch)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss += loss.item()
            
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss

def main():

    # set general parameters
    set_seed(123)
    epochs = 50
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_or_path = 'gpt2'
    min_class_value = 3
    n_training = 2560
    n_validation = round(0.2 * n_training)
    n_testing = round(0.1 * n_training)

    # Sequence Classification with GPT2 n_labels=28
    n_labels = 28 # 3 -..-> 31 => 10 included
    # n_labels = 25 # 3 - ..-> 27 => 9 included
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)
    tokenizer = load_tokenizer(model_name_or_path)

    # generate training and validation data
    causal_model = get_causal_model()
    train_inputs, train_labels = causal_model.generate_factual_dataset(n_training, input_sampler, inputFunction=tokenizePrompt)
    val_inputs, val_labels = causal_model.generate_factual_dataset(n_validation, input_sampler, inputFunction=tokenizePrompt)
    test_inputs, test_labels = causal_model.generate_factual_dataset(n_testing, input_sampler, inputFunction=tokenizePrompt)

    train_ds = Dataset.from_dict(
        {
            "labels": train_labels - min_class_value,
            "input_ids": train_inputs
        }
    )

    val_ds = Dataset.from_dict(
        {
            "labels": val_labels - min_class_value,
            "input_ids": val_inputs
        }
    )

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) 
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Load model to defined device.
    model.to(device)
    print('Model loaded to `%s`'%device)

    optimizer = AdamW(model.parameters(),
                    lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # default is 1e-8.
                    )

    # Total number of training steps is number of batches * number of epochs.
    # `train_dataloader` contains batched data so `len(train_dataloader)` gives 
    # us the number of batches.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

    # Store the average loss after each epoch so we can plot them.
    all_loss = {'train_loss':[], 'val_loss':[]}
    all_acc = {'train_acc':[], 'val_acc':[]}

    print('Epoch')
    for epoch in tqdm(range(epochs)):

        print()
        print('Training on batches...')

        # pass over a batch
        train_labels, train_predict, train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        train_acc = accuracy_score(train_labels, train_predict)

        # validation step
        print('Validation on batches...')
        valid_labels, valid_predict, val_loss = validation(val_dataloader, model, device)
        val_acc = accuracy_score(valid_labels, valid_predict)

        # Print loss and accuracy values to see how training evolves.
        print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
        print()

        # Store the loss value for plotting the learning curve.
        all_loss['train_loss'].append(train_loss)
        all_loss['val_loss'].append(val_loss)
        all_acc['train_acc'].append(train_acc)
        all_acc['val_acc'].append(val_acc)

    # plot loss and accuracy trends during training
    current_time = datetime.now()
    plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'], path=f'losses_{current_time.strftime("%Y%m%d_%H%M%S")[:-3]}.png')
    plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'], path=f'accuracies_{current_time.strftime("%Y%m%d_%H%M%S")[:-3]}.png')

    # testing phase
    test_ds = Dataset.from_dict(
        {
            "labels": test_labels - min_class_value,
            "input_ids": test_inputs
        }
    )

    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    true_labels, predictions_labels, avg_epoch_loss = validation(test_dataloader, model, device)
    print(f"Average epoch loss on test_ds: {avg_epoch_loss}")
    evaluation_report = classification_report(true_labels, predictions_labels, labels=list(val_labels.squeeze()))
    print(evaluation_report)

    # save model
    model_config.save_pretrained("/home/mpislar/align-transformers/my_experiments/trained_gpt2forseq")
    model.save_pretrained("/home/mpislar/align-transformers/my_experiments/trained_gpt2forseq")
   
if __name__ =="__main__":
    main()