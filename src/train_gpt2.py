import sys, os
sys.path.append(os.path.join('..', '..'))

import torch
import random
from sklearn.metrics import classification_report
from pyvene import CausalModel
from datetime import datetime
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import argparse
from ml_things import plot_dict
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
from causal_models import ArithmeticCausalModels
from utils import arithmetic_input_sampler

def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path)
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def tokenizePrompt(prompt):
    tokenizer = load_tokenizer("gpt2")
    prompt = f"{prompt['X']}+{prompt['Y']}+{prompt['Z']}=" # prompt for numerical causal model
    return tokenizer.encode(prompt, return_tensors='pt')

def train(model, dataloader, optimizer, scheduler, device):
    model.train()

    predictions_labels = []
    true_labels = []

    # total loss for this epoch.
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

        # clip the norm of the gradients to 1.0.
        # this is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step() # scheduler for the learnign rate

        logits = logits.detach().cpu().numpy()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    # calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # return all true labels and prediction for future evaluations.
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

    # calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss

def main():

    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument('--model_path', type=str, default='gpt2', help='model to finetune on the task')
    parser.add_argument('--results_path', type=str, default='training_gpt2_results/', help='path to the results folder')
    parser.add_argumnet('--epochs', type=int, default=50, help='epochs number for training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--n_training', type=int, default=2560, help='number of training samples')
    parser.add_argument('--seed', type=int, default=123, help='experiment seed to be able to reproduce the results')
    args = parser.parse_args()

    os.makedirs(args.results_path, exist_ok=True)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # min_class_value = 3
    offset = 3
    n_validation = round(0.2 * args.n_training)
    n_testing = round(0.1 * args.n_training)

    # Sequence Classification with GPT2 n_labels=28
    n_labels = 28 # 3 -..-> 31 => 10 included
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=args.model_path, num_labels=n_labels)
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.model_path, config=model_config)
    tokenizer = load_tokenizer(args.model_path)

    # generate training and validation data
    causal_model_family = ArithmeticCausalModels()
    causal_model = causal_model_family.get_model_by_id(1) # doesn't matter which causal model you choose to generate the factual data for training gpt2
    train_inputs, train_labels = causal_model.generate_factual_dataset(args.n_training, arithmetic_input_sampler, inputFunction=tokenizePrompt)
    val_inputs, val_labels = causal_model.generate_factual_dataset(n_validation, arithmetic_input_sampler, inputFunction=tokenizePrompt)
    test_inputs, test_labels = causal_model.generate_factual_dataset(n_testing, arithmetic_input_sampler, inputFunction=tokenizePrompt)

    train_ds = Dataset.from_dict(
        {
            "labels": train_labels - offset,
            "input_ids": train_inputs
        }
    )

    val_ds = Dataset.from_dict(
        {
            "labels": val_labels - offset,
            "input_ids": val_inputs
        }
    )

    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True) 
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # load model to defined device.
    model.to(device)
    print('Model loaded to `%s`'%device)

    optimizer = AdamW(model.parameters(),
                    lr = 2e-5,
                    eps = 1e-8
                    )

    # total number of training steps is number of batches * number of epochs
    total_steps = len(train_dataloader) * args.epochs

    # create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

    # store the average loss after each epoch for plotting later
    all_loss = {'train_loss':[], 'val_loss':[]}
    all_acc = {'train_acc':[], 'val_acc':[]}

    print('Epoch')
    for epoch in tqdm(range(args.epochs)):

        print()
        print('Training on batches...')

        # pass over a batch
        train_labels, train_predict, train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        train_acc = accuracy_score(train_labels, train_predict)

        # validation step
        print('Validation on batches...')
        valid_labels, valid_predict, val_loss = validation(val_dataloader, model, device)
        val_acc = accuracy_score(valid_labels, valid_predict)

        # print loss and accuracy values to see how training evolves.
        print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
        print()

        # store the loss value for plotting the learning curve.
        all_loss['train_loss'].append(train_loss)
        all_loss['val_loss'].append(val_loss)
        all_acc['train_acc'].append(train_acc)
        all_acc['val_acc'].append(val_acc)

    # plot loss and accuracy trends during training
    current_time = datetime.now()
    plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'], path=f'losses_{current_time.strftime("%Y%m%d_%H%M%S")[:-3]}.png')
    plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'], path=f'accuracies_{current_time.strftime("%Y%m%d_%H%M%S")[:-3]}.png')

    # test model
    test_ds = Dataset.from_dict(
        {
            "labels": test_labels - offset,
            "input_ids": test_inputs
        }
    )

    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    true_labels, predictions_labels, avg_epoch_loss = validation(test_dataloader, model, device)
    print(f"Average epoch loss on test_ds: {avg_epoch_loss}")
    evaluation_report = classification_report(true_labels, predictions_labels, labels=list(val_labels.squeeze()))
    print(evaluation_report)

    # save model
    model_config.save_pretrained("/home/mpislar/align-transformers/my_experiments/trained_gpt2forseq")
    model.save_pretrained("/home/mpislar/align-transformers/my_experiments/trained_gpt2forseq")
   
if __name__ =="__main__":
    main()