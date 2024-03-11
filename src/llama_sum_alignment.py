import sys, os
sys.path.append(os.path.join('..', '..'))

import pyvene
from pyvene import IntervenableRepresentationConfig, IntervenableConfig, IntervenableModel
import random
from transformers import Trainer, TrainingArguments
from datasets import Dataset

import torch
import seaborn as sns
from tqdm import tqdm, trange
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss

from pyvene import (
    IntervenableModel,
    BoundlessRotatedSpaceIntervention,
    IntervenableRepresentationConfig,
    IntervenableConfig,
)
from pyvene import create_llama
from pyvene import set_seed, count_parameters

def generate_sum_examples(num_examples=100):
    prompts = []
    answers = []

    for _ in range(num_examples):
        num1 = random.randint(1, 10)
        num2 = random.randint(1, 10)
        num3 = random.randint(1, 10)

        prompt = f"Calculate {num1}+{num2}+{num3}="
        answer = str(num1 + num2 + num3)

        prompts.append(prompt)
        answers.append(answer)

    return prompts, answers

def eval_llama(llama, tokenizer, prompts, labels):
    total_count = 0
    correct_count = 0
    with torch.no_grad():
        for prompt, label in zip(prompts, labels):
            # for k, v in inputs.items():
            #     if v is not None and isinstance(v, torch.Tensor):
            #         inputs[k] = v.to(llama.device)

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            # labels = tokenizer.encode(label, return_tensors="pt").to("cuda")
            output = llama.generate(input_ids['input_ids'], max_length=1, top_k=10, top_p=0.9)
            # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # generated_text = generated_text[len(prompt):].strip()

            print(prompt)
            print(tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

            # aligning forward!
    #         outputs = llama(
    #             input_ids=inputs["input_ids"],
    #             labels=labels["input_ids"]
    #         )

    #         pred_test_labels = torch.argmax(outputs.logits[:, -1], dim=-1)

    #         correct_labels = actual_test_labels == pred_test_labels

    #         total_count += len(correct_labels)
    #         correct_count += correct_labels.sum().tolist()
    # current_acc = round(correct_count / total_count, 2)
    # print(f"[WARNING: THIS NEEDS TO BE GOOD!] prealign task accuracy: {current_acc}")

def main():
    config, tokenizer, llama = create_llama()
    _ = llama.to("cuda")  # single gpu
    _ = llama.eval()  # always no grad on the model

    num_examples = 5
    prompts, labels = generate_sum_examples(num_examples)
    eval_llama(llama, tokenizer, prompts, labels)


if __name__ =="__main__":
    main()