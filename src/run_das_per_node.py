"""Per-node Distributed Alignment Search.

This is `run_das.py` with a selectable intervention *granularity*. The paper trains a
DAS rotation on the whole residual stream at a layer (`block_output`); here you choose
what a "node" is:

    --granularity block      -> block_output      (the paper's per-layer baseline)
    --granularity mlp        -> mlp_output        (that layer's MLP contribution only)
    --granularity attention  -> attention_output  (that layer's whole attention output)
    --granularity head       -> head_attention_value_output (one attention head)

Everything else - the counterfactual data (a genuine do(P) interchange built via
`generate_counterfactual_dataset` + `intervention_id`), the LowRankRotatedSpaceIntervention,
the IIA-from-classification-report eval - is identical to run_das.py, so the resulting IIA
per node means exactly what the paper's per-layer IIA means, just localized to a component.

NOTE on the `head` granularity: it uses pyvene's head-split component and the "h.pos" unit,
whose forward `unit_locations` format is more involved and can vary between pyvene versions.
The block/mlp/attention paths are drop-in identical to run_das.py and are the solid ones;
smoke-test the head path (e.g. --granularity head --layers 0 --heads 0 --n_training 256
--epochs 1) on your pyvene build before trusting a full sweep.
"""

import sys, os
sys.path.append(os.path.join('..', '..'))

import argparse
import random

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm, trange

from pyvene import count_parameters, set_seed
from pyvene import (
    IntervenableModel,
    IntervenableConfig,
    LowRankRotatedSpaceIntervention,
)

from transformers import GPT2Tokenizer, GPT2Config, GPT2ForSequenceClassification

from causal_models import ArithmeticCausalModels, SimpleSummingCausalModels
from utils import save_results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pyvene component string per granularity. "head" is handled specially (needs the h.pos unit).
GRANULARITY_TO_COMPONENT = {
    "block": "block_output",
    "mlp": "mlp_output",
    "attention": "attention_output",
    "head": "head_attention_value_output",
}


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenizePrompt(input):
    tokenizer = load_tokenizer("gpt2")
    prompt = f"{input['X']}+{input['Y']}+{input['Z']}="
    return tokenizer.encode(prompt, padding=True, return_tensors='pt')


def intervention_id(intervention):
    if "P" in intervention:
        return 0


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
    return {"accuracy": float(correct_count) / float(total_count)}


def calculate_loss(logits, labels, n_classes):
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = logits.contiguous().view(-1, n_classes)
    shift_labels = labels.contiguous().view(-1).to(shift_logits.device).long()
    return loss_fct(shift_logits, shift_labels)


def build_config(granularity, layer, head, low_rank_dimension, n_positions, model):
    """One pyvene IntervenableConfig for a single node at one layer."""
    component = GRANULARITY_TO_COMPONENT[granularity]

    if granularity == "head":
        rep = {
            "layer": layer,
            "component": component,
            "unit": "h.pos",
            "max_number_of_units": n_positions,
            "low_rank_dimension": low_rank_dimension,
        }
    else:
        rep = {
            "layer": layer,
            "component": component,
            "unit": "pos",
            "max_number_of_units": n_positions,
            "low_rank_dimension": low_rank_dimension,
        }

    return IntervenableConfig(
        rep,
        intervention_types=LowRankRotatedSpaceIntervention,
        model_type=type(model),
    )


def intervention_args(granularity, head, low_rank_dimension, n_positions, batch_size):
    """The unit_locations + subspaces passed to intervenable(...) for one node.

    For pos-based components both source and base intervene on all positions 0..n-1
    (pyvene reads a plain list as "same locations for source and base"). For the head
    component the "h.pos" unit needs a (source, base) tuple, each a [heads, positions]
    pair -- this is the version-sensitive bit flagged in the module docstring.
    """
    subspaces = [[[_ for _ in range(low_rank_dimension)]] * batch_size]

    if granularity == "head":
        # unit "h.pos": mirror pyvene's GET_LOC nesting. Each side is a single-intervention
        # list [[head_locs, pos_locs]]; head_locs and pos_locs are both [batch_size, num_units]
        # with equal nesting depth. One head across all positions -> num_units == n_positions.
        heads = [[head] * n_positions for _ in range(batch_size)]
        positions = [list(range(n_positions)) for _ in range(batch_size)]
        side = [[heads, positions]]
        unit_locations = {"sources->base": (side, side)}
    else:
        unit_locations = {"sources->base": [_ for _ in range(n_positions)]}

    return unit_locations, subspaces


def probe_seq_len(dataset, batch_size):
    """Actual tokenized sequence length of the generated counterfactual data. Derived from
    the real data (constant across the dataset) rather than a synthetic prompt: sampled
    inputs like '3+5+7=' can tokenize to a different length than '1+1+1=' due to BPE merges,
    and the DAS rotation must be sized to the length the model actually sees."""
    batch = next(iter(DataLoader(dataset, batch_size=min(batch_size, len(dataset)))))
    return int(batch["input_ids"].squeeze().shape[-1])


def run_intervenable(intervenable, base_ids, source_ids, granularity, head,
                      low_rank_dimension, n_positions):
    # pyvene's counterfactual dataset yields input_ids as float; the embedding lookup
    # needs integer indices. Casting is lossless (token ids are small integers).
    base_ids = base_ids.long()
    source_ids = source_ids.long()
    batch_size = base_ids.shape[0]
    # Intervene on every real token position; the prompt's tokenized length is whatever
    # GPT-2 actually produces (not a hardcoded 6), so read it off the batch.
    n_positions = base_ids.shape[1]
    unit_locations, subspaces = intervention_args(
        granularity, head, low_rank_dimension, n_positions, batch_size
    )
    return intervenable(
        {"input_ids": base_ids},
        [{"input_ids": source_ids}],
        unit_locations,
        subspaces=subspaces,
    )


def eval_intervenable(intervenable, eval_data, batch_size, granularity, head,
                       low_rank_dimension, n_positions, min_class_value=3):
    eval_labels, eval_preds = [], []
    with torch.no_grad():
        for inputs in tqdm(DataLoader(eval_data, batch_size), desc="Test"):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            inputs["input_ids"] = inputs["input_ids"].squeeze()
            inputs["source_input_ids"] = inputs["source_input_ids"].squeeze(2)

            _, counterfactual_outputs = run_intervenable(
                intervenable, inputs["input_ids"], inputs["source_input_ids"][:, 0],
                granularity, head, low_rank_dimension, n_positions,
            )
            eval_labels += [inputs["labels"].type(torch.long).squeeze() - min_class_value]
            eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1)]
    return classification_report(
        torch.cat(eval_labels).cpu(), torch.cat(eval_preds).cpu(), output_dict=True
    )


def nodes_to_run(granularity, layers, heads):
    """Yield (layer, head) pairs. head is None for non-head granularities."""
    for layer in layers:
        if granularity == "head":
            for head in heads:
                yield layer, head
        else:
            yield layer, None


def train_one_node(model, causal_family, train_id, granularity, layer, head,
                    low_rank_dimension, training_data, args, n_positions, n_classes,
                    min_class_value):
    config = build_config(granularity, layer, head, low_rank_dimension, n_positions, model)
    intervenable = IntervenableModel(config, model, use_fast=True)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()

    optimizer_params = []
    for k, v in intervenable.interventions.items():
        optimizer_params += [{"params": v.rotate_layer.parameters()}]
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr)

    intervenable.model.train()
    print(f"intervention trainable params: {intervenable.count_parameters()}")

    total_step = 0
    for epoch in trange(args.epochs, desc="Epoch"):
        torch.cuda.empty_cache()
        epoch_iterator = tqdm(
            DataLoader(
                training_data,
                batch_size=args.batch_size,
                sampler=batched_random_sampler(training_data, args.batch_size),
            ),
            desc=f"Epoch: {epoch}", position=0, leave=True,
        )
        for inputs in epoch_iterator:
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            inputs["input_ids"] = inputs["input_ids"].squeeze()
            inputs["source_input_ids"] = inputs["source_input_ids"].squeeze(2)

            _, counterfactual_outputs = run_intervenable(
                intervenable, inputs["input_ids"], inputs["source_input_ids"][:, 0],
                granularity, head, low_rank_dimension, n_positions,
            )
            eval_metrics = compute_metrics(
                counterfactual_outputs[0].argmax(1), inputs["labels"].squeeze() - min_class_value
            )
            loss = calculate_loss(
                counterfactual_outputs.logits, inputs["labels"] - min_class_value, n_classes
            )
            epoch_iterator.set_postfix({"loss": round(loss.item(), 2), "acc": eval_metrics["accuracy"]})

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if total_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                intervenable.set_zero_grad()
            total_step += 1

    return intervenable


def main():
    parser = argparse.ArgumentParser(description="Per-node Distributed Alignment Search.")
    parser.add_argument('--model_path', type=str, default="mara589/arithmetic-gpt2")
    parser.add_argument('--causal_model_type', type=str, choices=['arithmetic', 'simple'], default='arithmetic')
    parser.add_argument('--granularity', type=str, choices=list(GRANULARITY_TO_COMPONENT), default='mlp',
                        help="What a node is: block (per-layer, paper baseline), mlp, attention, or head.")
    parser.add_argument('--layers', type=int, nargs='*', default=None,
                        help="Layers to sweep (default: all).")
    parser.add_argument('--heads', type=int, nargs='*', default=None,
                        help="Heads to sweep for --granularity head (default: all).")
    parser.add_argument('--low_rank_dimensions', type=int, nargs='*', default=None,
                        help="Subspace dims to try. Default: [64,128,256] for block/mlp/attention, "
                             "[16,32,64] for head (a head is only head_dim-wide).")
    parser.add_argument('--n_positions', type=int, default=6,
                        help="Ignored: the tokenized prompt length is auto-probed at runtime.")
    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument('--n_training', type=int, default=2560)
    parser.add_argument('--n_testing', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=43)
    args = parser.parse_args()

    set_seed(args.seed)
    min_class_value = 3

    tokenizer = load_tokenizer('gpt2')
    model_config = GPT2Config.from_pretrained(args.model_path)
    model_config.pad_token_id = tokenizer.pad_token_id
    model = GPT2ForSequenceClassification.from_pretrained(args.model_path, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    n_classes = model.config.num_labels

    head_dim = model_config.n_embd // model_config.n_head
    if args.low_rank_dimensions is not None:
        low_rank_dimensions = args.low_rank_dimensions
    elif args.granularity == "head":
        low_rank_dimensions = [16, 32, 64]
    else:
        low_rank_dimensions = [64, 128, 256]

    if args.granularity == "head":
        over_cap = [k for k in low_rank_dimensions if k > head_dim]
        if over_cap:
            raise ValueError(
                f"low_rank_dimension {over_cap} exceeds head_dim={head_dim}; a single head "
                f"only spans {head_dim} dims. Pass smaller --low_rank_dimensions."
            )

    layers = args.layers if args.layers is not None else list(range(model_config.n_layer))
    heads = args.heads if args.heads is not None else list(range(model_config.n_head))

    family = ArithmeticCausalModels() if args.causal_model_type == 'arithmetic' else SimpleSummingCausalModels()

    # results/<causal_model_type>/per_node/<granularity>/...
    results_path = os.path.join(args.results_path, args.causal_model_type, "per_node", args.granularity)
    os.makedirs(results_path, exist_ok=True)

    for train_id, model_info in family.causal_models.items():
        print(f"generating do(P) counterfactual data for causal model {train_id} ({model_info['label']})...")
        training_data = model_info['causal_model'].generate_counterfactual_dataset(
            args.n_training, intervention_id, args.batch_size,
            device=device, sampler=model_info['causal_model'].sample_input_tree_balanced,
            input_function=tokenizePrompt,
        )

        seq_len = probe_seq_len(training_data, args.batch_size)
        print(f"tokenized prompt length (positions) for model {train_id}: {seq_len}")

        for low_rank_dimension in low_rank_dimensions:
            for layer, head in nodes_to_run(args.granularity, layers, heads):
                tag = f"L{layer}" + (f"H{head}" if head is not None else "")
                print(f"\n=== DAS {args.granularity} node {tag}, k={low_rank_dimension}, "
                      f"causal model {train_id} ===")

                intervenable = train_one_node(
                    model, family, train_id, args.granularity, layer, head,
                    low_rank_dimension, training_data, args, seq_len,
                    n_classes, min_class_value,
                )

                # `exp_id` in save_results encodes both the subspace dim and the node id so files
                # from different heads at the same layer/dim don't collide.
                exp_id = f"{low_rank_dimension}_{args.granularity}_{tag}"
                for test_id, test_info in family.causal_models.items():
                    if args.causal_model_type == 'simple' and test_id != train_id:
                        continue
                    testing_data = test_info['causal_model'].generate_counterfactual_dataset(
                        args.n_testing, intervention_id, args.batch_size,
                        device=device, sampler=test_info['causal_model'].sample_input_tree_balanced,
                        input_function=tokenizePrompt,
                    )
                    report = eval_intervenable(
                        intervenable, testing_data, args.batch_size, args.granularity, head,
                        low_rank_dimension, seq_len, min_class_value,
                    )
                    save_results(results_path, report, layer, exp_id, train_id, test_id)


if __name__ == "__main__":
    main()
