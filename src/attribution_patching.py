import argparse
import json
import os
import random

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def prompt(x, y, z):
    return f"{x}+{y}+{z}="


def make_pairs(num_pairs, attribute, min_value, max_value, label_offset, num_labels):
    pairs = []

    def change(v):
        choices = [n for n in range(min_value, max_value + 1) if n != v]
        return random.choice(choices)

    while len(pairs) < num_pairs:
        x = random.randint(min_value, max_value)
        y = random.randint(min_value, max_value)
        z = random.randint(min_value, max_value)

        sx, sy, sz = x, y, z

        if attribute == "x":
            sx = change(x)
        elif attribute == "y":
            sy = change(y)
        elif attribute == "z":
            sz = change(z)
        elif attribute == "xy":
            sx = change(x)
            sy = change(y)
        elif attribute == "xz":
            sx = change(x)
            sz = change(z)
        elif attribute == "yz":
            sy = change(y)
            sz = change(z)
        elif attribute == "total":
            sx = change(x)
            sy = change(y)
            sz = change(z)
        else:
            raise ValueError(f"Unknown attribute: {attribute}")

        base_answer = x + y + z
        source_answer = sx + sy + sz

        if base_answer == source_answer:
            continue

        base_label = base_answer - label_offset
        source_label = source_answer - label_offset

        if not (0 <= base_label < num_labels and 0 <= source_label < num_labels):
            continue

        pairs.append({
            "base_prompt": prompt(x, y, z),
            "source_prompt": prompt(sx, sy, sz),
            "base_answer": base_answer,
            "source_answer": source_answer,
            "base_values": {"x": x, "y": y, "z": z},
            "source_values": {"x": sx, "y": sy, "z": sz},
        })

    return pairs


class Cache:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.attn_inputs = {}
        self.mlp_outputs = {}

    def clear(self):
        self.attn_inputs = {}
        self.mlp_outputs = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def add_hooks(self, retain_grad=False):
        self.remove()
        self.clear()

        for layer_idx, block in enumerate(self.model.transformer.h):
            def save_attn_input(module, inputs, output, layer_idx=layer_idx):
                value = inputs[0]
                if retain_grad:
                    value.retain_grad()
                self.attn_inputs[layer_idx] = value

            def save_mlp_output(module, inputs, output, layer_idx=layer_idx):
                value = output
                if retain_grad:
                    value.retain_grad()
                self.mlp_outputs[layer_idx] = value

            # input to c_proj is concatenated attention-head output before projection
            self.handles.append(block.attn.c_proj.register_forward_hook(save_attn_input))
            self.handles.append(block.mlp.register_forward_hook(save_mlp_output))


def encode(tokenizer, prompts, device):
    batch = tokenizer(prompts, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}


def run_batch(model, tokenizer, pairs, args, device):
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    hidden = model.config.n_embd
    head_dim = hidden // n_heads

    base_prompts = [p["base_prompt"] for p in pairs]
    source_prompts = [p["source_prompt"] for p in pairs]

    base_inputs = encode(tokenizer, base_prompts, device)
    source_inputs = encode(tokenizer, source_prompts, device)

    source_cache = Cache(model)
    with torch.no_grad():
        source_cache.add_hooks(retain_grad=False)
        model(**source_inputs)
        source_attn = {k: v.detach() for k, v in source_cache.attn_inputs.items()}
        source_mlp = {k: v.detach() for k, v in source_cache.mlp_outputs.items()}
        source_cache.remove()

    model.zero_grad(set_to_none=True)

    base_cache = Cache(model)
    base_cache.add_hooks(retain_grad=True)
    outputs = model(**base_inputs)
    logits = outputs.logits

    base_labels = torch.tensor(
        [p["base_answer"] - args.label_offset for p in pairs],
        device=device,
        dtype=torch.long,
    )
    source_labels = torch.tensor(
        [p["source_answer"] - args.label_offset for p in pairs],
        device=device,
        dtype=torch.long,
    )

    idx = torch.arange(len(pairs), device=device)

    # Positive score means the patch would push the base prompt toward the source answer.
    metric = (logits[idx, source_labels] - logits[idx, base_labels]).mean()
    metric.backward()

    head_scores = torch.zeros(n_layers, n_heads, device=device)
    head_abs_scores = torch.zeros(n_layers, n_heads, device=device)
    mlp_scores = torch.zeros(n_layers, device=device)
    mlp_abs_scores = torch.zeros(n_layers, device=device)

    for layer in range(n_layers):
        base_attn = base_cache.attn_inputs[layer]
        grad_attn = base_attn.grad
        delta_attn = source_attn[layer] - base_attn.detach()

        grad_attn = grad_attn.view(grad_attn.shape[0], grad_attn.shape[1], n_heads, head_dim)
        delta_attn = delta_attn.view(delta_attn.shape[0], delta_attn.shape[1], n_heads, head_dim)

        per_head = grad_attn * delta_attn
        head_scores[layer] = per_head.sum(dim=(0, 1, 3))
        head_abs_scores[layer] = per_head.abs().sum(dim=(0, 1, 3))

        base_mlp = base_cache.mlp_outputs[layer]
        grad_mlp = base_mlp.grad
        delta_mlp = source_mlp[layer] - base_mlp.detach()

        per_mlp = grad_mlp * delta_mlp
        mlp_scores[layer] = per_mlp.sum()
        mlp_abs_scores[layer] = per_mlp.abs().sum()

    base_cache.remove()
    model.zero_grad(set_to_none=True)

    return {
        "metric": float(metric.detach().cpu()),
        "head_scores": head_scores.detach().cpu(),
        "head_abs_scores": head_abs_scores.detach().cpu(),
        "mlp_scores": mlp_scores.detach().cpu(),
        "mlp_abs_scores": mlp_abs_scores.detach().cpu(),
    }


def top_heads(scores, k=15):
    rows = []
    for layer in range(scores.shape[0]):
        for head in range(scores.shape[1]):
            rows.append({
                "layer": layer,
                "head": head,
                "score": float(scores[layer, head]),
            })
    return sorted(rows, key=lambda r: r["score"], reverse=True)[:k]


def top_mlps(scores, k=12):
    rows = [{"layer": i, "score": float(scores[i])} for i in range(scores.shape[0])]
    return sorted(rows, key=lambda r: r["score"], reverse=True)[:k]


def save_heatmap(scores, path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping heatmap")
        return

    plt.figure(figsize=(10, 5))
    plt.imshow(scores, aspect="auto")
    plt.colorbar(label="absolute attribution score")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Attribution patching scores")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="mara589/arithmetic-gpt2")
    parser.add_argument("--attribute", default="xy", choices=["x", "y", "z", "xy", "xz", "yz", "total"])
    parser.add_argument("--num_pairs", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--min_value", type=int, default=1)
    parser.add_argument("--max_value", type=int, default=9)
    parser.add_argument("--label_offset", type=int, default=3)
    parser.add_argument("--output_dir", default="results/attribution_patching")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model on {device}: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    num_labels = model.config.num_labels

    print(f"Generating pairs for attribute: {args.attribute}")
    pairs = make_pairs(
        args.num_pairs,
        args.attribute,
        args.min_value,
        args.max_value,
        args.label_offset,
        num_labels,
    )

    total = None
    batches = 0

    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start:start + args.batch_size]
        print(f"Running batch {batches + 1}")

        result = run_batch(model, tokenizer, batch, args, device)

        if total is None:
            total = result
        else:
            for key in total:
                total[key] += result[key]

        batches += 1

    head_scores = total["head_scores"] / batches
    head_abs_scores = total["head_abs_scores"] / batches
    mlp_scores = total["mlp_scores"] / batches
    mlp_abs_scores = total["mlp_abs_scores"] / batches
    metric = total["metric"] / batches

    output = {
        "model_name": args.model_name,
        "attribute": args.attribute,
        "num_pairs": args.num_pairs,
        "batch_size": args.batch_size,
        "metric_mean": metric,
        "head_scores": head_scores.tolist(),
        "head_abs_scores": head_abs_scores.tolist(),
        "mlp_scores": mlp_scores.tolist(),
        "mlp_abs_scores": mlp_abs_scores.tolist(),
        "top_heads": top_heads(head_abs_scores),
        "top_mlps": top_mlps(mlp_abs_scores),
        "example_pairs": pairs[:10],
    }

    json_path = os.path.join(args.output_dir, f"atp_arithmetic_{args.attribute}.json")
    png_path = os.path.join(args.output_dir, f"atp_arithmetic_{args.attribute}_heads.png")

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    save_heatmap(head_abs_scores.numpy(), png_path)

    print("\nSaved:")
    print(json_path)
    print(png_path)

    print("\nTop heads:")
    for row in output["top_heads"][:10]:
        print(f"Layer {row['layer']}, Head {row['head']}: {row['score']:.4f}")

    print("\nTop MLPs:")
    for row in output["top_mlps"][:10]:
        print(f"Layer {row['layer']}: {row['score']:.4f}")


if __name__ == "__main__":
    main()
