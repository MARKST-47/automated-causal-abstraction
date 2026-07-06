import argparse
import json
import os
import random

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from causal_models import ArithmeticCausalModels


def prompt(x, y, z):
    return f"{x}+{y}+{z}="


def encode(tokenizer, prompts, device):
    batch = tokenizer(prompts, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}


def make_causal_pairs(causal_model, num_pairs, min_value, max_value, label_offset, num_labels):
    """Generate (base, source) pairs together with the counterfactual answer that
    `causal_model` predicts under do(P = P_source), i.e. P takes the value it would have
    under the source input while the other parent of O keeps its base value. This mirrors
    the paper's Appendix C.2 example (base=14, source=18, counterfactual target=19 - a
    third value distinct from either run's true answer), generalized to any of the three
    ArithmeticCausalModels hypotheses via their own `parents`/`functions`.
    """
    p_parents = causal_model.parents["P"]
    other_parent = [v for v in causal_model.parents["O"] if v != "P"][0]

    pairs = []
    while len(pairs) < num_pairs:
        base = {v: random.randint(min_value, max_value) for v in ["X", "Y", "Z"]}
        source = {v: random.randint(min_value, max_value) for v in ["X", "Y", "Z"]}

        p_base = causal_model.functions["P"](*[base[v] for v in p_parents])
        p_source = causal_model.functions["P"](*[source[v] for v in p_parents])

        o_base = causal_model.functions["O"](p_base, base[other_parent])
        o_counterfactual = causal_model.functions["O"](p_source, base[other_parent])

        if o_base == o_counterfactual:
            continue

        base_label = o_base - label_offset
        cf_label = o_counterfactual - label_offset
        if not (0 <= base_label < num_labels and 0 <= cf_label < num_labels):
            continue

        pairs.append({
            "base_prompt": prompt(base["X"], base["Y"], base["Z"]),
            "source_prompt": prompt(source["X"], source["Y"], source["Z"]),
            "base_values": base,
            "source_values": source,
            "base_answer": o_base,
            "counterfactual_answer": o_counterfactual,
        })

    return pairs


def cache_node_activation(model, inputs, node_type, layer):
    """Run `inputs` through the model and capture one node's activation. No grad needed."""
    captured = {}

    def hook(module, inp, out):
        captured["value"] = (inp[0] if node_type == "attn" else out).detach().clone()

    site = model.transformer.h[layer].attn.c_proj if node_type == "attn" else model.transformer.h[layer].mlp
    handle = site.register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return captured["value"]


def patched_forward(model, inputs, node_type, layer, head, source_value, n_heads):
    """Run `inputs` through the model, substituting one node's activation with `source_value`."""
    if node_type == "attn":
        def prehook(module, args):
            z = args[0].clone()
            b, s, h = z.shape
            head_dim = h // n_heads
            z = z.view(b, s, n_heads, head_dim)
            src = source_value.view(b, s, n_heads, head_dim)
            z[:, :, head, :] = src[:, :, head, :]
            return (z.view(b, s, h),) + args[1:]

        handle = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(prehook)
    else:
        def posthook(module, inp, out):
            return source_value

        handle = model.transformer.h[layer].mlp.register_forward_hook(posthook)

    with torch.no_grad():
        outputs = model(**inputs)
    handle.remove()
    return outputs.logits


def node_iia(model, tokenizer, pairs, node_type, layer, head, device, n_heads, label_offset, batch_size):
    correct = 0
    total = 0

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]

        base_inputs = encode(tokenizer, [p["base_prompt"] for p in chunk], device)
        source_inputs = encode(tokenizer, [p["source_prompt"] for p in chunk], device)

        if base_inputs["input_ids"].shape != source_inputs["input_ids"].shape:
            raise ValueError(
                "Base and source prompts tokenized to different lengths; keep "
                "--min_value/--max_value within a single digit width so positions line up."
            )

        source_value = cache_node_activation(model, source_inputs, node_type, layer)
        patched_logits = patched_forward(model, base_inputs, node_type, layer, head, source_value, n_heads)

        predicted = patched_logits.argmax(dim=-1)
        counterfactual = torch.tensor(
            [p["counterfactual_answer"] - label_offset for p in chunk], device=device
        )
        correct += (predicted == counterfactual).sum().item()
        total += len(chunk)

    return correct / total


def safe_filename(label):
    return "".join(c if c.isalnum() else "_" for c in label)


def run_for_causal_model(model, tokenizer, causal_model, label, args, device):
    n_layers = model.config.n_layer
    n_heads = model.config.n_head

    pairs = make_causal_pairs(
        causal_model, args.num_pairs, args.min_value, args.max_value, args.label_offset, model.config.num_labels
    )

    results = {
        "causal_model_label": label,
        "num_pairs": len(pairs),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "nodes": [],
    }

    for layer in range(n_layers):
        for head in range(n_heads):
            iia = node_iia(
                model, tokenizer, pairs, "attn", layer, head, device, n_heads, args.label_offset, args.batch_size
            )
            results["nodes"].append({"type": "attn", "layer": layer, "head": head, "iia": iia})
            print(f"[{label}] attn L{layer}H{head}: IIA={iia:.3f}")

        iia = node_iia(
            model, tokenizer, pairs, "mlp", layer, None, device, n_heads, args.label_offset, args.batch_size
        )
        results["nodes"].append({"type": "mlp", "layer": layer, "head": None, "iia": iia})
        print(f"[{label}] mlp  L{layer}: IIA={iia:.3f}")

    results["nodes"].sort(key=lambda r: r["iia"], reverse=True)
    return results


def plot_node_heatmap(results, output_dir):
    """Layers x (heads + MLP column) heatmap of real per-node IIA for one causal model."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping heatmap")
        return

    import numpy as np

    n_layers, n_heads = results["n_layers"], results["n_heads"]
    grid = np.zeros((n_layers, n_heads + 1))
    for row in results["nodes"]:
        col = row["head"] if row["type"] == "attn" else n_heads
        grid[row["layer"], col] = row["iia"]

    fig, ax = plt.subplots(figsize=(1.0 * (n_heads + 1) + 2, 0.5 * n_layers + 2))
    im = ax.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax, label="real IIA (patched)")
    ax.set_xticks(range(n_heads + 1))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)] + ["MLP"])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)])
    ax.set_xlabel("Component")
    ax.set_ylabel("Layer")
    ax.set_title(f"Per-node IIA for {results['causal_model_label']}")
    fig.tight_layout()

    path = os.path.join(output_dir, f"node_iia_{safe_filename(results['causal_model_label'])}_heatmap.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_comparison(all_results, output_dir):
    """Best-node-per-layer IIA across causal models, analogous to Figure 1 of the paper."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping comparison plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for results in all_results:
        n_layers = results["n_layers"]
        best_per_layer = [0.0] * n_layers
        for row in results["nodes"]:
            best_per_layer[row["layer"]] = max(best_per_layer[row["layer"]], row["iia"])
        ax.plot(range(n_layers), best_per_layer, marker="o", label=results["causal_model_label"])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Best single-node real IIA")
    ax.set_ylim(0, 1.05)
    ax.set_title("Best per-layer node IIA by causal model hypothesis")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "node_iia_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Real (non-approximated) per-node interchange intervention accuracy: "
                     "patch one attention head or MLP layer at a time and check whether the "
                     "output matches the causal model's do(P=P_source) prediction."
    )
    parser.add_argument("--model_name", default="mara589/arithmetic-gpt2")
    parser.add_argument("--causal_model_id", type=int, default=None, choices=[1, 2, 3],
                         help="1=(X+Y)+Z, 2=(X+Z)+Y, 3=X+(Y+Z). Omit to run all three.")
    parser.add_argument("--num_pairs", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_value", type=int, default=1)
    parser.add_argument("--max_value", type=int, default=9,
                         help="Keep base/source numbers the same digit width so prompts tokenize "
                              "to equal length (see the ValueError in node_iia otherwise).")
    parser.add_argument("--label_offset", type=int, default=3,
                         help="mara589/arithmetic-gpt2 has 28 output classes for sums 3..30 "
                              "(X, Y, Z each range 1..10 during fine-tuning); 3 is the minimum "
                              "possible X+Y+Z and must match model.config.num_labels - 1 + 3.")
    parser.add_argument("--output_dir", default="results/node_iia")
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

    print(f"Model has {model.config.num_labels} output classes "
          f"(expected 28, for sums 3..30, if --label_offset=3 is correct for this checkpoint).")

    arithmetic_family = ArithmeticCausalModels()
    model_infos = (
        {args.causal_model_id: arithmetic_family.causal_models[args.causal_model_id]}
        if args.causal_model_id is not None
        else arithmetic_family.causal_models
    )

    all_results = []
    for train_id, model_info in model_infos.items():
        causal_model = model_info["causal_model"]
        label = model_info["label"]

        results = run_for_causal_model(model, tokenizer, causal_model, label, args, device)
        all_results.append(results)

        out_path = os.path.join(args.output_dir, f"node_iia_{safe_filename(label)}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved: {out_path}")
        print(f"Top nodes for {label}:")
        for row in results["nodes"][:10]:
            tag = f"attn L{row['layer']}H{row['head']}" if row["type"] == "attn" else f"mlp  L{row['layer']}"
            print(f"  {tag}: IIA={row['iia']:.3f}")

        plot_node_heatmap(results, args.output_dir)

    if len(all_results) > 1:
        plot_comparison(all_results, args.output_dir)


if __name__ == "__main__":
    main()
