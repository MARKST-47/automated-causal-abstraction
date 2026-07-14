"""Per-layer cumulative per-head faithfulness, across layers.

Reads the per-head IIA reports written by run_das_per_node.py:
    <results_path>/<ctype>/per_node/head/results_<test>/<train>_report_layer_<L>_tkn_<k>_head_L<L>H<H>.json

For each causal model it aggregates the 12 head IIAs at a layer into one number
(sum = "cumulative faithfulness", or mean/max) and plots it across layers, one line per
causal model. Only the diagonal (train == test) is used. No full-stream / block comparison.
"""

import argparse
import json
import os
import re

import numpy as np

FILENAME_RE = re.compile(
    r"^(?P<train>\d+)_report_layer_(?P<layer>\d+)_tkn_"
    r"(?P<k>\d+)_(?P<gran>[a-z]+)_L(?P<layer2>\d+)(H(?P<head>\d+))?\.json$"
)
TESTDIR_RE = re.compile(r"^results_(?P<test>\d+)$")


def get_labels(causal_model_type):
    from causal_models import ArithmeticCausalModels, SimpleSummingCausalModels
    fam = ArithmeticCausalModels() if causal_model_type == "arithmetic" else SimpleSummingCausalModels()
    return {tid: info["label"] for tid, info in fam.causal_models.items()}


def parse_heads(results_path, ctype, k=None):
    """Return {train_id: {layer: [head iia, ...]}} on the diagonal (train == test)."""
    base = os.path.join(results_path, ctype, "per_node", "head")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No per-head results at {base}")
    out = {}
    for test_dir in os.listdir(base):
        md = TESTDIR_RE.match(test_dir)
        if not md:
            continue
        test_id = int(md.group("test"))
        for fname in os.listdir(os.path.join(base, test_dir)):
            m = FILENAME_RE.match(fname)
            if not m or m.group("gran") != "head":
                continue
            train_id, layer = int(m.group("train")), int(m.group("layer"))
            if train_id != test_id:                        # diagonal only
                continue
            if k is not None and int(m.group("k")) != k:
                continue
            with open(os.path.join(base, test_dir, fname)) as f:
                acc = json.load(f).get("accuracy")
            if acc is not None:
                out.setdefault(train_id, {}).setdefault(layer, []).append(float(acc))
    return out


def aggregate(head_data, how):
    """{model: {layer: [head iias]}} -> {model: {layer: scalar}} using sum/mean/max."""
    fn = {"sum": np.sum, "mean": np.mean, "max": np.max}[how]
    return {m: {l: float(fn(v)) for l, v in layers.items()} for m, layers in head_data.items()}


def main():
    parser = argparse.ArgumentParser(description="Cumulative per-head faithfulness across layers.")
    parser.add_argument("--results_path", default="results/")
    parser.add_argument("--causal_model_type", choices=["arithmetic", "simple"], default="arithmetic")
    parser.add_argument("--head_k", type=int, default=64, help="k of the per-head runs (default 64).")
    parser.add_argument("--aggregate", choices=["sum", "mean", "max"], default="sum",
                        help="How to combine the 12 head IIAs into one per-layer number.")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    head_raw = parse_heads(args.results_path, args.causal_model_type, args.head_k)
    if not head_raw:
        raise ValueError("No per-head results found (granularity 'head').")

    agg = aggregate(head_raw, args.aggregate)
    labels = get_labels(args.causal_model_type)
    out_dir = args.output_dir or os.path.join(args.results_path, args.causal_model_type, "per_node", "plots")
    os.makedirs(out_dir, exist_ok=True)

    ylabel = {"sum": "cumulative per-head IIA (Σ over heads)",
              "mean": "mean per-head IIA",
              "max": "best single-head IIA"}[args.aggregate]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cmap = plt.get_cmap("tab10")
    all_layers = set()
    for i, m in enumerate(sorted(agg)):
        layers = sorted(agg[m])
        all_layers.update(layers)
        ys = [agg[m][l] for l in layers]
        ax.plot(layers, ys, marker="o", linewidth=2, color=cmap(i), label=labels.get(m, f"cm{m}"))

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(all_layers))
    ax.set_ylim(bottom=0)
    ax.set_title(f"Per-head faithfulness across layers ({args.aggregate})")
    ax.legend(title="Causal model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(out_dir, f"cumulative_faithfulness_{args.aggregate}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")

    print(f"\nPer-layer {args.aggregate} of head IIA:")
    for m in sorted(agg):
        row = ", ".join(f"L{l}:{agg[m][l]:.2f}" for l in sorted(agg[m]))
        print(f"  {labels.get(m, f'cm{m}')}: {row}")


if __name__ == "__main__":
    main()
