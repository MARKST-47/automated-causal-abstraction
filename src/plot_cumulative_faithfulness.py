"""Per-layer cumulative per-head faithfulness vs. full-stream (block) DAS.

Reads the per-node IIA reports written by run_das_per_node.py:
    <results_path>/<ctype>/per_node/head/results_<test>/<train>_report_layer_<L>_tkn_<k>_head_L<L>H<H>.json
    <results_path>/<ctype>/per_node/block/results_<test>/<train>_report_layer_<L>_tkn_<k>_block_L<L>.json

For each causal model it aggregates the 12 head IIAs at a layer into one number
(sum = "cumulative faithfulness", or mean/max), plots it across layers, and overlays the
IIA of DAS trained on the full residual stream (block) at the same layers. This answers:
does the layer-wise profile of the heads track where the full-stream rotation finds P?

Both series are also shown normalised to their own max so the *shape* alignment is visible
despite different scales, and Pearson/Spearman correlations are printed and put in the title.
Only the diagonal (train == test) is used.
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


def parse_granularity(results_path, ctype, granularity, k=None):
    """Return {train_id: {layer: [iia, ...]}} on the diagonal (train==test). For 'head' the
    list holds one entry per head; for 'block' it holds a single entry."""
    base = os.path.join(results_path, ctype, "per_node", granularity)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No results at {base}")
    out = {}
    for test_dir in os.listdir(base):
        md = TESTDIR_RE.match(test_dir)
        if not md:
            continue
        test_id = int(md.group("test"))
        for fname in os.listdir(os.path.join(base, test_dir)):
            m = FILENAME_RE.match(fname)
            if not m or m.group("gran") != granularity:
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


def aggregate_heads(head_data, how):
    """{model: {layer: [head iias]}} -> {model: {layer: scalar}} using sum/mean/max."""
    fn = {"sum": np.sum, "mean": np.mean, "max": np.max}[how]
    return {m: {l: float(fn(v)) for l, v in layers.items()} for m, layers in head_data.items()}


def block_scalar(block_data):
    """{model: {layer: [iia]}} -> {model: {layer: scalar}} (average if several k matched)."""
    return {m: {l: float(np.mean(v)) for l, v in layers.items()} for m, layers in block_data.items()}


def _pearson(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return _pearson(ra, rb)


def plot_model(ax, label, head_layer, block_layer, how):
    """Twin-axis: cumulative per-head (bars, left) + full-stream block IIA (line, right).
    X-axis spans every layer the heads cover; block is overlaid wherever it exists."""
    layers = sorted(head_layer)                            # all head layers, not the intersection
    head_y = [head_layer.get(l, np.nan) for l in layers]
    block_y = [block_layer.get(l, np.nan) for l in layers]

    common = [l for l in layers if l in head_layer and l in block_layer]
    if len(common) >= 2:
        hv = [head_layer[l] for l in common]
        bv = [block_layer[l] for l in common]
        r, rho = _pearson(hv, bv), _spearman(hv, bv)
        corr = f"  |  Pearson r={r:.2f}, Spearman ρ={rho:.2f}"
    else:
        corr = ""

    import matplotlib.pyplot as plt  # noqa: F401 (ensures backend chosen before twinx)
    ax.bar(layers, head_y, color="#4C72B0", alpha=0.75, label=f"Σ per-head IIA ({how})")
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"per-head IIA ({how})", color="#4C72B0")
    ax.set_xticks(layers)

    ax2 = ax.twinx()
    ax2.plot(layers, block_y, color="#C44E52", marker="o", linewidth=2, label="full-stream (block) IIA")
    ax2.set_ylabel("full-stream IIA", color="#C44E52")
    ax2.set_ylim(0, max([b for b in block_y if not np.isnan(b)] + [0.01]) * 1.15)

    ax.set_title(f"{label}{corr}", fontsize=10)
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labs, loc="upper right", fontsize=7)


def plot_normalised(ax, label, head_layer, block_layer, how):
    """Both series scaled to their own max, overlaid on one axis (shape comparison).
    Each series spans its own layers, so a partial block sweep still overlays cleanly."""
    hl = sorted(head_layer)
    if not hl:
        return
    hv = np.array([head_layer[l] for l in hl], float)
    hn = hv / hv.max() if hv.max() > 0 else hv
    ax.plot(hl, hn, color="#4C72B0", marker="s", linewidth=2, label=f"per-head ({how}), norm.")

    bl = sorted(block_layer)
    if bl:
        bv = np.array([block_layer[l] for l in bl], float)
        bn = bv / bv.max() if bv.max() > 0 else bv
        ax.plot(bl, bn, color="#C44E52", marker="o", linewidth=2, label="full-stream, norm.")

    ax.set_xlabel("Layer")
    ax.set_ylabel("normalised to own max")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(hl)
    ax.set_title(label, fontsize=10)
    ax.legend(loc="upper right", fontsize=7)


def main():
    parser = argparse.ArgumentParser(description="Per-layer cumulative per-head faithfulness vs full-stream DAS.")
    parser.add_argument("--results_path", default="results/")
    parser.add_argument("--causal_model_type", choices=["arithmetic", "simple"], default="arithmetic")
    parser.add_argument("--head_k", type=int, default=64, help="k of the per-head runs (default 64).")
    parser.add_argument("--block_k", type=int, default=None,
                        help="k of the full-stream runs (default: any; averaged if several).")
    parser.add_argument("--aggregate", choices=["sum", "mean", "max"], default="sum",
                        help="How to combine the 12 head IIAs into one per-layer number.")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    head_raw = parse_granularity(args.results_path, args.causal_model_type, "head", args.head_k)
    block_raw = parse_granularity(args.results_path, args.causal_model_type, "block", args.block_k)
    if not head_raw:
        raise ValueError("No per-head results found (granularity 'head').")
    if not block_raw:
        print("! No block (full-stream) results found — plotting per-head only, no comparison.")

    head_agg = aggregate_heads(head_raw, args.aggregate)
    block_agg = block_scalar(block_raw)
    labels = get_labels(args.causal_model_type)
    out_dir = args.output_dir or os.path.join(args.results_path, args.causal_model_type, "per_node", "plots")
    os.makedirs(out_dir, exist_ok=True)

    import matplotlib.pyplot as plt
    models = sorted(head_agg)

    # Figure 1: twin-axis (absolute), one subplot per model.
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4), squeeze=False)
    for ax, m in zip(axes[0], models):
        plot_model(ax, labels.get(m, f"cm{m}"), head_agg.get(m, {}), block_agg.get(m, {}), args.aggregate)
    fig.suptitle("Cumulative per-head faithfulness vs. full-stream DAS", y=1.02)
    fig.tight_layout()
    p1 = os.path.join(out_dir, f"cumulative_faithfulness_{args.aggregate}.png")
    fig.savefig(p1, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p1}")

    # Figure 2: normalised overlay (shape alignment), one subplot per model.
    if block_agg:
        fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4), squeeze=False)
        for ax, m in zip(axes[0], models):
            plot_normalised(ax, labels.get(m, f"cm{m}"), head_agg.get(m, {}), block_agg.get(m, {}), args.aggregate)
        fig.suptitle("Layer profile alignment (normalised): heads vs. full stream", y=1.02)
        fig.tight_layout()
        p2 = os.path.join(out_dir, f"cumulative_faithfulness_{args.aggregate}_normalised.png")
        fig.savefig(p2, dpi=200, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {p2}")

    # Correlation summary
    print("\nLayer-profile alignment (per-head cumulative vs full-stream):")
    for m in models:
        common = sorted(set(head_agg.get(m, {})) & set(block_agg.get(m, {})))
        if len(common) >= 2:
            hv = [head_agg[m][l] for l in common]
            bv = [block_agg[m][l] for l in common]
            print(f"  {labels.get(m, f'cm{m}')}: Pearson r={_pearson(hv, bv):+.2f}, "
                  f"Spearman ρ={_spearman(hv, bv):+.2f}  ({len(common)} layers)")
        else:
            print(f"  {labels.get(m, f'cm{m}')}: not enough overlapping layers for correlation")


if __name__ == "__main__":
    main()
