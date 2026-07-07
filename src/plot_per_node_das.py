"""Aggregate and plot per-node DAS IIA produced by run_das_per_node.py.

run_das_per_node.py writes, per node, a sklearn classification_report JSON via
utils.save_results at:

    results/<causal_model_type>/per_node/<granularity>/results_<test_id>/
        <train_id>_report_layer_<layer>_tkn_<k>_<granularity>_L<layer>[H<head>].json

where the report's "accuracy" field is the Interchange Intervention Accuracy (IIA).

By default we plot the *diagonal* (train_id == test_id): the IIA of a node that was
aligned to P for causal model M, evaluated on M's own do(P) counterfactuals -- i.e.
"how faithfully does this node represent P under hypothesis M". Off-diagonal entries
(cross-model generalization) are kept in the records and can be plotted with --all_pairs.

    head  granularity -> per-(train model, k) heatmap over layers x heads
    block/mlp/attention -> IIA-vs-layer curves (one line per k), plus an across-model
                            comparison at each k (analogous to the paper's Figure 1).
"""

import argparse
import json
import os
import re

FILENAME_RE = re.compile(
    r"^(?P<train>\d+)_report_layer_(?P<layer>\d+)_tkn_"
    r"(?P<k>\d+)_(?P<gran>[a-z]+)_L(?P<layer2>\d+)(H(?P<head>\d+))?\.json$"
)
TESTDIR_RE = re.compile(r"^results_(?P<test>\d+)$")


def get_labels(causal_model_type):
    from causal_models import ArithmeticCausalModels, SimpleSummingCausalModels
    family = ArithmeticCausalModels() if causal_model_type == "arithmetic" else SimpleSummingCausalModels()
    return {tid: info["label"] for tid, info in family.causal_models.items()}


def parse_records(results_path, causal_model_type, granularity):
    base = os.path.join(results_path, causal_model_type, "per_node", granularity)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No per-node DAS results at {base}")

    records = []
    for test_dir in os.listdir(base):
        m_dir = TESTDIR_RE.match(test_dir)
        if not m_dir:
            continue
        test_id = int(m_dir.group("test"))

        for fname in os.listdir(os.path.join(base, test_dir)):
            m = FILENAME_RE.match(fname)
            if not m or m.group("gran") != granularity:
                continue
            with open(os.path.join(base, test_dir, fname)) as f:
                report = json.load(f)
            iia = report.get("accuracy")
            if iia is None:
                continue
            records.append({
                "train": int(m.group("train")),
                "test": test_id,
                "layer": int(m.group("layer")),
                "head": int(m.group("head")) if m.group("head") is not None else None,
                "k": int(m.group("k")),
                "iia": float(iia),
            })
    if not records:
        raise ValueError(f"No parseable per-node result files under {base}")
    return records


def safe_label(label):
    return "".join(c if c.isalnum() else "_" for c in label)


def plot_head_heatmaps(records, labels, granularity, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    n_layers = max(r["layer"] for r in records) + 1
    n_heads = max(r["head"] for r in records) + 1

    for train_id in sorted({r["train"] for r in records}):
        for k in sorted({r["k"] for r in records}):
            grid = np.full((n_layers, n_heads), np.nan)
            for r in records:
                if r["train"] == train_id and r["test"] == train_id and r["k"] == k:
                    grid[r["layer"], r["head"]] = r["iia"]

            fig, ax = plt.subplots(figsize=(0.7 * n_heads + 3, 0.5 * n_layers + 2))
            im = ax.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="viridis")
            fig.colorbar(im, ax=ax, label="IIA (trained DAS rotation)")
            ax.set_xticks(range(n_heads)); ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
            ax.set_yticks(range(n_layers)); ax.set_yticklabels([f"L{l}" for l in range(n_layers)])
            ax.set_xlabel("Head"); ax.set_ylabel("Layer")
            ax.set_title(f"Per-head DAS IIA  |  model {labels.get(train_id, train_id)}  |  k={k}")
            fig.tight_layout()
            path = os.path.join(output_dir, f"perhead_das_{safe_label(labels.get(train_id, str(train_id)))}_k{k}.png")
            fig.savefig(path, dpi=200); plt.close(fig)
            print(f"Saved: {path}")


def plot_layer_curves(records, labels, granularity, output_dir):
    import matplotlib.pyplot as plt

    ks = sorted({r["k"] for r in records})
    train_ids = sorted({r["train"] for r in records})

    # (1) per causal model: IIA vs layer, one line per k (diagonal train==test)
    for train_id in train_ids:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k in ks:
            pts = sorted(
                (r["layer"], r["iia"]) for r in records
                if r["train"] == train_id and r["test"] == train_id and r["k"] == k
            )
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=f"k={k}")
        ax.set_xlabel("Layer"); ax.set_ylabel("IIA"); ax.set_ylim(0, 1.05)
        ax.set_title(f"Per-{granularity} DAS IIA vs layer  |  model {labels.get(train_id, train_id)}")
        ax.legend(); fig.tight_layout()
        path = os.path.join(output_dir, f"per{granularity}_das_{safe_label(labels.get(train_id, str(train_id)))}.png")
        fig.savefig(path, dpi=200); plt.close(fig)
        print(f"Saved: {path}")

    # (2) across causal models at each k (paper Figure 1 style)
    for k in ks:
        fig, ax = plt.subplots(figsize=(8, 5))
        for train_id in train_ids:
            pts = sorted(
                (r["layer"], r["iia"]) for r in records
                if r["train"] == train_id and r["test"] == train_id and r["k"] == k
            )
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=labels.get(train_id, str(train_id)))
        ax.set_xlabel("Layer"); ax.set_ylabel("IIA"); ax.set_ylim(0, 1.05)
        ax.set_title(f"Per-{granularity} DAS IIA by causal model  |  k={k}")
        ax.legend(); fig.tight_layout()
        path = os.path.join(output_dir, f"per{granularity}_das_compare_k{k}.png")
        fig.savefig(path, dpi=200); plt.close(fig)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot per-node DAS IIA from run_das_per_node.py outputs.")
    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument('--causal_model_type', type=str, choices=['arithmetic', 'simple'], default='arithmetic')
    parser.add_argument('--granularity', type=str, choices=['block', 'mlp', 'attention', 'head'], default='mlp')
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Where to write PNGs (default: alongside the results, in .../plots).")
    args = parser.parse_args()

    records = parse_records(args.results_path, args.causal_model_type, args.granularity)
    labels = get_labels(args.causal_model_type)

    output_dir = args.output_dir or os.path.join(
        args.results_path, args.causal_model_type, "per_node", args.granularity, "plots"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loaded {len(records)} per-node result files for granularity '{args.granularity}'.")
    if args.granularity == "head":
        plot_head_heatmaps(records, labels, args.granularity, output_dir)
    else:
        plot_layer_curves(records, labels, args.granularity, output_dir)


if __name__ == "__main__":
    main()
