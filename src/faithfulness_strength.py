"""Faithfulness-strength trade-off (the paper's Figure 2) from per-input IIA graphs.

construct_graph.py saves, per (causal model label, low_rank_dimension, layer), a
`<label>_graph_<lrd>_<layer>.pt` file: an N x N tensor (N = 1000 inputs) whose entry [i][j]
is the interchange-intervention accuracy of that model on the input pair (base_i, source_j).
A model perfectly abstracts an input region when the subgraph over those inputs has average
edge weight 1.

This script implements the greedy input-space partitioning (Algorithm 2 in the paper) at a
range of faithfulness thresholds lambda, then plots STRENGTH (fraction of inputs assigned to
a non-trivial model) vs. lambda. Run it once per dimensionality setting (e.g. the paper's
fixed k=256, and your Boundless run) and overlay the curves to show whether Boundless keeps
the same strength at each faithfulness level with far fewer dimensions.

STRENGTH definition (paper Definition 2): 1 - |inputs on the trivial model| / N. The trivial
model is the full-sum hypothesis that abstracts everything perfectly but explains nothing
(default label "(X+Y+Z)").
"""

import argparse
import os

import numpy as np
import torch


def subgraph_iia(W, nodes):
    """Average edge weight (IIA) of the subgraph induced by `nodes` on adjacency matrix W."""
    if len(nodes) < 2:
        return 1.0                      # a single input is trivially 'explained'
    idx = np.fromiter(nodes, dtype=int)
    sub = W[np.ix_(idx, idx)]
    total = sub.sum() - np.trace(sub)   # off-diagonal sum (both directions)
    pairs = len(nodes) * (len(nodes) - 1)
    return float(total / pairs)


def greedy_region(W, degree_order, available, lam):
    """Grow a subgraph from highest-degree nodes while its IIA stays >= lam. Returns the
    node set (subset of `available`)."""
    chosen = []
    for node in degree_order:
        if node not in available:
            continue
        trial = chosen + [node]
        if subgraph_iia(W, trial) >= lam:
            chosen = trial
    return set(chosen)


def partition_at_lambda(graphs, lam, n_nodes, trivial_label):
    """Greedily assign inputs to models at faithfulness lambda. Returns (strength, assignment
    dict label->count). Non-trivial models each grab the largest region they can explain at
    >= lam; whatever is left goes to the trivial model."""
    assigned = set()
    counts = {}

    candidate_labels = [l for l in graphs if l != trivial_label]
    # Precompute degree orderings once (thresholded at lam for a meaningful degree).
    degree_orders = {}
    for label in candidate_labels:
        W = graphs[label]
        deg = ((W >= lam).sum(axis=1))
        degree_orders[label] = list(np.argsort(-deg))

    while len(assigned) < n_nodes:
        best_label, best_region = None, set()
        available = set(range(n_nodes)) - assigned
        for label in candidate_labels:
            region = greedy_region(graphs[label], degree_orders[label], available, lam)
            if len(region) > len(best_region):
                best_label, best_region = label, region
        if not best_region:
            break                        # no non-trivial model can explain any remaining input
        counts[best_label] = counts.get(best_label, 0) + len(best_region)
        assigned |= best_region

    trivial_count = n_nodes - len(assigned)
    if trivial_count:
        counts[trivial_label] = counts.get(trivial_label, 0) + trivial_count

    strength = 1.0 - trivial_count / n_nodes
    return strength, counts


def load_graphs(graphs_dir, lrd, layer, labels):
    """Load {label}_graph_{lrd}_{layer}.pt for each requested label."""
    graphs = {}
    for label in labels:
        path = os.path.join(graphs_dir, f"{label}_graph_{lrd}_{layer}.pt")
        if not os.path.exists(path):
            print(f"  ! missing {path}; skipping this model")
            continue
        W = torch.load(path, map_location="cpu")
        W = W.numpy() if hasattr(W, "numpy") else np.asarray(W)
        np.fill_diagonal(W, 0.0)
        graphs[label] = W
    return graphs


def default_labels(causal_model_type):
    from causal_models import ArithmeticCausalModels, SimpleSummingCausalModels
    labels = [info["label"] for info in ArithmeticCausalModels().causal_models.values()]
    if causal_model_type == "arithmetic":
        # include the simple-summing models too (construct_graph merges both families)
        labels += [info["label"] for info in SimpleSummingCausalModels().causal_models.values()]
    return labels


def main():
    parser = argparse.ArgumentParser(description="Faithfulness-strength trade-off from IIA graphs.")
    parser.add_argument("--graphs_dir", default="results/graphs",
                        help="Folder with <label>_graph_<lrd>_<layer>.pt files.")
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--low_rank_dimension", type=int, default=256,
                        help="lrd in the graph filenames (use your Boundless dim, e.g. 768, for the Boundless curve).")
    parser.add_argument("--causal_model_type", choices=["arithmetic", "simple"], default="arithmetic")
    parser.add_argument("--trivial_label", default="(X+Y+Z)",
                        help="The trivial full-sum model that abstracts everything but explains nothing.")
    parser.add_argument("--lambdas", type=float, nargs="*", default=[0.8, 0.9, 0.95, 1.0])
    parser.add_argument("--n_nodes", type=int, default=1000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    labels = default_labels(args.causal_model_type)
    print(f"Loading graphs (lrd={args.low_rank_dimension}, layer={args.layer}) for {len(labels)} models ...")
    graphs = load_graphs(args.graphs_dir, args.low_rank_dimension, args.layer, labels)
    if args.trivial_label not in graphs and graphs:
        print(f"  ! trivial model '{args.trivial_label}' graph not found; leftover inputs still "
              f"count as trivial, but check --trivial_label matches your labels.")
    if not graphs:
        raise ValueError(f"No graphs loaded from {args.graphs_dir}")

    curve = []
    for lam in sorted(args.lambdas):
        strength, counts = partition_at_lambda(graphs, lam, args.n_nodes, args.trivial_label)
        curve.append((lam, strength))
        breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"  lambda={lam:.2f}  strength={strength:.3f}   [{breakdown}]")

    import matplotlib.pyplot as plt
    xs, ys = zip(*curve)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, marker="o", linewidth=2,
            label=f"lrd={args.low_rank_dimension}, layer={args.layer}")
    ax.set_xlabel("Faithfulness threshold  λ")
    ax.set_ylabel("Strength (fraction of inputs on non-trivial models)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Faithfulness-strength trade-off (combined causal models)")
    ax.legend()
    fig.tight_layout()
    out = args.output or os.path.join(args.graphs_dir, f"faithfulness_strength_lrd{args.low_rank_dimension}_layer{args.layer}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
