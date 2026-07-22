"""Boundless-DAS metrics: learned dimensionality (hard + soft) and the IIA-vs-k frontier.

run_das (Boundless variant) saves one intervenable per (causal model, layer) at
    <root>/cm_<train_id>/intervenable_<dim>_<layer>          (dim = 768 for Boundless)
Each BoundlessRotatedSpaceIntervention stores `intervention_boundaries` (a fraction of the
embedding dim) and a `temperature`. Two ways to read the learned dimensionality:

  * hard k = round(boundary_fraction * embed_dim)            -- the rounded cutoff
  * soft k = sum of the sigmoid boundary mask over the 768 dims   -- the honest, real
    number of dimensions the intervention actually uses (Boundless's mask is soft).

This script produces:
  (2) a per-layer plot of the learned dimension (soft, with hard shown dashed), one line
      per causal model, against dashed reference lines for the paper's fixed k.
  (1) [optional, with --reports_root] the IIA-vs-k efficiency frontier: each Boundless node
      as a point (learned k, IIA), showing whether Boundless matches IIA at fewer dims than
      the paper's fixed k.
"""

import argparse
import json
import os
import re

import torch
import pyvene as pv
from transformers import GPT2ForSequenceClassification

CKPT_RE = re.compile(r"^intervenable_(?P<dim>\d+)_(?P<layer>\d+)$")
CM_RE = re.compile(r"^cm_(?P<cm>\d+)$")
REPORT_RE = re.compile(r"^(?P<train>\d+)_report_layer_(?P<layer>\d+)_tkn_(?P<exp>.+)\.json$")
TESTDIR_RE = re.compile(r"^results_(?P<test>\d+)$")


def get_labels(causal_model_type):
    from causal_models import ArithmeticCausalModels, SimpleSummingCausalModels
    family = ArithmeticCausalModels() if causal_model_type == "arithmetic" else SimpleSummingCausalModels()
    return {tid: info["label"] for tid, info in family.causal_models.items()}


def sigmoid_boundary_mask(boundary_frac, temperature, embed_dim):
    """The soft mask Boundless applies over the embed_dim dimensions:
       sigmoid((j - 0)/temp) * sigmoid((cutoff - j)/temp),  cutoff = boundary_frac*embed_dim.
    Returns the mask tensor (length embed_dim)."""
    pop = torch.arange(embed_dim).float()
    cutoff = boundary_frac * embed_dim
    return torch.sigmoid((pop - 0.0) / temperature) * torch.sigmoid((cutoff - pop) / temperature)


def extract_dims(interv_model, embed_dim):
    """Return dict with boundary fraction, hard k, soft k, temperature — or None if the
    checkpoint isn't a Boundless intervention (no learned boundary)."""
    interv_key = list(interv_model.interventions.keys())[0]
    module = interv_model.interventions[interv_key]          # stock pyvene: no [0]
    if not hasattr(module, "intervention_boundaries"):
        return None

    frac = float(torch.clamp(module.intervention_boundaries.detach().cpu(), 1e-3, 1).flatten()[0])
    hard_k = round(frac * embed_dim)

    temp = None
    if hasattr(module, "temperature") and module.temperature is not None:
        t = module.temperature
        temp = float(t.detach().cpu().flatten()[0]) if torch.is_tensor(t) else float(t)
    if temp and temp > 0:
        soft_k = float(sigmoid_boundary_mask(frac, temp, embed_dim).sum())
    else:
        soft_k = float(hard_k)     # no temperature stored -> soft == hard

    return {"frac": frac, "hard_k": hard_k, "soft_k": soft_k, "temperature": temp}


def collect_dims(root, base_model, dim, embed_dim):
    """Walk <root>/cm_*/intervenable_<dim>_<layer> -> {cm_id: {layer: dims_dict}}."""
    data = {}
    for cm_name in sorted(os.listdir(root)):
        cm_match = CM_RE.match(cm_name)
        if not cm_match:
            continue
        cm_id = int(cm_match.group("cm"))
        cm_dir = os.path.join(root, cm_name)
        for ckpt_name in sorted(os.listdir(cm_dir)):
            m = CKPT_RE.match(ckpt_name)
            if not m or int(m.group("dim")) != dim:
                continue
            layer = int(m.group("layer"))
            path = os.path.join(cm_dir, ckpt_name)
            try:
                interv_model = pv.IntervenableModel.load(path, model=base_model)
                dims = extract_dims(interv_model, embed_dim)
            except Exception as e:
                print(f"  ! skipping {path}: {e}")
                continue
            if dims is None:
                print(f"  ! {path} not Boundless (no learned boundary); skipping")
                continue
            data.setdefault(cm_id, {})[layer] = dims
            print(f"  cm {cm_id} L{layer}: frac={dims['frac']:.4f} hard_k={dims['hard_k']} soft_k={dims['soft_k']:.1f}")
    return data


def collect_iia(reports_root):
    """Diagonal (train==test) IIA per (cm, layer) from save_results JSONs under reports_root.
    Returns {cm_id: {layer: iia}}. reports_root holds results_<test>/ subfolders."""
    iia = {}
    for test_dir in os.listdir(reports_root):
        md = TESTDIR_RE.match(test_dir)
        if not md:
            continue
        test_id = int(md.group("test"))
        for fname in os.listdir(os.path.join(reports_root, test_dir)):
            m = REPORT_RE.match(fname)
            if not m:
                continue
            train_id, layer = int(m.group("train")), int(m.group("layer"))
            if train_id != test_id:          # diagonal only
                continue
            with open(os.path.join(reports_root, test_dir, fname)) as f:
                acc = json.load(f).get("accuracy")
            if acc is not None:
                iia.setdefault(train_id, {})[layer] = float(acc)
    return iia


def plot_dimensions(data, labels, paper_ks, embed_dim, output_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, cm_id in enumerate(sorted(data)):
        layers = sorted(data[cm_id])
        soft = [data[cm_id][l]["soft_k"] for l in layers]
        hard = [data[cm_id][l]["hard_k"] for l in layers]
        color = cmap(i)
        ax.plot(layers, soft, marker="o", linewidth=2, color=color,
                label=f"{labels.get(cm_id, f'cm{cm_id}')} (soft)")
        ax.plot(layers, hard, marker="x", linewidth=1, linestyle=":", color=color, alpha=0.6)

    for fixed_k in paper_ks:
        ax.axhline(fixed_k, linestyle="--", color="gray", alpha=0.7, linewidth=1)
        ax.text(ax.get_xlim()[1], fixed_k, f" paper k={fixed_k}", va="center", ha="left",
                color="gray", fontsize=9)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Active dimensions k  (solid=soft, dotted=hard)")
    ax.set_ylim(0, embed_dim)
    ax.set_title("Boundless DAS: learned dimensionality per layer vs. paper's fixed k")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def plot_frontier(data, iia, labels, paper_ks, output_path, use_soft=True):
    """IIA-vs-k scatter: each Boundless node a point (learned k, IIA)."""
    import matplotlib.pyplot as plt

    key = "soft_k" if use_soft else "hard_k"
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10")
    plotted = 0
    for i, cm_id in enumerate(sorted(data)):
        xs, ys, ann = [], [], []
        for layer in sorted(data[cm_id]):
            if cm_id in iia and layer in iia[cm_id]:
                xs.append(data[cm_id][layer][key])
                ys.append(iia[cm_id][layer])
                ann.append(layer)
        if not xs:
            continue
        ax.scatter(xs, ys, color=cmap(i), label=labels.get(cm_id, f"cm{cm_id}"), zorder=3)
        for x, y, l in zip(xs, ys, ann):
            ax.annotate(f"L{l}", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        plotted += len(xs)

    if plotted == 0:
        print("  ! no (k, IIA) pairs matched between checkpoints and reports; skipping frontier")
        plt.close(fig)
        return

    for fixed_k in paper_ks:
        ax.axvline(fixed_k, linestyle="--", color="gray", alpha=0.6, linewidth=1)
        ax.text(fixed_k, ax.get_ylim()[1], f"paper k={fixed_k}", rotation=90,
                va="top", ha="right", color="gray", fontsize=8)

    ax.set_xlabel(f"Learned active dimensions k ({'soft' if use_soft else 'hard'})")
    ax.set_ylabel("IIA")
    ax.set_ylim(0, 1.02)
    ax.set_title("Boundless DAS efficiency frontier: IIA vs. learned k\n(up-and-left of the fixed-k lines = more compact than the paper)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Boundless-DAS dimensionality + IIA-vs-k frontier.")
    parser.add_argument("--model_id", default="mara589/arithmetic-gpt2")
    parser.add_argument("--root", default="results/arithmetic/intervenable_models",
                        help="Folder with cm_<id>/intervenable_<dim>_<layer> checkpoints.")
    parser.add_argument("--causal_model_type", choices=["arithmetic", "simple"], default="arithmetic")
    parser.add_argument("--dim", type=int, default=768, help="<dim> in the Boundless checkpoint names.")
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--paper_ks", type=int, nargs="*", default=[64, 128, 256])
    parser.add_argument("--reports_root", default=None,
                        help="Folder with results_<test>/ report JSONs (enables the IIA-vs-k frontier).")
    parser.add_argument("--output_dir", default=None, help="Where PNGs go (default: <root>).")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        raise FileNotFoundError(f"No checkpoint root at {args.root}")

    print(f"Loading base model {args.model_id} ...")
    base_model = GPT2ForSequenceClassification.from_pretrained(args.model_id)

    print(f"Reading Boundless checkpoints under {args.root} (dim={args.dim}) ...")
    data = collect_dims(args.root, base_model, args.dim, args.embed_dim)
    if not data:
        raise ValueError(f"No Boundless checkpoints (intervenable_{args.dim}_*) under {args.root}")

    labels = get_labels(args.causal_model_type)
    out_dir = args.output_dir or args.root
    os.makedirs(out_dir, exist_ok=True)

    # (2) dimensionality per layer
    plot_dimensions(data, labels, args.paper_ks, args.embed_dim,
                    os.path.join(out_dir, "boundless_dimensions.png"))

    # (1) efficiency frontier (needs IIA)
    if args.reports_root:
        print(f"Reading IIA reports under {args.reports_root} ...")
        iia = collect_iia(args.reports_root)
        plot_frontier(data, iia, labels, args.paper_ks,
                      os.path.join(out_dir, "boundless_efficiency_frontier.png"))

    print("\nLearned-dimension summary (hard / soft, out of {}):".format(args.embed_dim))
    for cm_id in sorted(data):
        row = ", ".join(f"L{l}:{data[cm_id][l]['hard_k']}/{data[cm_id][l]['soft_k']:.0f}"
                        for l in sorted(data[cm_id]))
        print(f"  {labels.get(cm_id, f'cm{cm_id}')}: {row}")


if __name__ == "__main__":
    main()
