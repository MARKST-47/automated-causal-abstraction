#!/usr/bin/env python3
"""
Boundless DAS - Extension Results
Comparing against DAS Baseline (Pislar et al., CLeaR 2025)

Converted from boundless_das_extension.ipynb into a parameterized script.

This script loads IIA result JSONs produced by the DAS / Boundless DAS runs,
builds the comparison figures, writes them to disk, and prints summary stats.

Typical usage
-------------
    python boundless_das_extension.py --base /path/to/Causal_Project

Override any of the defaults, e.g.:
    python boundless_das_extension.py \
        --base . \
        --k-new 768 --k-old 64 \
        --arith-models 1 2 3 \
        --simple-models 1 2 3 4 \
        --ref-layer 0 --num-layers 12 \
        --num-classes 28 \
        --show

Run `python boundless_das_extension.py --help` for the full list of inputs.
"""

import argparse
import json
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate Boundless DAS extension figures and stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--base", type=Path, default=Path("."),
                   help="Project root directory.")
    p.add_argument("--src", type=Path, default=None,
                   help="Path to the 'src' dir holding causal_models.py "
                        "(default: <base>/src).")
    p.add_argument("--old-dir", type=Path, default=None,
                   help="Baseline (k=old) arithmetic results dir "
                        "(default: <base>/baseline_results/arithmetic).")
    p.add_argument("--new-dir", type=Path, default=None,
                   help="Boundless DAS (k=new) arithmetic results dir "
                        "(default: <base>/results/arithmetic).")
    p.add_argument("--simple-dir", type=Path, default=None,
                   help="Simple/copy task results dir "
                        "(default: <base>/results/simple).")
    p.add_argument("--atp-dir", type=Path, default=None,
                   help="Attribution patching results dir "
                        "(default: <base>/results/attribution_patching_test).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Directory to write figures/tables to "
                        "(default: <new-dir>/plots).")

    # Run configuration
    p.add_argument("--k-new", type=int, default=768,
                   help="Subspace dim for the Boundless DAS run (file suffix tkn_K).")
    p.add_argument("--k-old", type=int, default=64,
                   help="Subspace dim for the baseline run (file suffix tkn_K).")
    p.add_argument("--arith-models", type=int, nargs="+", default=[1, 2, 3],
                   help="Arithmetic causal-model ids to load (self-IIA).")
    p.add_argument("--simple-models", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="Simple/copy causal-model ids to load (self-IIA).")
    p.add_argument("--baseline-model", type=int, default=1,
                   help="Which model id the baseline (k=old) run covers.")
    p.add_argument("--num-layers", type=int, default=12,
                   help="Number of transformer layers.")
    p.add_argument("--ref-layer", type=int, default=0,
                   help="Reference layer used for the cross-IIA heatmap.")
    p.add_argument("--num-classes", type=int, default=28,
                   help="Number of output classes (sets the chance line = 1/N).")
    p.add_argument("--atp-json-name", type=str, default="atp_arithmetic_xy.json",
                   help="Filename of the AtP JSON inside --atp-dir.")
    p.add_argument("--atp-img-name", type=str, default="atp_arithmetic_xy_heads.png",
                   help="Filename of the pre-rendered AtP heatmap inside --atp-dir.")

    # Output behaviour
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    p.add_argument("--show", action="store_true",
                   help="Display figures interactively (default: save only).")

    args = p.parse_args(argv)

    # Derive defaults that depend on --base / --new-dir
    if args.src is None:
        args.src = args.base / "src"
    if args.old_dir is None:
        args.old_dir = args.base / "baseline_results" / "arithmetic"
    if args.new_dir is None:
        args.new_dir = args.base / "results" / "arithmetic"
    if args.simple_dir is None:
        args.simple_dir = args.base / "results" / "simple"
    if args.atp_dir is None:
        args.atp_dir = args.base / "results" / "attribution_patching_test"
    if args.out_dir is None:
        args.out_dir = args.new_dir / "plots"

    return args


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_self_iia(results_root, cm_id, k, num_layers):
    """Self-IIA: trained on cm_id, tested on cm_id's own counterfactual data."""
    folder = results_root / f"results_{cm_id}"
    return {
        layer: json.load(
            open(folder / f"{cm_id}_report_layer_{layer}_tkn_{k}.json")
        )["accuracy"]
        for layer in range(num_layers)
    }


def load_cross_iia(results_root, train_id, test_id, k, num_layers):
    """Cross-IIA: trained on train_id, tested on test_id's data."""
    folder = results_root / f"results_{train_id}"
    return {
        layer: json.load(
            open(folder / f"{test_id}_report_layer_{layer}_tkn_{k}.json")
        )["accuracy"]
        for layer in range(num_layers)
    }


def load_baseline_iia(old_dir, model_id, k, num_layers):
    """Baseline run: only `model_id` self-IIA available."""
    folder = old_dir / f"results_{model_id}"
    return {
        layer: json.load(
            open(folder / f"{model_id}_report_layer_{layer}_tkn_{k}.json")
        )["accuracy"]
        for layer in range(num_layers)
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig1_baseline_vs_boundless(args, plt, cmap, layers, chance,
                               old_iia, new_self_iia, arith_labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Left - old baseline
    ax1.plot(layers, [old_iia[l] for l in layers], "o-", color=cmap(0),
             label=f"(X+Y)+Z  (k={args.k_old})", linewidth=2, markersize=6)
    ax1.axhline(chance, color="gray", ls="--", lw=1,
                label=f"Chance (1/{args.num_classes} ~ {chance:.3f})")
    ax1.set_title(f"Baseline DAS (k={args.k_old})\n"
                  f"cm_{args.baseline_model} only - others not trained", fontsize=11)
    ax1.set_xlabel("Layer"); ax1.set_ylabel("IIA")
    ax1.set_xticks(layers); ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.2)

    # Right - Boundless DAS
    for cm_id, iia in new_self_iia.items():
        ax2.plot(layers, [iia[l] for l in layers], "o-", color=cmap(cm_id - 1),
                 label=arith_labels.get(cm_id, f"cm_{cm_id}"),
                 linewidth=2, markersize=6)
    ax2.axhline(chance, color="gray", ls="--", lw=1, label="Chance")
    ax2.set_title(f"Boundless DAS (k={args.k_new})\n"
                  "All arithmetic models trained independently", fontsize=11)
    ax2.set_xlabel("Layer")
    ax2.set_xticks(layers); ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.2)

    plt.suptitle("Baseline DAS vs. Boundless DAS - IIA by Layer", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, plt, args, "fig1_baseline_vs_boundless.pdf")


def fig2_cm_comparison(args, plt, cmap, layers, chance,
                       old_iia, new_self_iia, arith_labels):
    bid = args.baseline_model
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(layers, [old_iia[l] for l in layers], "o--", color=cmap(0),
            label=f"(X+Y)+Z  k={args.k_old} fixed (Baseline DAS)",
            linewidth=2, markersize=6, alpha=0.7)
    ax.plot(layers, [new_self_iia[bid][l] for l in layers], "o-", color=cmap(0),
            label=f"(X+Y)+Z  k={args.k_new} learned (Boundless DAS)",
            linewidth=2, markersize=6)
    ax.axhline(chance, color="gray", ls=":", lw=1, label="Chance")

    ax.set_xlabel("Layer"); ax.set_ylabel("IIA")
    ax.set_title(f"cm_{bid} = {arith_labels.get(bid, '')}  -  "
                 f"k={args.k_old} Fixed vs. k={args.k_new} Boundless DAS")
    ax.set_xticks(layers); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, plt, args, "fig2_cm_comparison.pdf")

    best_old = max(old_iia.values())
    best_new = max(new_self_iia[bid].values())
    print(f"cm_{bid} peak IIA  -  k={args.k_old}: {best_old:.4f}   "
          f"k={args.k_new}: {best_new:.4f}   diff: {best_new - best_old:+.4f}")


def write_tables(args, pd, new_self_iia, old_iia, arith_labels):
    """The notebook used df.style (HTML only); here we save CSVs and print."""
    df_new = pd.DataFrame(
        {arith_labels.get(cm_id, f"cm_{cm_id}"): new_self_iia[cm_id]
         for cm_id in args.arith_models}
    ).rename_axis("Layer")
    df_old = pd.DataFrame(
        {f"(X+Y)+Z  k={args.k_old}": old_iia}
    ).rename_axis("Layer")

    new_path = args.out_dir / "table_boundless_iia.csv"
    old_path = args.out_dir / "table_baseline_iia.csv"
    df_new.to_csv(new_path)
    df_old.to_csv(old_path)

    print("\nBoundless DAS - IIA by layer (k=%d):" % args.k_new)
    print(df_new.round(4).to_string())
    print(f"  -> {new_path}")
    print("\nBaseline DAS - IIA by layer (k=%d):" % args.k_old)
    print(df_old.round(4).to_string())
    print(f"  -> {old_path}")


def fig3_cross_iia_heatmap(args, plt, np, layers, ref_layer,
                           new_dir, arith_labels):
    ids = args.arith_models
    n = len(ids)
    cross = np.zeros((n, n))
    for i, train_id in enumerate(ids):
        for j, test_id in enumerate(ids):
            cross[i, j] = load_cross_iia(
                new_dir, train_id, test_id, args.k_new, args.num_layers
            )[ref_layer]

    labels_list = [arith_labels.get(i, f"cm_{i}") for i in ids]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cross, cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="IIA")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"Test: {l}" for l in labels_list],
                       rotation=20, ha="right", fontsize=9)
    ax.set_yticklabels([f"Train: {l}" for l in labels_list], fontsize=9)
    ax.set_title(f"Cross-IIA Matrix - Boundless DAS "
                 f"(k={args.k_new}, layer {ref_layer})", fontsize=11)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cross[i, j]:.3f}", ha="center", va="center",
                    fontsize=11, color="black" if cross[i, j] > 0.3 else "white")

    plt.tight_layout()
    _save(fig, plt, args, "fig3_cross_iia_heatmap.pdf")

    print("Diagonal (self-IIA) should be highest in each row.")
    print("Off-diagonal near chance confirms probes are specific to "
          "their causal variable.")


def fig4_self_vs_cross(args, plt, cmap, layers, chance,
                       new_dir, new_self_iia, arith_labels):
    ids = args.arith_models
    fig, axes = plt.subplots(1, len(ids), figsize=(5 * len(ids), 4), sharey=True)
    if len(ids) == 1:
        axes = [axes]

    for idx, train_id in enumerate(ids):
        ax = axes[idx]
        ax.plot(layers, [new_self_iia[train_id][l] for l in layers], "o-",
                color=cmap(train_id - 1), linewidth=2, markersize=5,
                label=f"Self: {arith_labels.get(train_id, train_id)}")
        for test_id in ids:
            if test_id == train_id:
                continue
            cross_vals = [
                load_cross_iia(new_dir, train_id, test_id,
                               args.k_new, args.num_layers)[l]
                for l in layers
            ]
            ax.plot(layers, cross_vals, "^--", color=cmap(test_id - 1),
                    linewidth=1.2, markersize=4, alpha=0.6,
                    label=f"Cross: {arith_labels.get(test_id, test_id)}")

        ax.axhline(chance, color="gray", ls=":", lw=0.8, label="Chance")
        ax.set_title(f"Trained on: {arith_labels.get(train_id, train_id)}",
                     fontsize=10)
        ax.set_xlabel("Layer"); ax.set_xticks(layers)
        ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("IIA")
    plt.suptitle(f"Self vs. Cross-IIA - Boundless DAS (k={args.k_new})",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, plt, args, "fig4_self_vs_cross_all_layers.pdf")


def fig5_simple_vs_arithmetic(args, plt, cmap, layers, chance,
                              simp_self_iia, new_self_iia,
                              simple_labels, arith_labels):
    bid = args.baseline_model
    fig, ax = plt.subplots(figsize=(10, 5))

    for cm_id, iia in simp_self_iia.items():
        ax.plot(layers, [iia[l] for l in layers], "s-", color=cmap(cm_id + 2),
                label=simple_labels.get(cm_id, f"cm_{cm_id}"),
                linewidth=1.8, markersize=5)

    ax.plot(layers, [new_self_iia[bid][l] for l in layers], "o-", color=cmap(0),
            linewidth=2.5, markersize=7,
            label=f"{arith_labels.get(bid, '')} [arithmetic]", zorder=5)

    ax.axhline(chance, color="gray", ls=":", lw=0.8, label="Chance")
    ax.set_xlabel("Layer"); ax.set_ylabel("IIA")
    ax.set_title(f"Simple Copy Task vs. Arithmetic - "
                 f"Control Comparison (k={args.k_new})")
    ax.set_xticks(layers); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, plt, args, "fig5_simple_vs_arithmetic.pdf")


def fig6_attribution_patching(args, plt, np):
    atp_json_path = args.atp_dir / args.atp_json_name
    atp_img_path = args.atp_dir / args.atp_img_name

    atp_data = None
    if atp_json_path.exists():
        with open(atp_json_path) as f:
            atp_data = json.load(f)
        print("AtP JSON keys:", list(atp_data.keys())
              if isinstance(atp_data, dict) else type(atp_data))
    else:
        print("atp JSON not found at", atp_json_path)

    # Show the pre-generated AtP heatmap from the cluster run
    if atp_img_path.exists():
        import matplotlib.image as mpimg
        img = mpimg.imread(str(atp_img_path))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(img); ax.axis("off")
        ax.set_title("Attribution Patching - Which Heads Drive the "
                     "(X+Y) Computation?", fontsize=12)
        plt.tight_layout()
        _save(fig, plt, args, "fig6_atp_heatmap.pdf")
    else:
        print("AtP image not found - attempting re-plot from JSON")

    # Re-plot AtP from JSON if available and in the expected {layer:{head:score}} form
    if atp_data is not None:
        L = args.num_layers
        if isinstance(atp_data, dict) and all(str(i) in atp_data for i in range(L)):
            scores = np.zeros((L, L))
            for layer in range(L):
                for head in range(L):
                    scores[layer, head] = atp_data[str(layer)].get(str(head), 0.0)

            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(scores.T, cmap="Reds", aspect="auto")
            plt.colorbar(im, ax=ax, label="AtP Score")
            ax.set_xlabel("Layer"); ax.set_ylabel("Attention Head")
            ax.set_xticks(range(L)); ax.set_yticks(range(L))
            ax.set_title("Attribution Patching - Head Importance for (X+Y)+Z")
            plt.tight_layout()
            _save(fig, plt, args, "fig6_atp_replot.pdf")

            top_heads = sorted(
                [(layer, head, scores[layer, head])
                 for layer in range(L) for head in range(L)],
                key=lambda x: -x[2],
            )[:5]
            print("Top 5 heads by AtP score:")
            for layer, head, score in top_heads:
                print(f"  Layer {layer}, Head {head}: {score:.4f}")
        else:
            print("Unexpected AtP JSON format - inspect manually.")
            print(type(atp_data), str(atp_data)[:500])


def fig7_summary(args, plt, np, cmap, chance,
                 old_self_iia, new_self_iia, simp_self_iia, arith_labels):
    ids = args.arith_models
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: peak self-IIA comparison (baseline bar shown for every model that was trained)
    ax = axes[0]
    x = np.arange(len(ids))
    bar_old = [max(old_self_iia[i].values()) if i in old_self_iia else 0.0
               for i in ids]
    bar_new = [max(new_self_iia[cm_id].values()) for cm_id in ids]

    ax.bar(x - 0.2, bar_old, width=0.35,
           label=f"Baseline DAS k={args.k_old}", color=cmap(7), alpha=0.8)
    ax.bar(x + 0.2, bar_new, width=0.35,
           label=f"Boundless DAS k={args.k_new}", color=cmap(0), alpha=0.8)
    ax.axhline(chance, color="gray", ls=":", lw=1, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels([arith_labels.get(i, f"cm_{i}") for i in ids], fontsize=9)
    ax.set_ylabel("Peak IIA (best layer)")
    ax.set_title("Peak IIA per Causal Model")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis="y")

    # Right: number of models with meaningful IIA (above chance + 0.1 margin)
    threshold = chance + 0.1
    ax2 = axes[1]
    old_meaningful = sum(1 for iia in old_self_iia.values()
                         if max(iia.values()) > threshold)
    new_meaningful = sum(1 for iia in new_self_iia.values()
                         if max(iia.values()) > threshold)
    simp_meaningful = sum(1 for iia in simp_self_iia.values()
                          if max(iia.values()) > threshold)

    bars = ax2.bar(["Baseline DAS\n(arithmetic)", "Boundless DAS\n(arithmetic)",
                    "Boundless DAS\n(simple/copy)"],
                   [old_meaningful, new_meaningful, simp_meaningful],
                   color=[cmap(7), cmap(0), cmap(2)], alpha=0.85)
    for bar, val in zip(bars, [old_meaningful, new_meaningful, simp_meaningful]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 str(val), ha="center", fontsize=12, fontweight="bold")
    ax2.set_ylabel(f"Models with IIA > chance + 0.1")
    ax2.set_title("How Many Causal Models Were Successfully Localised?")
    ax2.set_ylim(0, 5); ax2.grid(True, alpha=0.2, axis="y")

    plt.suptitle("Summary - Baseline DAS vs. Boundless DAS", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, plt, args, "fig7_summary_comparison.pdf")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _save(fig, plt, args, filename):
    out = args.out_dir / filename
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"  saved {out}")
    if args.show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)

    # Configure matplotlib backend before importing pyplot.
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")  # headless / no display needed
    import matplotlib.pyplot as plt
    import matplotlib.cm as colormap
    import numpy as np
    import pandas as pd

    import warnings
    warnings.filterwarnings("ignore")

    # Make the user's src/ importable, then pull in label definitions.
    sys.path.insert(0, str(args.src))
    from causal_models import ArithmeticCausalModels, SimpleSummingCausalModels

    arith_labels = {i: info["label"]
                    for i, info in ArithmeticCausalModels().causal_models.items()}
    simple_labels = {i: info["label"]
                     for i, info in SimpleSummingCausalModels().causal_models.items()}
    print("Arithmetic models:", arith_labels)
    print("Simple models:", simple_labels)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cmap = colormap.get_cmap("tab10")
    layers = list(range(args.num_layers))
    chance = 1.0 / args.num_classes

    # --- Load IIA data ----------------------------------------------------- #
    old_self_iia = {}
    for cm_id in args.arith_models:
        try:
            old_self_iia[cm_id] = load_baseline_iia(
                args.old_dir, cm_id, args.k_old, args.num_layers)
        except FileNotFoundError:
            print(f"  (no baseline (k={args.k_old}) results for cm_{cm_id}; skipping)")
    if not old_self_iia:
        raise SystemExit(f"No baseline results found under {args.old_dir}")
    # fig1/fig2 still highlight one baseline model; use it if present, else the first found.
    old_iia = old_self_iia.get(args.baseline_model, next(iter(old_self_iia.values())))
    new_self_iia = {
        cm_id: load_self_iia(args.new_dir, cm_id, args.k_new, args.num_layers)
        for cm_id in args.arith_models
    }
    simp_self_iia = {
        cm_id: load_self_iia(args.simple_dir, cm_id, args.k_new, args.num_layers)
        for cm_id in args.simple_models
    }

    print(f"\nOld baseline (k={args.k_old}, {len(old_self_iia)} model(s)):")
    for cm_id, iia in old_self_iia.items():
        print(f"  cm_{cm_id} ({arith_labels.get(cm_id, '')}): "
              f"{min(iia.values()):.4f} - {max(iia.values()):.4f}")
    print(f"New Boundless DAS (k={args.k_new}, all models):")
    for cm_id, iia in new_self_iia.items():
        print(f"  cm_{cm_id} ({arith_labels.get(cm_id, '')}): "
              f"{min(iia.values()):.4f} - {max(iia.values()):.4f}")

    # --- Figures and tables ------------------------------------------------ #
    print("\n[Figure 1] Baseline vs. Boundless DAS")
    fig1_baseline_vs_boundless(args, plt, cmap, layers, chance,
                               old_iia, new_self_iia, arith_labels)

    print("\n[Figure 2] Direct comparison on the baseline model")
    fig2_cm_comparison(args, plt, cmap, layers, chance,
                       old_iia, new_self_iia, arith_labels)

    print("\n[Tables] IIA by layer")
    write_tables(args, pd, new_self_iia, old_iia, arith_labels)

    print("\n[Figure 3] Cross-IIA heatmap")
    fig3_cross_iia_heatmap(args, plt, np, layers, args.ref_layer,
                           args.new_dir, arith_labels)

    print("\n[Figure 4] Self vs. cross-IIA across all layers")
    fig4_self_vs_cross(args, plt, cmap, layers, chance,
                       args.new_dir, new_self_iia, arith_labels)

    print("\n[Figure 5] Simple/copy task control")
    fig5_simple_vs_arithmetic(args, plt, cmap, layers, chance,
                              simp_self_iia, new_self_iia,
                              simple_labels, arith_labels)

    print("\n[Figure 6] Attribution patching")
    fig6_attribution_patching(args, plt, np)

    print("\n[Figure 7] Summary")
    fig7_summary(args, plt, np, cmap, chance,
                 old_self_iia, new_self_iia, simp_self_iia, arith_labels)

    print(f"\nDone. All outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
