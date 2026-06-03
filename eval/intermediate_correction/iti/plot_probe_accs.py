"""
Plot per-head probe performance from probe_accs.npy (honest_llama-style).

Produces, for a given sweep dir:
    probe_accs_heatmap.png   layer x head val-accuracy heatmap, with top-K heads marked
    probe_accs_hist.png      distribution of all heads' val accuracy + top-K cutoff line

Usage:
    python plot_probe_accs.py --sweep-dir results/llavamed/sweep --num-heads 48
    python plot_probe_accs.py --sweep-dir results/llavamed/sweep --num-heads 16,32,48,64,80,96
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def top_k_heads(val_accs, k):
    """Return list of (layer, head) for the top-k by val accuracy."""
    L, H = val_accs.shape
    flat = np.argsort(val_accs.flatten())[::-1][:k]
    return [(int(i // H), int(i % H)) for i in flat]


def plot_heatmap(val_accs, k_marks, out_path, title):
    L, H = val_accs.shape
    # Wider for many heads; taller for many layers.
    fig, ax = plt.subplots(figsize=(max(6, H * 0.28), max(5, L * 0.28)))
    im = ax.imshow(val_accs, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Val accuracy")

    # Mark the largest top-K set with red squares
    if k_marks:
        ys = [h[0] for h in k_marks]
        xs = [h[1] for h in k_marks]
        ax.scatter(xs, ys, marker="s", s=18, facecolors="none",
                   edgecolors="red", linewidths=1.0,
                   label=f"top-{len(k_marks)} heads")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_hist(val_accs, k_list, out_path, title):
    flat = val_accs.flatten()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(flat, bins=50, color="steelblue", alpha=0.8)
    ax.set_xlabel("Per-head val accuracy")
    ax.set_ylabel("# heads")
    ax.set_title(title)
    # Cutoff line for each K (the accuracy of the K-th best head)
    sorted_desc = np.sort(flat)[::-1]
    colors = plt.cm.autumn(np.linspace(0, 0.8, len(k_list)))
    for K, c in zip(k_list, colors):
        if K <= len(sorted_desc):
            cutoff = sorted_desc[K - 1]
            ax.axvline(cutoff, color=c, linestyle="--", linewidth=1.2,
                       label=f"K={K} cutoff={cutoff:.3f}")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="chance (0.5)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--num-heads", default="48",
                    help="K value(s) to mark — comma-separated, e.g. 16,32,48,64,80,96")
    ap.add_argument("--out-dir", default=None, help="default: <sweep-dir>")
    args = ap.parse_args()

    acc_path = os.path.join(args.sweep_dir, "probe_accs.npy")
    if not os.path.exists(acc_path):
        raise SystemExit(f"Not found: {acc_path} (probe training may not have finished yet)")
    val_accs = np.load(acc_path)
    L, H = val_accs.shape
    k_list = [int(x) for x in str(args.num_heads).split(",") if x.strip()]
    out_dir = args.out_dir or args.sweep_dir

    model = os.path.basename(os.path.dirname(args.sweep_dir.rstrip("/")))
    print(f"{model}: probe_accs shape={val_accs.shape}  "
          f"mean={val_accs.mean():.4f}  max={val_accs.max():.4f}  "
          f"min={val_accs.min():.4f}")

    # Mark the largest K on the heatmap
    k_max = max(k_list)
    marks = top_k_heads(val_accs, k_max)

    # Which layers do the top heads concentrate in?
    layer_counts = np.bincount([m[0] for m in marks], minlength=L)
    top_layers = np.argsort(layer_counts)[::-1][:5]
    print(f"top-{k_max} heads concentrate in layers: "
          + ", ".join(f"L{l}({layer_counts[l]})" for l in top_layers if layer_counts[l] > 0))

    plot_heatmap(val_accs, marks, os.path.join(out_dir, "probe_accs_heatmap.png"),
                 f"{model}: per-head probe val accuracy (top-{k_max} marked)")
    plot_hist(val_accs, k_list, os.path.join(out_dir, "probe_accs_hist.png"),
              f"{model}: per-head probe val accuracy distribution")


if __name__ == "__main__":
    main()
