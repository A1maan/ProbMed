"""
Comparison plots: LR at last layer (offline) vs vector correction at intermediate layer (online).
Covers LLaVA-Med, CheXagent, MedGemma.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "LLaVA-Med\n(layer 15)": {
        "lr_file":  "llavamed/results/offline_correction_layer15/results.json",
        "vc_file":  "llavamed/results/vector_correction_layer15/results.json",
        "color":    "#4C72B0",
    },
    "CheXagent\n(layer 21)": {
        "lr_file":  "chexagent/results/offline_correction_layer21/results.json",
        "vc_file":  "chexagent/results/vector_correction_layer21_5k10ep/results.json",
        "color":    "#DD8452",
    },
    "MedGemma\n(layer 26)": {
        "lr_file":  "medgemma/results/offline_correction_layer26/results.json",
        "vc_file":  "medgemma/results/vector_correction_layer26_5k10ep/results.json",
        "color":    "#55A868",
    },
}

METRICS = {
    "overall":      "Overall Accuracy",
    "gt_yes":       "GT=Yes Accuracy",
    "gt_no":        "GT=No Accuracy",
    "adversarial":  "Adversarial Accuracy",
    "adv_paired":   "Adversarial Paired (strict)",
    "adv_wo_pair":  "Adversarial Paired (GT only)",
}


def load(path):
    with open(os.path.join(BASE, path)) as f:
        return json.load(f)


def get_metric(data, method, metric):
    results = data.get("results", data)
    return results.get(method, {}).get(metric, float("nan"))


# ── collect data ─────────────────────────────────────────────────────────────
records = {}
for model_label, cfg in MODELS.items():
    lr_data = load(cfg["lr_file"])
    vc_data = load(cfg["vc_file"])
    records[model_label] = {
        "lr":  lr_data,
        "vc":  vc_data,
        "color": cfg["color"],
    }

# ── Figure 1: grouped bar chart per metric ───────────────────────────────────
n_models  = len(MODELS)
n_metrics = len(METRICS)
model_labels = list(records.keys())

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

bar_width = 0.25
x = np.arange(n_models)

for ax_idx, (metric_key, metric_label) in enumerate(METRICS.items()):
    ax = axes[ax_idx]

    lr_vals  = [get_metric(records[m]["lr"], "logistic_regression", metric_key) for m in model_labels]
    vc_vals  = [get_metric(records[m]["vc"], "vector_correction",   metric_key) for m in model_labels]
    raw_vals = [get_metric(records[m]["lr"], "raw_model",           metric_key) for m in model_labels]

    colors = [records[m]["color"] for m in model_labels]

    b_raw = ax.bar(x - bar_width, raw_vals, bar_width, label="Raw model",
                   color=colors, alpha=0.35, edgecolor="white")
    b_lr  = ax.bar(x,             lr_vals,  bar_width, label="LR (last layer, offline)",
                   color=colors, alpha=0.75, edgecolor="white")
    b_vc  = ax.bar(x + bar_width, vc_vals,  bar_width, label="Vector correction (intermediate, online)",
                   color=colors, alpha=1.0,  edgecolor="white", hatch="//")

    for bars in [b_raw, b_lr, b_vc]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_title(metric_label, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

# shared legend
legend_handles = [
    mpatches.Patch(facecolor="gray", alpha=0.35, label="Raw model"),
    mpatches.Patch(facecolor="gray", alpha=0.75, label="LR (last layer, offline)"),
    mpatches.Patch(facecolor="gray", alpha=1.0,  hatch="//", label="Vector correction (intermediate, online)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, 0.01), frameon=True)

fig.suptitle("LR (last layer) vs Vector Correction (intermediate layer)\nper model and metric",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout(rect=[0, 0.06, 1, 1])
out1 = os.path.join(BASE, "comparison_lr_vs_vc.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out1}")


# ── Figure 2: per-model radar / line plot across metrics ─────────────────────
metric_keys   = list(METRICS.keys())
metric_labels = list(METRICS.values())

fig, axes = plt.subplots(1, n_models, figsize=(16, 5), sharey=True)
x = np.arange(len(metric_keys))

for ax, model_label in zip(axes, model_labels):
    color = records[model_label]["color"]
    lr_vals  = [get_metric(records[model_label]["lr"], "logistic_regression", m) for m in metric_keys]
    vc_vals  = [get_metric(records[model_label]["vc"], "vector_correction",   m) for m in metric_keys]
    raw_vals = [get_metric(records[model_label]["lr"], "raw_model",           m) for m in metric_keys]

    ax.plot(x, raw_vals, "o--", color=color, alpha=0.4,  linewidth=1.5, label="Raw model")
    ax.plot(x, lr_vals,  "s-",  color=color, alpha=0.8,  linewidth=2,   label="LR (last layer)")
    ax.plot(x, vc_vals,  "D",   color=color, alpha=1.0,  linewidth=2,   label="Vector correction", linestyle="-.")

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n") for l in metric_labels], fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(model_label, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

axes[0].set_ylabel("Accuracy", fontsize=10)
axes[0].legend(fontsize=8, loc="lower left")

fig.suptitle("Metric profiles: Raw / LR / Vector Correction per model",
             fontsize=12, fontweight="bold")
fig.tight_layout()
out2 = os.path.join(BASE, "comparison_profiles.png")
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out2}")


# ── Figure 3: delta (vc - lr) heatmap ────────────────────────────────────────
delta = np.full((n_models, n_metrics), np.nan)
for i, model_label in enumerate(model_labels):
    for j, metric_key in enumerate(metric_keys):
        lr_v = get_metric(records[model_label]["lr"], "logistic_regression", metric_key)
        vc_v = get_metric(records[model_label]["vc"], "vector_correction",   metric_key)
        delta[i, j] = vc_v - lr_v

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(delta, cmap="RdYlGn", vmin=-0.15, vmax=0.15, aspect="auto")
ax.set_xticks(range(n_metrics))
ax.set_xticklabels(metric_labels, rotation=20, ha="right", fontsize=9)
ax.set_yticks(range(n_models))
ax.set_yticklabels([m.replace("\n", " ") for m in model_labels], fontsize=9)
for i in range(n_models):
    for j in range(n_metrics):
        v = delta[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(v) < 0.08 else "white")
plt.colorbar(im, ax=ax, label="Vector correction − LR (positive = VC wins)")
ax.set_title("Delta: Vector Correction (intermediate) − LR (last layer)\ngreen = VC better, red = LR better",
             fontsize=11, fontweight="bold")
fig.tight_layout()
out3 = os.path.join(BASE, "comparison_delta_heatmap.png")
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out3}")
