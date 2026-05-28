"""
Experiment 2: Per-category breakdown across all 3 models.

For each model, loads cached hidden states and applies:
  - raw_model        : model_pred from metadata (no hidden states needed)
  - logistic_regression : re-fit LR on train hidden states, predict on test
  - vector_correction   : apply saved w/v/alpha to test hidden states

Breaks results down by the 5 question categories:
  modality, body_part, abnormality, entity, grounding

Outputs:
  results/{model}/category_breakdown.json
  results/{model}/category_breakdown.csv
  results/{model}/category_breakdown.png
  results/category_breakdown_all_models.csv   (combined)
  results/category_breakdown_all_models.png   (combined heatmap)

Usage:
    python category_breakdown.py
    python category_breakdown.py --models llavamed chexagent
"""

import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXPERIMENT_DIR)

from train_intermediate_correction import compute_metrics

# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------

CATEGORIES = ["modality", "body_part", "abnormality", "entity", "grounding"]

def collapse_qa_type(qa_type):
    for cat in CATEGORIES:
        if qa_type.startswith(cat):
            return cat
    return "other"


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

MODEL_CONFIG = {
    "llavamed": {
        "layer": 15,
        "vector_correction_dir": "vector_correction_layer15_full1ep",
    },
    "chexagent": {
        "layer": 21,
        "vector_correction_dir": "vector_correction_layer21_5k10ep",
    },
    "medgemma": {
        "layer": 26,
        "vector_correction_dir": "vector_correction_layer26_5k10ep",
    },
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_model_data(model, base_dir):
    cfg   = MODEL_CONFIG[model]
    layer = cfg["layer"]
    hs_dir  = os.path.join(base_dir, model, "results", f"hidden_states_layer{layer}")
    off_dir = os.path.join(base_dir, model, "results", f"offline_correction_layer{layer}")
    vc_dir  = os.path.join(base_dir, model, "results", cfg["vector_correction_dir"])

    print(f"[{model}] Loading metadata...")
    with open(os.path.join(hs_dir, "metadata.json")) as f:
        metadata = json.load(f)

    print(f"[{model}] Loading hidden states ({layer=})...")
    cache = np.load(os.path.join(hs_dir, "hidden_states_cache.npz"))
    H = cache["hidden_states"].astype(np.float32)   # (N, hidden_dim)
    ids_cache = cache["image_ids"].astype(int)       # (N,)

    print(f"[{model}] Hidden states shape: {H.shape}")

    gt_labels = cache["gt_labels"].astype(np.int32)
    image_ids = cache["image_ids"].astype(int)
    # Use logit comparison for raw predictions — more accurate than metadata string
    raw_preds = (cache["yes_logits"] > cache["no_logits"]).astype(np.int32)

    with open(os.path.join(off_dir, "test_image_ids.json")) as f:
        test_image_ids = set(json.load(f))

    assert len(metadata) == len(H), (
        f"metadata length {len(metadata)} != hidden states {len(H)}"
    )

    qa_types = np.array([m["qa_type"] for m in metadata])

    test_mask  = np.array([iid in test_image_ids for iid in image_ids])
    train_mask = ~test_mask

    print(f"[{model}] Train: {train_mask.sum()}  Test: {test_mask.sum()}")

    # Vector correction weights
    w     = np.load(os.path.join(vc_dir, "w.npy")).astype(np.float32)
    v     = np.load(os.path.join(vc_dir, "v.npy")).astype(np.float32)
    alpha = np.load(os.path.join(vc_dir, "alpha.npy")).item()

    return {
        "H":          H,
        "image_ids":  image_ids,
        "qa_types":   qa_types,
        "raw_preds":  raw_preds,
        "gt_labels":  gt_labels,
        "train_mask": train_mask,
        "test_mask":  test_mask,
        "w": w, "v": v, "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def lr_predictions(H_train, y_train, H_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    H_tr_sc = scaler.fit_transform(H_train)
    H_te_sc = scaler.transform(H_test)
    clf = LogisticRegression(max_iter=1000, random_state=42, solver="saga", n_jobs=-1)
    clf.fit(H_tr_sc, y_train)
    return clf.predict(H_te_sc)


def vector_correction_predictions(H_test, w, v, alpha):
    """Apply rank-1 correction: h' = h + alpha * (w·h) * v, then predict sign."""
    gate  = H_test @ w                         # (N,)
    H_corr = H_test + alpha * gate[:, None] * v  # (N, hidden_dim)
    # Prediction: sign of (h' · d_LR) — but we don't have d_LR here.
    # The vector correction was trained end-to-end with a yes/no logit objective,
    # so we need to re-derive a classifier on corrected hidden states.
    # Simplest: LR on corrected train hidden states → predict on corrected test.
    return H_corr


# ---------------------------------------------------------------------------
# Metrics per category
# ---------------------------------------------------------------------------

def compute_category_metrics(y_true, y_pred, qa_types, image_ids):
    """
    Returns dict: {category: metrics_dict}
    Also includes "overall" key for aggregate metrics.
    """
    results = {}
    collapsed = np.array([collapse_qa_type(qt) for qt in qa_types])

    # Overall
    results["overall"] = compute_metrics(y_true, y_pred, qa_types, image_ids)

    # Per category
    for cat in CATEGORIES:
        mask = collapsed == cat
        if mask.sum() == 0:
            continue
        results[cat] = compute_metrics(
            y_true[mask], y_pred[mask], qa_types[mask], image_ids[mask]
        )

    return results


# ---------------------------------------------------------------------------
# Main per-model computation
# ---------------------------------------------------------------------------

def run_model(model, base_dir, output_dir):
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    data = load_model_data(model, base_dir)

    H          = data["H"]
    image_ids  = data["image_ids"]
    qa_types   = data["qa_types"]
    raw_preds  = data["raw_preds"]
    gt_labels  = data["gt_labels"]
    train_mask = data["train_mask"]
    test_mask  = data["test_mask"]
    w, v, alpha = data["w"], data["v"], data["alpha"]

    H_train = H[train_mask]
    H_test  = H[test_mask]
    y_train = gt_labels[train_mask]
    y_test  = gt_labels[test_mask]
    qa_test = qa_types[test_mask]
    ids_test = image_ids[test_mask]
    raw_test = raw_preds[test_mask]

    # --- Method 1: raw model ---
    print(f"[{model}] Computing raw model metrics...")
    raw_cat = compute_category_metrics(y_test, raw_test, qa_test, ids_test)

    # --- Method 2: logistic regression ---
    print(f"[{model}] Fitting LR on {len(H_train)} train samples...")
    lr_preds = lr_predictions(H_train, y_train, H_test)
    lr_cat   = compute_category_metrics(y_test, lr_preds, qa_test, ids_test)

    # --- Method 3: vector correction ---
    # Apply rank-1 correction to hidden states, then fit LR on corrected train states
    print(f"[{model}] Applying vector correction (alpha={alpha:.4f})...")
    gate_train  = H_train @ w
    H_train_corr = H_train + alpha * gate_train[:, None] * v
    gate_test   = H_test @ w
    H_test_corr  = H_test  + alpha * gate_test[:, None]  * v

    print(f"[{model}] Fitting LR on corrected hidden states...")
    vc_preds = lr_predictions(H_train_corr, y_train, H_test_corr)
    vc_cat   = compute_category_metrics(y_test, vc_preds, qa_test, ids_test)

    all_methods = {
        "raw_model":          raw_cat,
        "logistic_regression": lr_cat,
        "vector_correction":  vc_cat,
    }

    return all_methods


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_model_outputs(model, all_methods, output_dir):
    model_out = os.path.join(output_dir, model)
    os.makedirs(model_out, exist_ok=True)

    # JSON
    with open(os.path.join(model_out, "category_breakdown.json"), "w") as f:
        json.dump(all_methods, f, indent=2)

    # CSV
    metric_cols = ["overall_acc", "adv_paired", "adversarial", "gt_yes", "gt_no"]
    rows = []
    for method, cat_results in all_methods.items():
        for cat, m in cat_results.items():
            rows.append({
                "model":      model,
                "method":     method,
                "category":   cat,
                "overall_acc": f"{m.get('overall', float('nan')):.4f}",
                "adv_paired":  f"{m.get('adv_paired', float('nan')):.4f}",
                "adversarial": f"{m.get('adversarial', float('nan')):.4f}",
                "gt_yes":      f"{m.get('gt_yes', float('nan')):.4f}",
                "gt_no":       f"{m.get('gt_no', float('nan')):.4f}",
            })

    csv_path = os.path.join(model_out, "category_breakdown.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "method", "category"] + metric_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{model}] Saved {csv_path}")

    # Plot: grouped bar per category, one subplot per metric
    plot_category_breakdown(model, all_methods, model_out)


def plot_category_breakdown(model, all_methods, output_dir):
    methods    = list(all_methods.keys())
    categories = ["overall"] + CATEGORIES
    metrics    = [("adv_paired", "Adv Paired"), ("overall", "Overall Acc"),
                  ("adversarial", "Adversarial"), ("gt_yes", "GT=Yes"), ("gt_no", "GT=No")]

    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()

    x      = np.arange(len(categories))
    width  = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax_i, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[ax_i]
        for m_i, method in enumerate(methods):
            vals = [
                all_methods[method].get(cat, {}).get(metric_key, float("nan"))
                for cat in categories
            ]
            offset = (m_i - len(methods) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=method, color=colors[m_i], alpha=0.85)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.15)
        ax.set_title(metric_label)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    for j in range(ax_i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Category Breakdown — {model}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "category_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[{model}] Saved {path}")


def save_combined_outputs(all_model_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Combined CSV
    metric_cols = ["overall_acc", "adv_paired", "adversarial", "gt_yes", "gt_no"]
    rows = []
    for model, all_methods in all_model_results.items():
        for method, cat_results in all_methods.items():
            for cat, m in cat_results.items():
                rows.append({
                    "model":       model,
                    "method":      method,
                    "category":    cat,
                    "overall_acc": f"{m.get('overall', float('nan')):.4f}",
                    "adv_paired":  f"{m.get('adv_paired', float('nan')):.4f}",
                    "adversarial": f"{m.get('adversarial', float('nan')):.4f}",
                    "gt_yes":      f"{m.get('gt_yes', float('nan')):.4f}",
                    "gt_no":       f"{m.get('gt_no', float('nan')):.4f}",
                })
    csv_path = os.path.join(output_dir, "category_breakdown_all_models.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "method", "category"] + metric_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    # Combined heatmap: adv_paired per (model x method) for each category
    plot_combined_heatmap(all_model_results, output_dir)


def plot_combined_heatmap(all_model_results, output_dir):
    models     = list(all_model_results.keys())
    methods    = ["raw_model", "logistic_regression", "vector_correction"]
    categories = ["overall"] + CATEGORIES
    metric_key = "adv_paired"

    ncols = len(categories)
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * len(models)))

    for col, cat in enumerate(categories):
        ax = axes[col]
        # Build matrix: rows=models, cols=methods
        matrix = np.full((len(models), len(methods)), float("nan"))
        for r, model in enumerate(models):
            for c, method in enumerate(methods):
                v = all_model_results[model].get(method, {}).get(cat, {}).get(metric_key, float("nan"))
                matrix[r, c] = v

        im = ax.imshow(matrix, vmin=0.3, vmax=0.8, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models if col == 0 else [], fontsize=9)
        ax.set_title(cat, fontsize=10, fontweight="bold")

        for r in range(len(models)):
            for c in range(len(methods)):
                val = matrix[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, color="black")

    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label=metric_key)
    fig.suptitle("Adv Paired Accuracy — Category Breakdown", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "category_breakdown_all_models.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 2: per-category breakdown")
    parser.add_argument("--models", nargs="+",
                        default=["llavamed", "chexagent", "medgemma"],
                        choices=["llavamed", "chexagent", "medgemma"])
    parser.add_argument("--base-dir",   default=EXPERIMENT_DIR)
    parser.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"
    ))
    args = parser.parse_args()

    all_model_results = {}
    for model in args.models:
        all_model_results[model] = run_model(model, args.base_dir, args.output_dir)
        save_model_outputs(model, all_model_results[model], args.output_dir)

    if len(args.models) > 1:
        save_combined_outputs(all_model_results, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
