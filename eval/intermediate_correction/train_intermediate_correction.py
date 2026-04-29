"""
Train correction models on intermediate-layer hidden states.

Outputs:
  results.json / results.csv   — accuracy metrics for each method
  trained_lr_direction.npy     — normalized LR direction for forwarded eval
  test_image_ids.json          — image IDs held out for the forwarded eval
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Models (same as train_lm_correction.py)
# ---------------------------------------------------------------------------

class BiasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.b


class CorrectionModel(nn.Module):
    def __init__(self, hidden_size, d):
        super().__init__()
        self.register_buffer("d", torch.tensor(d, dtype=torch.float32))
        self.w = nn.Parameter(0.01 * torch.randn(hidden_size))
        self.v = nn.Parameter(0.01 * torch.randn(hidden_size))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, h):
        base = (h * self.d).sum(-1)
        correction = self.alpha * (h @ self.w) * (self.d @ self.v)
        return base + correction


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_model(model, X_train, y_train, X_val, y_val, lr, epochs=300):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        criterion(model(Xt).squeeze(), yt).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        val_preds = (model(Xv).squeeze().numpy() > 0).astype(int)
    return float(accuracy_score(y_val, val_preds))


def sweep_lr(ModelClass, model_kwargs, X_tr, y_tr, X_val, y_val, lrs, epochs=300):
    best_lr, best_acc = lrs[0], 0.0
    for lr in lrs:
        model = ModelClass(**model_kwargs)
        val_acc = train_model(model, X_tr, y_tr, X_val, y_val, lr=lr, epochs=epochs)
        print(f"  lr={lr:.0e}  val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_lr = lr
    print(f"  -> best lr={best_lr:.0e}")
    return best_lr


def get_predictions(ModelClass, model_kwargs, X_train, y_train, X_test, lr, epochs=300):
    model = ModelClass(**model_kwargs)
    train_model(model, X_train, y_train, X_train, y_train, lr=lr, epochs=epochs)
    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
    return (scores > 0).astype(int)


# ---------------------------------------------------------------------------
# Metrics (same as train_lm_correction.py)
# ---------------------------------------------------------------------------

def compute_paired_adversarial_acc(y_true, y_pred, qa_types, image_ids):
    correct = {}
    img_qts = defaultdict(list)
    for img_id, qt, yt, yp in zip(image_ids, qa_types, y_true, y_pred):
        correct[(img_id, qt)] = int(yp == yt)
        img_qts[img_id].append(qt)

    pairs = []
    for img_id, qts in img_qts.items():
        qt_set = set(qts)
        if "modality_gt" in qt_set and "modality_hallu" in qt_set:
            pairs.append((correct[(img_id, "modality_gt")], correct[(img_id, "modality_hallu")]))
        if "body_part_gt" in qt_set and "body_part_hallu" in qt_set:
            pairs.append((correct[(img_id, "body_part_gt")], correct[(img_id, "body_part_hallu")]))
        for qt in qts:
            if qt.startswith("entity_gt_"):
                hallu = "entity_hallu_" + qt[len("entity_gt_"):]
                if hallu in qt_set:
                    pairs.append((correct[(img_id, qt)], correct[(img_id, hallu)]))
            if qt.startswith("grounding_gt_"):
                hallu = "grounding_hallu_" + qt[len("grounding_gt_"):]
                if hallu in qt_set:
                    pairs.append((correct[(img_id, qt)], correct[(img_id, hallu)]))

    if not pairs:
        return float("nan"), float("nan"), 0
    acc_with_pair = sum(1 for g, h in pairs if g == 1 and h == 1) / len(pairs)
    acc_without_pair = sum(1 for g, h in pairs if g == 1) / len(pairs)
    return acc_with_pair, acc_without_pair, len(pairs)


def compute_metrics(y_true, y_pred, qa_types, image_ids):
    adv_mask = np.array(["hallu" in qt for qt in qa_types])
    def safe_acc(mask):
        return float(accuracy_score(y_true[mask], y_pred[mask])) if mask.sum() > 0 else float("nan")
    acc_with_pair, acc_without_pair, n_pairs = compute_paired_adversarial_acc(
        y_true, y_pred, qa_types, image_ids)
    return {
        "overall": float(accuracy_score(y_true, y_pred)),
        "gt_yes": safe_acc(y_true == 1),
        "gt_no": safe_acc(y_true == 0),
        "adversarial": safe_acc(adv_mask),
        "adv_paired": acc_with_pair,
        "adv_wo_pair": acc_without_pair,
        "n_total": int(len(y_true)),
        "n_gt_yes": int((y_true == 1).sum()),
        "n_gt_no": int((y_true == 0).sum()),
        "n_adversarial": int(adv_mask.sum()),
        "n_pairs": n_pairs,
    }


def print_results_table(all_results):
    metrics = ["overall", "gt_yes", "gt_no", "adversarial", "adv_paired", "adv_wo_pair"]
    header = f"{'Method':<24}" + "".join(f"{m:>14}" for m in metrics)
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for method, res in all_results.items():
        row = f"{method:<24}" + "".join(f"{res.get(m, float('nan')):>14.4f}" for m in metrics)
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-file", required=True,
                        help="Path to hidden_states_cache.npz (intermediate layer)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache = np.load(args.cache_file)
    H = cache["hidden_states"].astype(np.float32)
    yes_logits = cache["yes_logits"]
    no_logits = cache["no_logits"]
    gt_labels = cache["gt_labels"].astype(np.int32)
    image_ids = cache["image_ids"]
    w_yes = cache["w_yes"].astype(np.float32)
    w_no = cache["w_no"].astype(np.float32)
    target_layer = int(cache["target_layer"]) if "target_layer" in cache else -1

    cache_dir = os.path.dirname(args.cache_file)
    with open(os.path.join(cache_dir, "metadata.json")) as f:
        metadata = json.load(f)

    qa_types = np.array([m["qa_type"] for m in metadata])
    logit_diffs = yes_logits - no_logits
    d = w_yes - w_no

    print(f"Loaded {len(H)} questions from layer {target_layer}.  "
          f"GT yes: {gt_labels.sum()},  no: {(gt_labels==0).sum()}")

    # Image-ID-based train/test split (same strategy as train_lm_correction.py)
    unique_ids = np.unique(image_ids)
    train_imgs, test_imgs = train_test_split(unique_ids, test_size=0.2, random_state=args.seed)
    train_mask = np.isin(image_ids, train_imgs)
    test_mask = np.isin(image_ids, test_imgs)

    H_train, H_test = H[train_mask], H[test_mask]
    y_train, y_test = gt_labels[train_mask], gt_labels[test_mask]
    ld_train, ld_test = logit_diffs[train_mask], logit_diffs[test_mask]
    qa_test = qa_types[test_mask]
    ids_test = image_ids[test_mask]

    print(f"Train: {train_mask.sum()}  Test: {test_mask.sum()}")

    # Inner val split for LR sweep
    inner_imgs, val_imgs = train_test_split(train_imgs, test_size=0.1, random_state=args.seed)
    inner_mask = np.isin(image_ids[train_mask], inner_imgs)
    val_mask = ~inner_mask
    H_inner, H_val = H_train[inner_mask], H_train[val_mask]
    y_inner, y_val = y_train[inner_mask], y_train[val_mask]
    ld_inner, ld_val = ld_train[inner_mask], ld_train[val_mask]

    LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    EPOCHS = 300
    hidden_size = H.shape[1]
    all_results = {}

    # Baseline: raw model logit diff (final-layer output, same for all methods)
    raw_preds = (ld_test > 0).astype(int)
    all_results["raw_model"] = compute_metrics(y_test, raw_preds, qa_test, ids_test)

    # Bias correction on logit diff
    print("\nBias LR sweep:")
    best_lr_bias = sweep_lr(BiasModel, {}, ld_inner.reshape(-1, 1), y_inner,
                            ld_val.reshape(-1, 1), y_val, LRS, EPOCHS)
    bias_preds = get_predictions(BiasModel, {}, ld_train.reshape(-1, 1), y_train,
                                 ld_test.reshape(-1, 1), best_lr_bias, EPOCHS)
    all_results["bias"] = compute_metrics(y_test, bias_preds, qa_test, ids_test)

    # Logistic regression on intermediate hidden states
    scaler = StandardScaler()
    H_train_scaled = scaler.fit_transform(H_train)
    H_test_scaled = scaler.transform(H_test)
    clf = LogisticRegression(max_iter=1000, random_state=args.seed, solver="saga", n_jobs=-1)
    clf.fit(H_train_scaled, y_train)
    all_results["logistic_regression"] = compute_metrics(
        y_test, clf.predict(H_test_scaled), qa_test, ids_test)

    # CorrectionModel on intermediate hidden states
    print("\nCorrection model LR sweep:")
    corr_kwargs = {"hidden_size": hidden_size, "d": d}
    best_lr_corr = sweep_lr(CorrectionModel, corr_kwargs, H_inner, y_inner, H_val, y_val, LRS, EPOCHS)
    corr_preds = get_predictions(CorrectionModel, corr_kwargs, H_train, y_train,
                                 H_test, best_lr_corr, EPOCHS)
    all_results["correction"] = compute_metrics(y_test, corr_preds, qa_test, ids_test)

    print_results_table(all_results)

    # Save results
    out_file = os.path.join(args.output_dir, "results.json")
    with open(out_file, "w") as f:
        json.dump({
            "target_layer": target_layer,
            "best_lr_bias": best_lr_bias,
            "best_lr_correction": best_lr_corr,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved: {out_file}")

    metric_cols = ["overall", "gt_yes", "gt_no", "adversarial", "adv_paired", "adv_wo_pair"]
    csv_file = os.path.join(args.output_dir, "results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + metric_cols)
        writer.writeheader()
        for method, res in all_results.items():
            writer.writerow({"method": method,
                             **{m: f"{res.get(m, float('nan')):.4f}" for m in metric_cols}})
    print(f"Saved: {csv_file}")

    # Save LR steering direction in original (unscaled) hidden-state space
    lr_direction = (clf.coef_[0] / scaler.scale_).astype(np.float32)
    lr_direction /= np.linalg.norm(lr_direction) + 1e-12
    dir_file = os.path.join(args.output_dir, "trained_lr_direction.npy")
    np.save(dir_file, lr_direction)
    print(f"Saved: {dir_file}  shape={lr_direction.shape}")

    # Save test image IDs so the forwarded eval uses the same held-out split
    test_ids_file = os.path.join(args.output_dir, "test_image_ids.json")
    with open(test_ids_file, "w") as f:
        json.dump([int(i) for i in test_imgs], f)
    print(f"Saved: {test_ids_file}")

    # Plots
    methods = list(all_results.keys())
    plot_metrics = {
        "overall": "Overall Accuracy",
        "gt_yes": "GT=Yes Accuracy",
        "gt_no": "GT=No Accuracy",
        "adversarial": "Adversarial (hallu-only)",
        "adv_paired": "Adversarial (paired, strict)",
        "adv_wo_pair": "Adversarial (paired, w.o. pair)",
    }
    ncols = 3
    nrows = (len(plot_metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    x = np.arange(len(methods))
    for i, (key, label) in enumerate(plot_metrics.items()):
        ax = axes[i]
        vals = [all_results[m].get(key, float("nan")) for m in methods]
        bars = ax.bar(x, vals, 0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.1)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Intermediate Correction (layer {target_layer})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "results_subplots.png"), dpi=150)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
