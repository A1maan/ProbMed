import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class BiasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.b


class CorrectionModel(nn.Module):
    def __init__(self, hidden_size, d):
        super().__init__()
        self.register_buffer("d", torch.tensor(d, dtype=torch.float32)) # not learnable param
        self.w = nn.Parameter(torch.zeros(hidden_size))
        self.v = nn.Parameter(torch.zeros(hidden_size))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, h):
        base = (h * self.d).sum(-1)
        correction = self.alpha * (h @ self.w) * (self.d @ self.v)
        return base + correction


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
        criterion(model(Xt), yt).backward()
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


def compute_metrics(y_true, y_pred, qa_types):
    adv_mask = np.array(["hallu" in qt for qt in qa_types])

    def safe_acc(mask):
        return float(accuracy_score(y_true[mask], y_pred[mask])) if mask.sum() > 0 else float("nan")

    return {
        "overall": float(accuracy_score(y_true, y_pred)),
        "gt_yes": safe_acc(y_true == 1),
        "gt_no": safe_acc(y_true == 0),
        "adversarial": safe_acc(adv_mask),
        "n_total": int(len(y_true)),
        "n_gt_yes": int((y_true == 1).sum()),
        "n_gt_no": int((y_true == 0).sum()),
        "n_adversarial": int(adv_mask.sum()),
    }


def print_results_table(all_results):
    metrics = ["overall", "gt_yes", "gt_no", "adversarial"]
    header = f"{'Method':<24}" + "".join(f"{m:>14}" for m in metrics)
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for method, res in all_results.items():
        row = f"{method:<24}" + "".join(f"{res.get(m, float('nan')):>14.4f}" for m in metrics)
        print(row)
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache = np.load(os.path.join(args.cache_dir, "hidden_states_cache.npz"))
    H = cache["hidden_states"].astype(np.float32)
    yes_logits = cache["yes_logits"]
    no_logits = cache["no_logits"]
    gt_labels = cache["gt_labels"].astype(np.int32)
    image_ids = cache["image_ids"]
    w_yes = cache["w_yes"].astype(np.float32)
    w_no = cache["w_no"].astype(np.float32)

    with open(os.path.join(args.cache_dir, "metadata.json")) as f:
        metadata = json.load(f)

    qa_types = np.array([m["qa_type"] for m in metadata])
    logit_diffs = yes_logits - no_logits
    d = w_yes - w_no

    print(f"Loaded {len(H)} questions.  GT yes: {gt_labels.sum()},  no: {(gt_labels==0).sum()}")

    unique_ids = np.unique(image_ids)
    train_imgs, test_imgs = train_test_split(unique_ids, test_size=0.2, random_state=args.seed)
    train_mask = np.isin(image_ids, train_imgs)
    test_mask = np.isin(image_ids, test_imgs)

    H_train, H_test = H[train_mask], H[test_mask]
    y_train, y_test = gt_labels[train_mask], gt_labels[test_mask]
    ld_train, ld_test = logit_diffs[train_mask], logit_diffs[test_mask]
    qa_test = qa_types[test_mask]

    print(f"Train: {train_mask.sum()}  Test: {test_mask.sum()}")

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

    raw_preds = (ld_test > 0).astype(int)
    all_results["raw_model"] = compute_metrics(y_test, raw_preds, qa_test)

    print("\nBias LR sweep:")
    best_lr_bias = sweep_lr(BiasModel, {}, ld_inner.reshape(-1, 1), y_inner, ld_val.reshape(-1, 1), y_val, LRS, EPOCHS)
    bias_preds = get_predictions(BiasModel, {}, ld_train.reshape(-1, 1), y_train, ld_test.reshape(-1, 1), best_lr_bias, EPOCHS)
    all_results["bias"] = compute_metrics(y_test, bias_preds, qa_test)

    clf = LogisticRegression(max_iter=1000, random_state=args.seed, solver="lbfgs")
    clf.fit(H_train, y_train)
    all_results["logistic_regression"] = compute_metrics(y_test, clf.predict(H_test), qa_test)

    print("\nCorrection model LR sweep:")
    corr_kwargs = {"hidden_size": hidden_size, "d": d}
    best_lr_corr = sweep_lr(CorrectionModel, corr_kwargs, H_inner, y_inner, H_val, y_val, LRS, EPOCHS)
    corr_preds = get_predictions(CorrectionModel, corr_kwargs, H_train, y_train, H_test, best_lr_corr, EPOCHS)
    all_results["correction"] = compute_metrics(y_test, corr_preds, qa_test)

    print_results_table(all_results)

    out_file = os.path.join(args.output_dir, "results.json")
    with open(out_file, "w") as f:
        json.dump({"best_lr_bias": best_lr_bias, "best_lr_correction": best_lr_corr, "results": all_results}, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
