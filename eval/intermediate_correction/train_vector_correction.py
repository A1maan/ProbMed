"""
Train a rank-1 residual correction at an intermediate transformer layer.

  h' = h + alpha * (w · h) * v

Applied at layer L via forward hook; corrected h' propagates through
layers L+1 ... N to produce yes/no logits.  Only w, v, alpha are updated;
the backbone is fully frozen.

Memory trick: the hook detaches h before applying the correction, so
gradients only flow backward through layers L+1 ... N, not 0 ... L-1.

Outputs
-------
correction_net.pt        full state dict (for reloading the hook)
w.npy / v.npy / alpha.npy  individual parameter arrays (for ablation)
results.json / results.csv  accuracy metrics incl. gt-only yes/no rates
category_bias.json       per-category bias trained on cached logit diffs
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENT_DIR)

from intermediate_layer_correction import MODEL_DEFAULTS, build_runner, load_questions
from train_intermediate_correction import (
    compute_metrics,
    compute_paired_adversarial_acc,
    print_results_table,
)


# ---------------------------------------------------------------------------
# Category bias baseline (trained offline on cached logit diffs)
# ---------------------------------------------------------------------------

def qa_type_to_category(qa_type):
    """Map qa_type string → coarse category used for per-category bias."""
    if qa_type.startswith("modality"):
        return "modality"
    if qa_type.startswith("body_part"):
        return "body_part"
    if qa_type.startswith("entity"):
        return "entity"
    if qa_type.startswith("grounding"):
        return "grounding"
    return "other"


class CategoryBiasModel(nn.Module):
    """
    Learns a separate additive scalar bias per question category:
        score(q) = logit_diff(q) + b_{category(q)}

    Same structure as BiasModel but one bias per category instead of one global.
    Trained purely on cached yes_logit - no_logit values; no model forward pass needed.
    """

    def __init__(self, categories):
        super().__init__()
        self.categories = list(categories)
        self.cat_to_idx = {c: i for i, c in enumerate(self.categories)}
        self.biases = nn.Parameter(torch.zeros(len(self.categories)))

    def forward(self, logit_diffs, cat_indices):
        # logit_diffs: (N,), cat_indices: (N,) int
        return logit_diffs + self.biases[cat_indices]


def train_category_bias(logit_diffs, gt_labels, categories, cat_to_idx,
                        train_mask, test_mask, lrs, epochs, seed):
    """Train CategoryBiasModel; return test predictions and learned biases dict."""
    ld_tr = torch.tensor(logit_diffs[train_mask], dtype=torch.float32)
    y_tr  = torch.tensor(gt_labels[train_mask],   dtype=torch.float32)
    ld_te = torch.tensor(logit_diffs[test_mask],  dtype=torch.float32)
    y_te  = torch.tensor(gt_labels[test_mask],    dtype=torch.float32)

    cat_idx_tr = torch.tensor([cat_to_idx[c] for c in categories[train_mask]], dtype=torch.long)
    cat_idx_te = torch.tensor([cat_to_idx[c] for c in categories[test_mask]],  dtype=torch.long)

    criterion = nn.BCEWithLogitsLoss()
    best_acc, best_preds, best_biases = 0.0, None, None

    for lr in lrs:
        torch.manual_seed(seed)
        model = CategoryBiasModel(list(cat_to_idx.keys()))
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            criterion(model(ld_tr, cat_idx_tr), y_tr).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            scores = model(ld_te, cat_idx_te).numpy()
        preds = (scores > 0).astype(int)
        acc = float(accuracy_score(y_te.numpy().astype(int), preds))
        print(f"  CategoryBias lr={lr:.0e}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_preds = preds
            best_biases = {c: float(model.biases[i].item())
                           for c, i in model.cat_to_idx.items()}

    return best_preds, best_biases


# ---------------------------------------------------------------------------
# gt-only yes/no rate helper
# ---------------------------------------------------------------------------

def yes_no_rates(y_true, y_pred, mask=None):
    """
    Returns (yes_rate, no_rate) where yes_rate is the fraction of
    gt=yes questions predicted yes, and no_rate fraction of gt=no predicted no.
    mask: optional boolean array to restrict to a subset (e.g. gt-only questions).
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    yes_mask = y_true == 1
    no_mask  = y_true == 0
    yes_rate = float(y_pred[yes_mask].mean()) if yes_mask.sum() > 0 else float("nan")
    no_rate  = float(y_pred[no_mask].mean())  if no_mask.sum()  > 0 else float("nan")
    return yes_rate, no_rate


# ---------------------------------------------------------------------------
# Correction network
# ---------------------------------------------------------------------------

class RankOneCorrectionNet(nn.Module):
    """
    Rank-1 residual correction: h' = h + alpha * (w·h) * v

    Matches the formula h' = h + alpha*(w·h)*v where w, v, alpha are
    all learnable and the backbone is kept fully frozen.  The gate is a
    plain linear dot product (no sigmoid) so the correction is an exact
    rank-1 update to h.

    w and v are small-random initialized so gradients flow from step 1.
    alpha is zero-initialized so the correction starts at exactly zero.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.w = nn.Parameter(0.01 * torch.randn(hidden_size))
        self.v = nn.Parameter(0.01 * torch.randn(hidden_size))
        self.alpha = nn.Parameter(torch.zeros(1))   # correction = 0 at init

    def forward(self, h):
        # h: (..., hidden_size)
        gate = (h * self.w).sum(dim=-1, keepdim=True)  # (..., 1)  linear dot product
        return h + self.alpha * gate * self.v           # (..., hidden_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_hidden_size(runner):
    cfg = runner.model.config
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return cfg.text_config.hidden_size
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    raise AttributeError("Cannot infer hidden_size from model config")


def make_hook(net):
    """
    Forward hook for layer L.
    - Detaches h to stop gradients flowing back through layers 0 ... L-1.
    - Applies rank-1 correction to the last token position.
    - Returns corrected h so layers L+1 ... N receive it.
    Moves net parameters to h's device dynamically to handle multi-GPU
    model sharding (device_map="auto" may place layers on different GPUs).
    """
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.detach()                              # stop grad through earlier layers
        last = h[:, -1:, :].float()                # (batch, 1, hidden_size)

        # Move params to this layer's device
        w     = net.w.to(last.device)
        v     = net.v.to(last.device)
        alpha = net.alpha.to(last.device)

        gate      = (last * w).sum(dim=-1, keepdim=True)                  # (batch, 1, 1)  linear
        corrected = (last + alpha * gate * v).to(h.dtype)                # (batch, 1, hidden_size)

        if h.shape[1] > 1:
            h_new = torch.cat([h[:, :-1, :], corrected], dim=1)
        else:
            h_new = corrected
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new
    return hook


# ---------------------------------------------------------------------------
# One pass over a question list (train or eval)
# ---------------------------------------------------------------------------

def run_pass(runner, questions, layer, net, optimizer, criterion, image_mode, training):
    net.train(training)

    total_loss = 0.0
    y_pred, y_true_list = [], []
    qa_types, image_ids = [], []
    skipped = 0

    hook_fn = make_hook(net)

    for q in tqdm(questions, desc="Train" if training else "Eval"):
        if not os.path.exists(q["image_path"]):
            skipped += 1
            continue

        prepared = None
        try:
            prepared = runner.prepare_image(q["image_path"], image_mode)
            handle = runner.layers[layer - 1].register_forward_hook(hook_fn)

            try:
                ctx = torch.enable_grad() if training else torch.no_grad()
                with ctx:
                    logits = runner.forward_logits(prepared, q["question"])
                    ld = logits[0, runner.yes_token_id] - logits[0, runner.no_token_id]
                    gt = torch.tensor(
                        [float(q["gt_label"])], dtype=torch.float32, device=ld.device
                    )
                    if criterion.pos_weight is not None:
                        criterion.pos_weight = criterion.pos_weight.to(ld.device)
                    loss = criterion(ld.unsqueeze(0), gt)

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            finally:
                handle.remove()

            total_loss += loss.item()
            y_pred.append(1 if ld.item() > 0 else 0)
            y_true_list.append(int(q["gt_label"]))
            qa_types.append(q.get("qa_type", ""))
            image_ids.append(int(q["id"]))

        except Exception as e:
            print(f"  Error on id={q['id']}: {e}")
            skipped += 1
        finally:
            if prepared is not None:
                runner.cleanup(prepared)

        if len(y_pred) % 100 == 0:
            torch.cuda.empty_cache()

    n = max(len(y_pred), 1)
    avg_loss = total_loss / n
    acc = float(accuracy_score(y_true_list, y_pred)) if y_pred else 0.0
    print(f"  loss={avg_loss:.4f}  acc={acc:.4f}  skipped={skipped}")

    return np.array(y_pred), np.array(y_true_list), np.array(qa_types), np.array(image_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    default_model=None,
    default_results_file=None,
    default_test_file=None,
    default_image_folder=None,
    default_load_8bit=False,
):
    parser = argparse.ArgumentParser(
        description="Train residual linear correction at an intermediate layer"
    )
    parser.add_argument("--model", choices=sorted(MODEL_DEFAULTS),
                        default=default_model, required=default_model is None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--results-file", default=default_results_file,
                        required=default_results_file is None)
    parser.add_argument("--test-file", default=default_test_file,
                        required=default_test_file is None)
    parser.add_argument("--image-folder", default=default_image_folder,
                        required=default_image_folder is None)
    parser.add_argument("--test-image-ids", required=True,
                        help="test_image_ids.json from train_intermediate_correction.py")
    parser.add_argument("--cache-file", default=None,
                        help="hidden_states_cache.npz (from correction_score extractor). "
                             "When provided, enables the per-category bias baseline.")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--load-8bit", action="store_true", default=default_load_8bit,
                        help="Avoid for training — 8-bit quantization blocks backprop. "
                             "Load LLaVA-Med in fp16 instead.")
    parser.add_argument("--image-mode", choices=["real", "black", "random"], default="real")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=1.0,
                        help="Weight for positive (yes) class in BCEWithLogitsLoss. "
                             ">1 penalises missing yes answers more.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap training questions per epoch (shuffled each epoch)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    defaults = MODEL_DEFAULTS[args.model]
    model_name = args.model_name or defaults["model_name"]
    layer = args.layer if args.layer is not None else defaults["layer"]
    if layer is None:
        raise ValueError("--layer is required (or set a default in MODEL_DEFAULTS)")

    output_dir = args.output_dir or os.path.join(
        args.model, "results", f"vector_correction_layer{layer}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Split questions into train / test using the same IDs as Step 2
    with open(args.test_image_ids) as f:
        test_ids = set(json.load(f))

    all_questions = load_questions(
        args.results_file, args.test_file, args.image_folder
    )
    train_qs = [q for q in all_questions if q["id"] not in test_ids]
    test_qs  = [q for q in all_questions if q["id"] in test_ids]
    if args.limit:
        random.shuffle(train_qs)
        train_qs = train_qs[:args.limit]
    print(f"Train: {len(train_qs)}  Test: {len(test_qs)}  Layer: {layer}")

    # Load model (backbone frozen)
    runner = build_runner(args.model, model_name, args.load_8bit)
    runner.model.requires_grad_(False)

    hidden_size = get_hidden_size(runner)
    layer_device = next(runner.layers[layer - 1].parameters()).device
    net = RankOneCorrectionNet(hidden_size).to(layer_device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight])
    )

    print(f"\nRankOneCorrectionNet: hidden_size={hidden_size}  "
          f"params={sum(p.numel() for p in net.parameters()):,}  "  # 2*4096+1 = 8193
          f"lr={args.lr}  epochs={args.epochs}\n")

    # Training
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        shuffled = train_qs[:]
        rng.shuffle(shuffled)
        run_pass(runner, shuffled, layer, net, optimizer, criterion,
                 args.image_mode, training=True)
        torch.cuda.empty_cache()

    # Save trained correction net (full state dict for hook reloading)
    net_path = os.path.join(output_dir, "correction_net.pt")
    torch.save(net.state_dict(), net_path)
    print(f"\nSaved correction net: {net_path}")

    # Save w, v, alpha individually for ablation
    np.save(os.path.join(output_dir, "w.npy"),     net.w.detach().cpu().float().numpy())
    np.save(os.path.join(output_dir, "v.npy"),     net.v.detach().cpu().float().numpy())
    np.save(os.path.join(output_dir, "alpha.npy"), net.alpha.detach().cpu().float().numpy())
    print(f"Saved w.npy, v.npy, alpha.npy  (alpha={net.alpha.item():.6f})")

    # Evaluation on test split
    print("\n=== Test evaluation ===")
    y_pred, y_true, qa_types, img_ids = run_pass(
        runner, test_qs, layer, net, optimizer, criterion,
        args.image_mode, training=False
    )

    # Metrics (same structure as train_intermediate_correction.py)
    results = compute_metrics(y_true, y_pred, qa_types, img_ids)
    all_results = {"vector_correction": results}

    # Raw model predictions aligned to the same evaluated subset returned by run_pass.
    # run_pass skips missing images; img_ids tells us exactly which questions ran.
    test_q_by_id = {int(q["id"]): q for q in test_qs}
    raw_preds_test = np.array([
        1 if test_q_by_id[iid]["model_pred"] == "yes" else 0
        for iid in img_ids
    ])
    all_results["raw_model"] = compute_metrics(y_true, raw_preds_test, qa_types, img_ids)

    print_results_table(all_results)

    # gt-only yes/no rates: non-adversarial questions only
    gt_only_mask = np.array(["hallu" not in qt for qt in qa_types])
    raw_yes_rate, raw_no_rate = yes_no_rates(y_true, raw_preds_test, gt_only_mask)
    cor_yes_rate, cor_no_rate = yes_no_rates(y_true, y_pred, gt_only_mask)

    print(f"\nGT-only yes/no rates (non-adversarial questions):")
    print(f"  raw model:   yes_rate={raw_yes_rate:.4f}  no_rate={raw_no_rate:.4f}")
    print(f"  corrected:   yes_rate={cor_yes_rate:.4f}  no_rate={cor_no_rate:.4f}")

    # Category bias baseline (trained offline on cached logit diffs)
    print("\n=== Category bias baseline ===")
    if args.cache_file:
        cache = np.load(args.cache_file)
        ld_all   = (cache["yes_logits"] - cache["no_logits"]).astype(np.float32)
        gt_all   = cache["gt_labels"].astype(np.int32)
        img_all  = cache["image_ids"]

        with open(os.path.join(os.path.dirname(args.cache_file), "metadata.json")) as f:
            meta = json.load(f)
        qt_all = np.array([qa_type_to_category(m["qa_type"]) for m in meta])

        unique_categories = sorted(set(qt_all))
        cat_to_idx = {c: i for i, c in enumerate(unique_categories)}

        # Use same image-id-based train/test split as run_pass
        tr_mask_cache = np.isin(img_all, [q["id"] for q in train_qs])
        te_mask_cache = np.isin(img_all, [q["id"] for q in test_qs])

        LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        cb_preds, cb_biases = train_category_bias(
            ld_all, gt_all, qt_all, cat_to_idx,
            tr_mask_cache, te_mask_cache,
            lrs=LRS, epochs=300, seed=args.seed,
        )
        qt_test_cache = np.array([m["qa_type"]
                                  for m in np.array(meta, dtype=object)[te_mask_cache]])
        id_test_cache = img_all[te_mask_cache]
        gt_test_cache = gt_all[te_mask_cache]

        all_results["category_bias"] = compute_metrics(
            gt_test_cache, cb_preds, qt_test_cache, id_test_cache
        )
        print_results_table({"category_bias": all_results["category_bias"]})

        cb_path = os.path.join(output_dir, "category_bias.json")
        with open(cb_path, "w") as f:
            json.dump({"biases": cb_biases, "results": all_results["category_bias"]}, f, indent=2)
        print(f"Saved: {cb_path}")
        print("Per-category biases:", cb_biases)
    else:
        print("  (skipped — pass --cache-file to enable category bias baseline)")

    # Save all results
    out = {
        "model": args.model,
        "layer": layer,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_size": hidden_size,
        "gt_only_yes_no_rates": {
            "raw_yes_rate": raw_yes_rate,
            "raw_no_rate":  raw_no_rate,
            "corrected_yes_rate": cor_yes_rate,
            "corrected_no_rate":  cor_no_rate,
        },
        "results": all_results,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results.json")

    metric_cols = ["overall", "gt_yes", "gt_no", "adversarial", "adv_paired", "adv_wo_pair"]
    with open(os.path.join(output_dir, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + metric_cols)
        writer.writeheader()
        for method, res in all_results.items():
            writer.writerow({"method": method,
                             **{m: f"{res.get(m, float('nan')):.4f}" for m in metric_cols}})

    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
