"""
Phase 2: Train per-head LR probes on extracted activations, select top-K heads,
and run ITI inference on the test set.

Reads head_activations.npz produced by extract_head_activations.py.
Saves:
    probe_accs.npy          (num_layers, num_heads) val accuracy per head
    top_heads.json          list of [layer, head] pairs selected
    directions.npz          (num_selected_layers, num_heads*head_dim) direction tensors
    results.json / .csv     test metrics

Usage:
    python train_iti_probes.py --model llavamed --num-heads 48 --alpha 15
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXPERIMENT_DIR)

from intermediate_layer_correction import MODEL_DEFAULTS, build_runner, load_questions
from train_intermediate_correction import compute_metrics, print_results_table


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probes(activations, labels, train_idxs, val_idxs, seed):
    """
    Fit one LR probe per (layer, head). Returns:
        probes:   list of fitted clf, indexed [layer * num_heads + head]
        val_accs: (num_layers, num_heads) array
    """
    N, num_layers, num_heads, head_dim = activations.shape

    X_train_all = activations[train_idxs]   # (n_train, L, H, D)
    X_val_all   = activations[val_idxs]     # (n_val,   L, H, D)
    y_train = labels[train_idxs]
    y_val   = labels[val_idxs]

    probes = []
    val_accs = np.zeros((num_layers, num_heads), dtype=np.float32)

    for layer in tqdm(range(num_layers), desc="training probes"):
        for head in range(num_heads):
            X_tr = X_train_all[:, layer, head, :]
            X_va = X_val_all[:,   layer, head, :]
            clf = LogisticRegression(max_iter=1000, random_state=seed, solver="saga")
            clf.fit(X_tr, y_train)
            val_accs[layer, head] = accuracy_score(y_val, clf.predict(X_va))
            probes.append(clf)

    return probes, val_accs


def get_top_heads(val_accs, num_to_select):
    """Return list of (layer, head) pairs sorted by descending val accuracy."""
    num_layers, num_heads = val_accs.shape
    flat_accs = val_accs.flatten()
    top_flat = np.argsort(flat_accs)[::-1][:num_to_select]
    return [(int(idx // num_heads), int(idx % num_heads)) for idx in top_flat]


# ---------------------------------------------------------------------------
# Build per-layer steering directions
# ---------------------------------------------------------------------------

def build_directions(top_heads, probes, tuning_activations, num_heads, head_dim,
                     use_center_of_mass, tuning_labels):
    """
    For each selected layer, build a full (num_heads * head_dim,) direction vector
    with non-zero slices only at the selected head positions.
    Scaled by the std of projections on the tuning set (as in honest_llama).

    Returns dict: {layer_idx: direction_tensor (num_heads*head_dim,)}
    """
    top_heads_by_layer = {}
    for layer, head in top_heads:
        top_heads_by_layer.setdefault(layer, []).append(head)

    directions = {}
    for layer, heads in top_heads_by_layer.items():
        full_dir = np.zeros(num_heads * head_dim, dtype=np.float32)
        for head in heads:
            probe_idx = layer * num_heads + head
            if use_center_of_mass:
                acts = tuning_activations[:, layer, head, :]
                true_mean  = acts[tuning_labels == 1].mean(axis=0)
                false_mean = acts[tuning_labels == 0].mean(axis=0)
                d = (true_mean - false_mean).astype(np.float32)
            else:
                d = probes[probe_idx].coef_[0].astype(np.float32)

            d = d / (np.linalg.norm(d) + 1e-12)

            # scale by std of projections on tuning activations
            proj_vals = tuning_activations[:, layer, head, :] @ d
            proj_std = float(np.std(proj_vals))

            full_dir[head * head_dim: (head + 1) * head_dim] = d * proj_std

        directions[layer] = full_dir

    return directions


# ---------------------------------------------------------------------------
# ITI inference
# ---------------------------------------------------------------------------

def run_iti_inference(runner, questions, directions, alpha, test_ids, logger=None):
    """
    Run forward pass with ITI hooks. directions: {layer_idx: np.ndarray (H*D,)}.
    Returns list of prediction records.
    """
    direction_tensors = {
        layer: torch.tensor(d) for layer, d in directions.items()
    }

    records = []
    skipped = 0

    for q in tqdm(questions, desc=f"ITI alpha={alpha:.1f}"):
        if not os.path.exists(q["image_path"]):
            skipped += 1
            continue

        handles = []

        def make_hook(d_tensor):
            def hook_fn(_module, args, _output):
                h = args[0]  # (batch, seq_len, H*D)
                h[0, -1, :] = h[0, -1, :] + alpha * d_tensor.to(
                    device=h.device, dtype=h.dtype
                )
            return hook_fn

        for layer_idx, d_tensor in direction_tensors.items():
            h = runner.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
                make_hook(d_tensor)
            )
            handles.append(h)

        prepared = None
        try:
            prepared = runner.prepare_image(q["image_path"], "real")
            with torch.inference_mode():
                logits = runner.forward_logits(prepared, q["question"])

            yes_l = logits[0, runner.yes_token_id].item()
            no_l  = logits[0, runner.no_token_id].item()
            records.append({
                "id":       int(q["id"]),
                "qa_type":  q.get("qa_type", ""),
                "gt_label": int(q["gt_label"]),
                "pred":     1 if yes_l > no_l else 0,
            })
        except Exception as e:
            if logger:
                logger.warning(f"Error on id={q['id']}: {e}")
            skipped += 1
        finally:
            for h in handles:
                h.remove()
            if prepared is not None:
                runner.cleanup(prepared)

        if len(records) % 100 == 0:
            torch.cuda.empty_cache()

    if skipped:
        print(f"  skipped {skipped} questions")
    return records


def records_to_arrays(records):
    return (
        np.array([r["pred"]     for r in records]),
        np.array([r["gt_label"] for r in records]),
        np.array([r["qa_type"]  for r in records]),
        np.array([r["id"]       for r in records]),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ITI probes and run intervention")
    parser.add_argument("--model",          choices=sorted(MODEL_DEFAULTS), required=True)
    parser.add_argument("--model-name",     default=None)
    parser.add_argument("--results-file",   required=True)
    parser.add_argument("--test-file",      required=True)
    parser.add_argument("--image-folder",   required=True)
    parser.add_argument("--activations-dir", default=None,
                        help="Dir containing head_activations.npz (default: model/results/iti_head_activations)")
    parser.add_argument("--output-dir",     default=None)
    parser.add_argument("--num-heads",      type=int, default=48,
                        help="Number of top heads to intervene on")
    parser.add_argument("--alpha",          type=float, default=15.0,
                        help="ITI intervention strength")
    parser.add_argument("--val-ratio",      type=float, default=0.2)
    parser.add_argument("--use-com",        action="store_true", default=False,
                        help="Use center-of-mass direction instead of LR coef")
    parser.add_argument("--load-8bit",      action="store_true", default=False)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    defaults   = MODEL_DEFAULTS[args.model]
    model_name = args.model_name or defaults["model_name"]

    acts_dir = args.activations_dir or os.path.join(
        EXPERIMENT_DIR, args.model, "results", "iti_head_activations"
    )
    output_dir = args.output_dir or os.path.join(
        EXPERIMENT_DIR, args.model, "results",
        f"iti_top{args.num_heads}_alpha{int(args.alpha)}"
        + ("_com" if args.use_com else "")
    )
    os.makedirs(output_dir, exist_ok=True)

    # --- Load activations ---
    npz_path = os.path.join(acts_dir, "head_activations.npz")
    print(f"Loading activations from {npz_path}")
    npz = np.load(npz_path)
    activations = npz["activations"]   # (N, L, H, D)
    labels      = npz["gt_labels"].astype(int)
    N, num_layers, num_heads, head_dim = activations.shape
    print(f"Activations shape: {activations.shape}")

    with open(os.path.join(acts_dir, "metadata.json")) as f:
        meta = json.load(f)
    act_image_ids = np.array([m["image_id"] for m in meta])

    # --- Train / val / test split (image-id level, matching existing convention) ---
    all_questions = load_questions(args.results_file, args.test_file, args.image_folder)

    # Reuse test_image_ids from offline_correction if available, else 20% holdout
    test_ids_path = os.path.join(
        EXPERIMENT_DIR, args.model, "results",
        f"offline_correction_layer{defaults['layer']}", "test_image_ids.json"
    )
    if os.path.exists(test_ids_path):
        with open(test_ids_path) as f:
            test_ids = set(json.load(f))
        print(f"Reusing test split from {test_ids_path}  ({len(test_ids)} images)")
    else:
        all_img_ids = list({q["id"] for q in all_questions})
        rng = np.random.RandomState(args.seed)
        rng.shuffle(all_img_ids)
        n_test = max(1, int(0.2 * len(all_img_ids)))
        test_ids = set(all_img_ids[:n_test])
        print(f"Using fresh 20% test split ({len(test_ids)} images)")

    test_qs  = [q for q in all_questions if q["id"] in test_ids]
    train_qs = [q for q in all_questions if q["id"] not in test_ids]

    # Probe train/val split on the activation indices (exclude test images)
    train_act_mask = np.isin(act_image_ids, [q["id"] for q in train_qs])
    train_idxs = np.where(train_act_mask)[0]
    rng = np.random.RandomState(args.seed)
    rng.shuffle(train_idxs)
    n_val = max(1, int(args.val_ratio * len(train_idxs)))
    val_idxs   = train_idxs[:n_val]
    inner_idxs = train_idxs[n_val:]

    print(f"Probe split — train: {len(inner_idxs)}  val: {len(val_idxs)}  test_qs: {len(test_qs)}")

    # --- Train probes ---
    print(f"\nTraining {num_layers * num_heads} probes ({num_layers}L x {num_heads}H)...")
    probes, val_accs = train_probes(activations, labels, inner_idxs, val_idxs, args.seed)

    np.save(os.path.join(output_dir, "probe_accs.npy"), val_accs)
    print(f"Val acc — mean: {val_accs.mean():.4f}  max: {val_accs.max():.4f}")

    top_heads = get_top_heads(val_accs, args.num_heads)
    print(f"Top {args.num_heads} heads (layer, head): {sorted(top_heads)[:10]} ...")
    with open(os.path.join(output_dir, "top_heads.json"), "w") as f:
        json.dump(top_heads, f)

    # --- Build directions ---
    directions = build_directions(
        top_heads, probes, activations[train_idxs], num_heads, head_dim,
        args.use_com, labels[train_idxs]
    )
    np.savez(
        os.path.join(output_dir, "directions.npz"),
        **{f"layer_{l}": d for l, d in directions.items()}
    )

    # --- Load model and run inference ---
    print(f"\nLoading model for inference...")
    runner = build_runner(args.model, model_name, args.load_8bit)
    runner.model.eval()

    print(f"Running ITI inference (alpha={args.alpha})...")
    rec_iti = run_iti_inference(runner, test_qs, directions, args.alpha, test_ids)
    print("Running baseline inference (alpha=0)...")
    rec_raw = run_iti_inference(runner, test_qs, directions, 0.0, test_ids)

    all_results = {
        "raw_model": compute_metrics(*records_to_arrays(rec_raw)),
        "iti":       compute_metrics(*records_to_arrays(rec_iti)),
    }
    print_results_table(all_results)

    # --- Save outputs ---
    out = {
        "method":     "iti",
        "model":      args.model,
        "num_heads":  args.num_heads,
        "alpha":      args.alpha,
        "use_com":    args.use_com,
        "results":    all_results,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    metric_cols = ["overall", "gt_yes", "gt_no", "adversarial", "adv_paired", "adv_wo_pair"]
    with open(os.path.join(output_dir, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + metric_cols)
        writer.writeheader()
        for method, res in all_results.items():
            writer.writerow({
                "method": method,
                **{m: f"{res.get(m, float('nan')):.4f}" for m in metric_cols}
            })

    print(f"\nDone. Results in: {output_dir}")


if __name__ == "__main__":
    main()
