"""
Phase 2: Train per-head LR probes on extracted activations, select top-K heads,
and run ITI inference on the test set.

Reads head_activations.npz produced by extract_head_activations.py.
Trains probes + loads model ONCE, then sweeps the K x alpha grid (model and the
alpha=0 baseline are reused across all configs).

Saves under results/<model>/sweep/:
    probe_accs.npy                  (num_layers, num_heads) val accuracy per head
    sweep_summary.csv               one row per (K, alpha) config + shared baseline
    iti_top{K}_alpha{A}/            per-config: top_heads.json, directions.npz, results.json/.csv

Usage (single config):
    python train_iti_probes.py --model llavamed --num-heads 48 --alpha 15
Usage (sweep):
    python train_iti_probes.py --model llavamed --num-heads 16,32,48 --alpha 15,30,50,100
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
ITI_DIR = os.path.dirname(os.path.abspath(__file__))  # outputs live here, under iti/
sys.path.insert(0, EXPERIMENT_DIR)

from intermediate_layer_correction import MODEL_DEFAULTS, build_runner, load_questions
from train_intermediate_correction import compute_metrics, print_results_table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_head_activations import get_head_config, get_o_proj


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
            clf = LogisticRegression(max_iter=1000, random_state=seed)
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
            # forward_pre_hook signature is (module, args); mutate input to o_proj in place
            def hook_fn(_module, args):
                h = args[0]  # (batch, seq_len, H*D)
                h[0, -1, :] = h[0, -1, :] + alpha * d_tensor.to(
                    device=h.device, dtype=h.dtype
                )
            return hook_fn

        for layer_idx, d_tensor in direction_tensors.items():
            h = get_o_proj(runner.layers[layer_idx]).register_forward_pre_hook(
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
                        help="Dir containing head_activations.npz (default: iti/results/<model>/iti_head_activations)")
    parser.add_argument("--output-dir",     default=None,
                        help="Base dir for sweep outputs (default: iti/results/<model>/sweep)")
    parser.add_argument("--num-heads",      type=str, default="48",
                        help="Number of top heads to intervene on. Comma-separated list to sweep, e.g. 16,32,48")
    parser.add_argument("--alpha",          type=str, default="15",
                        help="ITI intervention strength. Comma-separated list to sweep, e.g. 15,30,50,100")
    parser.add_argument("--val-ratio",      type=float, default=0.2)
    parser.add_argument("--use-com",        action="store_true", default=False,
                        help="Use center-of-mass direction instead of LR coef")
    parser.add_argument("--load-8bit",      action="store_true", default=False)
    parser.add_argument("--limit-test",     type=int, default=None,
                        help="Cap number of test questions for inference (smoke testing only)")
    parser.add_argument("--num-chunks",     type=int, default=1,
                        help="Shard test images across this many workers (1 = no sharding)")
    parser.add_argument("--chunk-idx",      type=int, default=0,
                        help="Which test-image shard this worker handles [0, num_chunks)")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    defaults   = MODEL_DEFAULTS[args.model]
    model_name = args.model_name or defaults["model_name"]

    # Parse sweep lists
    k_list     = [int(x)   for x in str(args.num_heads).split(",") if x.strip()]
    alpha_list = [float(x) for x in str(args.alpha).split(",")     if x.strip()]
    print(f"Sweep grid — K: {k_list}  alpha: {alpha_list}  ({len(k_list)*len(alpha_list)} configs)")

    acts_dir = args.activations_dir or os.path.join(
        ITI_DIR, "results", args.model, "iti_head_activations"
    )
    sweep_dir = args.output_dir or os.path.join(
        ITI_DIR, "results", args.model, "sweep" + ("_com" if args.use_com else "")
    )
    os.makedirs(sweep_dir, exist_ok=True)

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

    # Reuse test_image_ids from offline_correction so adv_paired is comparable to the
    # logistic_regression baseline. defaults['layer'] is None for chexagent/medgemma, so
    # glob for the offline_correction_layer* dir instead of constructing the path from it.
    import glob
    test_ids_path = None
    if defaults.get("layer") is not None:
        cand = os.path.join(EXPERIMENT_DIR, args.model, "results",
                            f"offline_correction_layer{defaults['layer']}", "test_image_ids.json")
        if os.path.exists(cand):
            test_ids_path = cand
    if test_ids_path is None:
        matches = sorted(glob.glob(os.path.join(
            EXPERIMENT_DIR, args.model, "results",
            "offline_correction_layer*", "test_image_ids.json")))
        if len(matches) > 1:
            print(f"WARNING: multiple offline_correction test splits found, using first: {matches}")
        if matches:
            test_ids_path = matches[0]

    if test_ids_path is not None:
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

    if args.limit_test is not None:
        test_qs = test_qs[:args.limit_test]
        print(f"[smoke] capping test_qs to {len(test_qs)} questions")

    # Shard test questions by image_id (keeps gt/hallu PAIRS together for adv_paired).
    sharded = args.num_chunks > 1
    if sharded:
        shard_img_ids = sorted({q["id"] for q in test_qs})
        rng_s = np.random.RandomState(args.seed)
        rng_s.shuffle(shard_img_ids)
        mine = set(shard_img_ids[args.chunk_idx::args.num_chunks])
        test_qs = [q for q in test_qs if q["id"] in mine]
        print(f"[shard {args.chunk_idx}/{args.num_chunks}] {len(mine)} images, {len(test_qs)} test_qs")

    # Probe train/val split on the activation indices (exclude test images)
    train_act_mask = np.isin(act_image_ids, [q["id"] for q in train_qs])
    train_idxs = np.where(train_act_mask)[0]
    rng = np.random.RandomState(args.seed)
    rng.shuffle(train_idxs)
    n_val = max(1, int(args.val_ratio * len(train_idxs)))
    val_idxs   = train_idxs[:n_val]
    inner_idxs = train_idxs[n_val:]

    print(f"Probe split — train: {len(inner_idxs)}  val: {len(val_idxs)}  test_qs: {len(test_qs)}")

    # --- Train probes ONCE (independent of K and alpha) ---
    print(f"\nTraining {num_layers * num_heads} probes ({num_layers}L x {num_heads}H)...")
    probes, val_accs = train_probes(activations, labels, inner_idxs, val_idxs, args.seed)
    np.save(os.path.join(sweep_dir, "probe_accs.npy"), val_accs)
    print(f"Val acc — mean: {val_accs.mean():.4f}  max: {val_accs.max():.4f}")

    # --- Load model ONCE ---
    print(f"\nLoading model for inference...")
    runner = build_runner(args.model, model_name, args.load_8bit)
    runner.model.eval()

    metric_cols = ["overall", "gt_yes", "gt_no", "adversarial", "adv_paired", "adv_wo_pair"]
    suffix = f"_chunk{args.chunk_idx}" if sharded else ""
    # records[cfg_key] = list of per-question records; cfg_key "raw" is the alpha=0 baseline
    records_by_cfg = {}

    # --- Baseline (alpha=0) inference ONCE (identical for every config) ---
    print("Running baseline inference (alpha=0)...")
    records_by_cfg["raw"] = run_iti_inference(runner, test_qs, {}, 0.0, test_ids)

    # --- Sweep K x alpha ---
    for K in k_list:
        top_heads = get_top_heads(val_accs, K)
        directions = build_directions(
            top_heads, probes, activations[train_idxs], num_heads, head_dim,
            args.use_com, labels[train_idxs]
        )
        for alpha in alpha_list:
            cfg_name = f"iti_top{K}_alpha{int(alpha)}" + ("_com" if args.use_com else "")
            cfg_dir = os.path.join(sweep_dir, cfg_name)
            os.makedirs(cfg_dir, exist_ok=True)
            # Direction artifacts are shard-independent; only chunk 0 writes them.
            if args.chunk_idx == 0:
                with open(os.path.join(cfg_dir, "top_heads.json"), "w") as f:
                    json.dump(top_heads, f)
                np.savez(os.path.join(cfg_dir, "directions.npz"),
                         **{f"layer_{l}": d for l, d in directions.items()})

            print(f"\n=== config K={K} alpha={alpha} ===")
            rec_iti = run_iti_inference(runner, test_qs, directions, alpha, test_ids)
            records_by_cfg[cfg_name] = rec_iti
            if not sharded:
                _write_config_results(cfg_dir, args, K, alpha,
                                      records_by_cfg["raw"], rec_iti, metric_cols)

    # Persist raw records for this run (merge step needs them when sharded; harmless otherwise)
    with open(os.path.join(sweep_dir, f"records{suffix}.json"), "w") as f:
        json.dump(records_by_cfg, f)
    print(f"Saved records: {sweep_dir}/records{suffix}.json")

    if sharded:
        print(f"[shard {args.chunk_idx}] done. Run merge_sweep.py to aggregate all shards.")
        return

    _write_sweep_summary(sweep_dir, records_by_cfg, k_list, alpha_list, args, metric_cols)


def _write_config_results(cfg_dir, args, K, alpha, rec_raw, rec_iti, metric_cols):
    raw_metrics = compute_metrics(*records_to_arrays(rec_raw))
    iti_metrics = compute_metrics(*records_to_arrays(rec_iti))
    all_results = {"raw_model": raw_metrics, "iti": iti_metrics}
    print_results_table(all_results)
    out = {"method": "iti", "model": args.model, "num_heads": K,
           "alpha": alpha, "use_com": args.use_com, "results": all_results}
    with open(os.path.join(cfg_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(cfg_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method"] + metric_cols)
        w.writeheader()
        for method, res in all_results.items():
            w.writerow({"method": method,
                        **{m: f"{res.get(m, float('nan')):.4f}" for m in metric_cols}})


def _write_sweep_summary(sweep_dir, records_by_cfg, k_list, alpha_list, args, metric_cols):
    raw_metrics = compute_metrics(*records_to_arrays(records_by_cfg["raw"]))
    sweep_rows = []
    for K in k_list:
        for alpha in alpha_list:
            cfg_name = f"iti_top{K}_alpha{int(alpha)}" + ("_com" if args.use_com else "")
            m = compute_metrics(*records_to_arrays(records_by_cfg[cfg_name]))
            sweep_rows.append({"num_heads": K, "alpha": alpha,
                               **{c: m.get(c, float("nan")) for c in metric_cols}})

    with open(os.path.join(sweep_dir, "sweep_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["num_heads", "alpha"] + metric_cols)
        w.writeheader()
        w.writerow({"num_heads": "raw", "alpha": 0,
                    **{c: f"{raw_metrics.get(c, float('nan')):.4f}" for c in metric_cols}})
        for r in sweep_rows:
            w.writerow({"num_heads": r["num_heads"], "alpha": r["alpha"],
                        **{c: f"{r[c]:.4f}" for c in metric_cols}})

    print(f"\nBaseline adv_paired: {raw_metrics.get('adv_paired', float('nan')):.4f}")
    best = max(sweep_rows, key=lambda r: (r["adv_paired"] if r["adv_paired"] == r["adv_paired"] else -1))
    print(f"Best ITI config: K={best['num_heads']} alpha={best['alpha']}  "
          f"adv_paired={best['adv_paired']:.4f}")
    print(f"\nDone. Sweep results in: {sweep_dir}/sweep_summary.csv")


if __name__ == "__main__":
    main()
