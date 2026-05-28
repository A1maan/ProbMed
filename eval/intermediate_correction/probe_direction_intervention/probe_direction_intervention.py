"""
Per-GPU worker for probe-direction intervention.

Registers a forward hook at layer L:  h[:, -1, :] += alpha * d_LR
No training — alpha is swept on val, then test is run once with best alpha.

Called by run_val_sweep.py and run_test_eval.py (via run_batch.py).
Not intended to be run directly by users.

Phase 1 — val sweep (all alphas on this chunk's val questions):
    CUDA_VISIBLE_DEVICES=K python probe_direction_intervention.py \
        --model llavamed --phase val \
        --num-chunks N --chunk-idx K --layer L ...

Phase 2 — test eval (best_alpha and alpha=0 on this chunk's test questions):
    CUDA_VISIBLE_DEVICES=K python probe_direction_intervention.py \
        --model llavamed --phase test --best-alpha 0.5 \
        --num-chunks N --chunk-idx K --layer L ...

Single-GPU (no chunking, runs full sweep + test internally):
    python probe_direction_intervention.py --model llavamed ...
"""

import argparse
import csv
import json
import logging
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXPERIMENT_DIR)

from intermediate_layer_correction import MODEL_DEFAULTS, build_runner, load_questions
from train_intermediate_correction import compute_metrics, print_results_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging(output_dir, chunk_idx=None, phase=None):
    os.makedirs(output_dir, exist_ok=True)
    parts = ["run"]
    if phase:
        parts.append(phase)
    if chunk_idx is not None:
        parts.append(f"chunk{chunk_idx}")
    log_path = os.path.join(output_dir, f"{'-'.join(parts)}.log")
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )
    return logging.getLogger(__name__)


def require_file(path, label):
    if not os.path.exists(path):
        print(f"ERROR: required file not found — {label}\n  path: {path}", flush=True)
        sys.exit(1)


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k] if k < len(chunks) else []


def alpha_to_tag(alpha):
    return f"{alpha:+.6f}".replace("+", "p").replace("-", "n").replace(".", "d")


# ---------------------------------------------------------------------------
# Single-pass inference with a fixed alpha
# ---------------------------------------------------------------------------

def eval_with_alpha(runner, questions, layer, d_LR_tensor, alpha, logger):
    """Run questions through the model with hook h += alpha*d_LR at layer L."""
    records = []
    skipped = 0

    def hook_fn(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        corrected = h.clone()
        corrected[:, -1, :] = corrected[:, -1, :] + alpha * d_LR_tensor.to(
            device=corrected.device, dtype=corrected.dtype
        )
        if isinstance(output, tuple):
            return (corrected,) + output[1:]
        return corrected

    for q in tqdm(questions, desc=f"alpha={alpha:+.2f}", leave=False):
        if not os.path.exists(q["image_path"]):
            skipped += 1
            continue
        prepared = None
        try:
            prepared = runner.prepare_image(q["image_path"], "real")
            handle = runner.layers[layer - 1].register_forward_hook(hook_fn)
            try:
                with torch.inference_mode():
                    logits = runner.forward_logits(prepared, q["question"])
            finally:
                handle.remove()

            yes_l = logits[0, runner.yes_token_id].item()
            no_l  = logits[0, runner.no_token_id].item()
            records.append({
                "id":       int(q["id"]),
                "qa_type":  q.get("qa_type", ""),
                "gt_label": int(q["gt_label"]),
                "pred":     1 if yes_l > no_l else 0,
            })

        except Exception as e:
            logger.warning(f"Error on id={q['id']}: {e}")
            skipped += 1
        finally:
            if prepared is not None:
                runner.cleanup(prepared)

        if len(records) % 100 == 0:
            torch.cuda.empty_cache()

    if skipped:
        logger.info(f"  skipped {skipped} questions")

    return records


def records_to_arrays(records):
    y_pred    = np.array([r["pred"]     for r in records])
    y_true    = np.array([r["gt_label"] for r in records])
    qa_types  = np.array([r["qa_type"]  for r in records])
    image_ids = np.array([r["id"]       for r in records])
    return y_pred, y_true, qa_types, image_ids


# ---------------------------------------------------------------------------
# Phase 1: val sweep — one chunk
# ---------------------------------------------------------------------------

def run_chunk_val_phase(args, runner, layer, d_LR_tensor, val_qs, output_dir, logger):
    """Sweep all alphas on this chunk's val slice. Saves per-alpha JSONL."""
    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2, args.alpha_step)
    alphas = [round(float(a), 6) for a in alphas]

    val_chunk = get_chunk(val_qs, args.num_chunks, args.chunk_idx)
    logger.info(f"Phase 1  chunk={args.chunk_idx}  val_questions={len(val_chunk)}")

    chunk_dir = os.path.join(output_dir, f"chunk{args.chunk_idx}")
    os.makedirs(chunk_dir, exist_ok=True)

    for alpha in alphas:
        logger.info(f"  alpha={alpha:+.2f}")
        recs = eval_with_alpha(runner, val_chunk, layer, d_LR_tensor, alpha, logger)
        out_path = os.path.join(chunk_dir, f"val_alpha_{alpha_to_tag(alpha)}.jsonl")
        with open(out_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        logger.info(f"    saved {len(recs)} records → {out_path}")


# ---------------------------------------------------------------------------
# Phase 2: test eval — one chunk
# ---------------------------------------------------------------------------

def run_chunk_test_phase(args, runner, layer, d_LR_tensor, test_qs, output_dir, logger):
    """Run test questions for best_alpha and alpha=0 (raw baseline)."""
    test_chunk = get_chunk(test_qs, args.num_chunks, args.chunk_idx)
    logger.info(f"Phase 2  chunk={args.chunk_idx}  test_questions={len(test_chunk)}")

    chunk_dir = os.path.join(output_dir, f"chunk{args.chunk_idx}")
    os.makedirs(chunk_dir, exist_ok=True)

    for alpha in [args.best_alpha, 0.0]:
        logger.info(f"  test alpha={alpha:+.2f}")
        recs = eval_with_alpha(runner, test_chunk, layer, d_LR_tensor, alpha, logger)
        out_path = os.path.join(chunk_dir, f"test_alpha_{alpha_to_tag(alpha)}.jsonl")
        with open(out_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        logger.info(f"    saved {len(recs)} records → {out_path}")


# ---------------------------------------------------------------------------
# Single-GPU mode: full pipeline in one process
# ---------------------------------------------------------------------------

def run_single_gpu_mode(args, runner, layer, d_LR_tensor,
                        val_qs, test_qs, output_dir, logger):
    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2, args.alpha_step)
    alphas = [round(float(a), 6) for a in alphas]

    logger.info(f"Sweeping {len(alphas)} alphas on val set ({len(val_qs)} questions)")

    sweep_results = {}
    best_alpha, best_val_adv_paired = alphas[0], -1.0

    for alpha in alphas:
        logger.info(f"Alpha={alpha:+.2f}")
        records = eval_with_alpha(runner, val_qs, layer, d_LR_tensor, alpha, logger)
        if not records:
            logger.warning(f"  No predictions for alpha={alpha}, skipping")
            continue
        yp, yt, qts, ids = records_to_arrays(records)
        m = compute_metrics(yt, yp, qts, ids)
        sweep_results[str(alpha)] = m
        logger.info(f"  adv_paired={m['adv_paired']:.4f}  overall={m['overall']:.4f}")
        if m["adv_paired"] > best_val_adv_paired:
            best_val_adv_paired = m["adv_paired"]
            best_alpha = alpha

    logger.info(f"Best alpha: {best_alpha:+.2f}  (val adv_paired={best_val_adv_paired:.4f})")
    _save_alpha_sweep_plot(sweep_results, best_alpha, args.model, layer, output_dir)

    logger.info(f"Test eval  alpha={best_alpha:+.2f}")
    rec_corr = eval_with_alpha(runner, test_qs, layer, d_LR_tensor, best_alpha, logger)
    logger.info("Test eval  alpha=0 (raw baseline)")
    rec_raw  = eval_with_alpha(runner, test_qs, layer, d_LR_tensor, 0.0,        logger)

    all_results = {
        "raw_model":       compute_metrics(*records_to_arrays(rec_raw)),
        "probe_direction": compute_metrics(*records_to_arrays(rec_corr)),
    }
    print_results_table(all_results)
    _save_outputs(args, layer, output_dir, sweep_results, best_alpha,
                  best_val_adv_paired, all_results, logger)


# ---------------------------------------------------------------------------
# Saving helpers (single-GPU mode)
# ---------------------------------------------------------------------------

def _save_alpha_sweep_plot(sweep_results, best_alpha, model, layer, output_dir):
    sweep_alphas  = [float(a) for a in sweep_results]
    sweep_adv     = [sweep_results[str(a)]["adv_paired"] for a in sweep_alphas]
    sweep_overall = [sweep_results[str(a)]["overall"]    for a in sweep_alphas]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sweep_alphas, sweep_adv,     marker="o", label="adv_paired")
    ax.plot(sweep_alphas, sweep_overall, marker="s", label="overall")
    ax.axvline(best_alpha, color="red", linestyle="--", alpha=0.6,
               label=f"best α={best_alpha:+.2f}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Alpha sweep — {model} layer {layer}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "alpha_sweep.png"), dpi=150)
    plt.close(fig)


def _save_outputs(args, layer, output_dir, sweep_results, best_alpha,
                  best_val_adv_paired, all_results, logger):
    import shutil

    methods = list(all_results.keys())
    plot_metrics = {
        "overall":     "Overall Accuracy",
        "gt_yes":      "GT=Yes Accuracy",
        "gt_no":       "GT=No Accuracy",
        "adversarial": "Adversarial (hallu-only)",
        "adv_paired":  "Adversarial (paired, strict)",
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
    fig.suptitle(
        f"Probe Direction Intervention — {args.model} layer {layer}",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "results_subplots.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved results_subplots.png")

    out = {
        "method":              "probe_direction_intervention",
        "model":               args.model,
        "layer":               layer,
        "d_LR_source":         args.direction_file,
        "best_alpha_val":      best_alpha,
        "best_val_adv_paired": best_val_adv_paired,
        "alpha_sweep_val":     sweep_results,
        "results":             all_results,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved results.json")

    metric_cols = ["overall", "gt_yes", "gt_no", "adversarial", "adv_paired", "adv_wo_pair"]
    with open(os.path.join(output_dir, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + metric_cols)
        writer.writeheader()
        for method, res in all_results.items():
            writer.writerow({
                "method": method,
                **{m: f"{res.get(m, float('nan')):.4f}" for m in metric_cols}
            })
    logger.info("Saved results.csv")

    src = args.test_image_ids
    if src and os.path.exists(src):
        shutil.copy(src, os.path.join(output_dir, "test_image_ids.json"))
        logger.info("Copied test_image_ids.json")

    logger.info(f"Done. Results in: {output_dir}")


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
        description="Probe-direction intervention per-GPU worker"
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
    parser.add_argument("--direction-file", default=None)
    parser.add_argument("--test-image-ids", default=None)
    parser.add_argument("--layer",      type=int,   default=None)
    parser.add_argument("--alpha-min",  type=float, default=-2.0)
    parser.add_argument("--alpha-max",  type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.25)
    parser.add_argument("--num-chunks", type=int,   default=1)
    parser.add_argument("--chunk-idx",  type=int,   default=0)
    parser.add_argument("--phase",      choices=["val", "test"], default=None,
                        help="val=sweep all alphas on val; test=run best alpha on test")
    parser.add_argument("--best-alpha", type=float, default=None,
                        help="Best alpha from val phase (required for --phase test)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--load-8bit", action="store_true", default=default_load_8bit)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    defaults   = MODEL_DEFAULTS[args.model]
    model_name = args.model_name or defaults["model_name"]
    layer      = args.layer if args.layer is not None else defaults["layer"]
    if layer is None:
        print("ERROR: --layer is required", flush=True)
        sys.exit(1)

    model_results_base = os.path.join(
        EXPERIMENT_DIR, args.model, "results", f"offline_correction_layer{layer}"
    )
    args.direction_file = args.direction_file or os.path.join(
        model_results_base, "trained_lr_direction.npy"
    )
    args.test_image_ids = args.test_image_ids or os.path.join(
        model_results_base, "test_image_ids.json"
    )

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.model, "results", f"probe_direction_layer{layer}"
    )

    is_chunk = args.num_chunks > 1
    logger = setup_logging(
        output_dir,
        chunk_idx=args.chunk_idx if is_chunk else None,
        phase=args.phase if is_chunk else None,
    )

    require_file(args.direction_file, "trained_lr_direction.npy")
    require_file(args.test_image_ids, "test_image_ids.json")
    require_file(args.results_file,   "results_file")
    require_file(args.test_file,      "test_file")

    logger.info(f"Model:      {args.model} ({model_name})")
    logger.info(f"Layer:      {layer}")
    logger.info(f"Direction:  {args.direction_file}")
    logger.info(f"Chunks:     {args.chunk_idx + 1}/{args.num_chunks}")
    logger.info(f"Output dir: {output_dir}")

    d_LR = np.load(args.direction_file).astype(np.float32)
    d_LR = d_LR / (np.linalg.norm(d_LR) + 1e-12)
    d_LR_tensor = torch.tensor(d_LR)
    logger.info(f"d_LR: shape={d_LR.shape}  norm={np.linalg.norm(d_LR):.4f}")

    with open(args.test_image_ids) as f:
        test_ids = set(json.load(f))

    all_questions = load_questions(args.results_file, args.test_file, args.image_folder)
    train_qs = [q for q in all_questions if q["id"] not in test_ids]
    test_qs  = [q for q in all_questions if q["id"] in test_ids]

    rng = np.random.RandomState(args.seed)
    train_img_ids = list({q["id"] for q in train_qs})
    rng.shuffle(train_img_ids)
    n_val = max(1, int(0.2 * len(train_img_ids)))
    val_img_ids = set(train_img_ids[:n_val])
    val_qs = [q for q in train_qs if q["id"] in val_img_ids]

    logger.info(f"Train: {len(train_qs)}  Val: {len(val_qs)}  Test: {len(test_qs)}")

    runner = build_runner(args.model, model_name, args.load_8bit)
    runner.model.eval()

    if is_chunk:
        if args.phase == "val":
            run_chunk_val_phase(args, runner, layer, d_LR_tensor, val_qs, output_dir, logger)
        elif args.phase == "test":
            if args.best_alpha is None:
                print("ERROR: --best-alpha required for --phase test", flush=True)
                sys.exit(1)
            run_chunk_test_phase(args, runner, layer, d_LR_tensor, test_qs, output_dir, logger)
        else:
            print("ERROR: --phase val or --phase test required when --num-chunks > 1", flush=True)
            sys.exit(1)
    else:
        run_single_gpu_mode(
            args, runner, layer, d_LR_tensor, val_qs, test_qs, output_dir, logger
        )


if __name__ == "__main__":
    main()
