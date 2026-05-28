"""
Phase 1: Sweep all alphas on the val set across multiple GPUs.

Launches one subprocess per GPU (--phase val), waits for all to finish,
merges per-chunk val JSONL outputs, selects best alpha by adv_paired,
and writes best_alpha.json to output_dir.

Can be run standalone or called from run_batch.py.

Usage:
    python run_val_sweep.py --model llavamed --layer 15 --num-chunks 4
    python run_val_sweep.py --model chexagent --layer 21 --num-chunks 4
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXPERIMENT_DIR)

from train_intermediate_correction import compute_metrics
from intermediate_layer_correction import MODEL_DEFAULTS

POLL_INTERVAL = 30


def _last_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return ""
            chunk = min(size, 2048)
            f.seek(-chunk, 2)
            lines = f.read(chunk).decode(errors="replace").splitlines()
        for line in reversed(lines):
            if line.strip():
                return line.strip()
    except OSError:
        pass
    return ""


def alpha_to_tag(alpha):
    return f"{alpha:+.6f}".replace("+", "p").replace("-", "n").replace(".", "d")


def launch_val_chunks(args, output_dir, script_path, alphas):
    procs = {}
    for chunk_idx in range(args.num_chunks):
        python_bin = os.environ.get("PYTHON", "/venv/main/bin/python3")
        if not os.path.exists(python_bin):
            python_bin = "python3"
        cmd = (
            f"CUDA_VISIBLE_DEVICES={chunk_idx} "
            f"{python_bin} {script_path} "
            f"--phase val "
            f"--num-chunks {args.num_chunks} "
            f"--chunk-idx {chunk_idx} "
            f"--layer {args.layer} "
            f"--alpha-min {args.alpha_min} "
            f"--alpha-max {args.alpha_max} "
            f"--alpha-step {args.alpha_step} "
            f"--output-dir {output_dir} "
        )
        if args.direction_file:
            cmd += f"--direction-file {args.direction_file} "
        if args.test_image_ids:
            cmd += f"--test-image-ids {args.test_image_ids} "
        if args.load_8bit:
            cmd += "--load-8bit "

        log_path = os.path.join(output_dir, f"chunk{chunk_idx}-val.log")
        print(f"[Val chunk {chunk_idx}] Starting  log={log_path}")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT)
        procs[chunk_idx] = {"proc": proc, "log_path": log_path, "log_file": log_file}
    return procs


def wait_for_chunks(procs, label):
    start = time.time()
    while True:
        running = [i for i, p in procs.items() if p["proc"].poll() is None]
        if not running:
            break
        elapsed = int(time.time() - start)
        print(f"\n[{label}  {elapsed}s] {len(running)} chunk(s) running: {running}")
        for i in running:
            last = _last_line(procs[i]["log_path"])
            if last:
                print(f"  [Chunk {i}] {last}")
        time.sleep(POLL_INTERVAL)
    for p in procs.values():
        p["log_file"].close()


def check_chunks(procs, label):
    failed = []
    for idx, p in procs.items():
        rc = p["proc"].returncode
        if rc != 0:
            failed.append(idx)
            print(f"[{label} Chunk {idx}] FAILED (exit {rc}) — see {p['log_path']}")
        else:
            print(f"[{label} Chunk {idx}] OK")
    if failed:
        print(f"\nWARNING: chunks {failed} failed. Proceeding with available data.")
    return failed


def merge_val_chunks(output_dir, num_chunks, alphas):
    """Merge per-chunk val JSONL files → {alpha: [records]}."""
    merged = {}
    for alpha in alphas:
        tag = alpha_to_tag(alpha)
        recs = []
        for chunk_idx in range(num_chunks):
            path = os.path.join(output_dir, f"chunk{chunk_idx}", f"val_alpha_{tag}.jsonl")
            if not os.path.exists(path):
                print(f"  Warning: missing {path}")
                continue
            with open(path) as f:
                for line in f:
                    if line.strip():
                        recs.append(json.loads(line))
        merged[alpha] = recs
    return merged


def records_to_arrays(records):
    y_pred    = np.array([r["pred"]     for r in records])
    y_true    = np.array([r["gt_label"] for r in records])
    qa_types  = np.array([r["qa_type"]  for r in records])
    image_ids = np.array([r["id"]       for r in records])
    return y_pred, y_true, qa_types, image_ids


def select_best_alpha(val_merged, alphas):
    """Compute val metrics for each alpha, return best by adv_paired."""
    sweep_results = {}
    best_alpha, best_val_adv_paired = alphas[0], -1.0

    for alpha in alphas:
        recs = val_merged.get(alpha, [])
        if not recs:
            print(f"  No val records for alpha={alpha}, skipping")
            continue
        yp, yt, qts, ids = records_to_arrays(recs)
        m = compute_metrics(yt, yp, qts, ids)
        sweep_results[str(alpha)] = m
        print(f"  alpha={alpha:+.2f}  adv_paired={m['adv_paired']:.4f}  overall={m['overall']:.4f}")
        if m["adv_paired"] > best_val_adv_paired:
            best_val_adv_paired = m["adv_paired"]
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha:+.2f}  (val adv_paired={best_val_adv_paired:.4f})")
    return best_alpha, best_val_adv_paired, sweep_results


def run_val_sweep(args, output_dir, per_model_script, alphas):
    """
    Run Phase 1 end-to-end: launch chunks, wait, merge, pick best alpha.
    Returns (best_alpha, best_val_adv_paired, sweep_results).
    Saves best_alpha.json to output_dir.
    """
    print("=" * 60)
    print(f"Phase 1: Val sweep — {args.model}  layer={args.layer}")
    print(f"GPUs: {args.num_chunks}  alphas: {len(alphas)}")
    print("=" * 60)

    procs = launch_val_chunks(args, output_dir, per_model_script, alphas)
    wait_for_chunks(procs, "val")
    check_chunks(procs, "val")

    print("\nMerging val results...")
    val_merged = merge_val_chunks(output_dir, args.num_chunks, alphas)

    print("Selecting best alpha...")
    best_alpha, best_val_adv_paired, sweep_results = select_best_alpha(val_merged, alphas)

    # Persist for run_test_eval.py (and reproducibility)
    best_alpha_path = os.path.join(output_dir, "best_alpha.json")
    with open(best_alpha_path, "w") as f:
        json.dump({
            "best_alpha":          best_alpha,
            "best_val_adv_paired": best_val_adv_paired,
            "sweep_results":       sweep_results,
        }, f, indent=2)
    print(f"Saved best_alpha.json → {best_alpha_path}")

    return best_alpha, best_val_adv_paired, sweep_results


def main():
    parser = argparse.ArgumentParser(description="Phase 1: val alpha sweep (multi-GPU)")
    parser.add_argument("--model", choices=sorted(MODEL_DEFAULTS), required=True)
    parser.add_argument("--layer",          type=int,   default=None)
    parser.add_argument("--num-chunks",     type=int,   default=4)
    parser.add_argument("--alpha-min",      type=float, default=-2.0)
    parser.add_argument("--alpha-max",      type=float, default=2.0)
    parser.add_argument("--alpha-step",     type=float, default=0.25)
    parser.add_argument("--direction-file", default=None)
    parser.add_argument("--test-image-ids", default=None)
    parser.add_argument("--output-dir",     default=None)
    parser.add_argument("--load-8bit",      action="store_true", default=False)
    args = parser.parse_args()

    defaults = MODEL_DEFAULTS[args.model]
    args.layer = args.layer if args.layer is not None else defaults["layer"]
    if args.layer is None:
        print("ERROR: --layer is required", flush=True)
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(
        script_dir, args.model, "results", f"probe_direction_layer{args.layer}"
    )
    os.makedirs(output_dir, exist_ok=True)

    per_model_script = os.path.join(
        script_dir, args.model,
        f"probe_direction_intervention_{args.model}.py"
    )
    if not os.path.exists(per_model_script):
        print(f"ERROR: per-model script not found: {per_model_script}")
        sys.exit(1)

    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2, args.alpha_step)
    alphas = [round(float(a), 6) for a in alphas]

    run_val_sweep(args, output_dir, per_model_script, alphas)


if __name__ == "__main__":
    main()
