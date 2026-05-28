"""
Phase 2: Run test eval with the best alpha selected in Phase 1.

Reads best_alpha.json from output_dir (written by run_val_sweep.py),
launches one subprocess per GPU (--phase test --best-alpha <value>),
waits for all to finish, merges test JSONL outputs, and saves final
metrics: results.json, results.csv, alpha_sweep.png, results_subplots.png.

Can be run standalone (after run_val_sweep.py) or called from run_batch.py.

Usage:
    python run_test_eval.py --model llavamed --layer 15 --num-chunks 4
    python run_test_eval.py --model chexagent --layer 21 --num-chunks 4
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXPERIMENT_DIR)

from train_intermediate_correction import compute_metrics, print_results_table
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


def launch_test_chunks(args, output_dir, script_path, best_alpha):
    procs = {}
    for chunk_idx in range(args.num_chunks):
        python_bin = os.environ.get("PYTHON", "/venv/main/bin/python3")
        if not os.path.exists(python_bin):
            python_bin = "python3"
        cmd = (
            f"CUDA_VISIBLE_DEVICES={chunk_idx} "
            f"{python_bin} {script_path} "
            f"--phase test "
            f"--best-alpha {best_alpha} "
            f"--num-chunks {args.num_chunks} "
            f"--chunk-idx {chunk_idx} "
            f"--layer {args.layer} "
            f"--output-dir {output_dir} "
        )
        if args.direction_file:
            cmd += f"--direction-file {args.direction_file} "
        if args.test_image_ids:
            cmd += f"--test-image-ids {args.test_image_ids} "
        if args.load_8bit:
            cmd += "--load-8bit "

        log_path = os.path.join(output_dir, f"chunk{chunk_idx}-test.log")
        print(f"[Test chunk {chunk_idx}] Starting  log={log_path}")
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


def merge_test_chunks(output_dir, num_chunks, alphas):
    """Merge per-chunk test JSONL files → {alpha: [records]}."""
    merged = {}
    for alpha in alphas:
        tag = alpha_to_tag(alpha)
        recs = []
        for chunk_idx in range(num_chunks):
            path = os.path.join(output_dir, f"chunk{chunk_idx}", f"test_alpha_{tag}.jsonl")
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


def save_final_outputs(args, layer, output_dir, sweep_results,
                       best_alpha, best_val_adv_paired, test_merged):
    test_recs_corr = test_merged.get(best_alpha, [])
    test_recs_raw  = test_merged.get(0.0, [])

    all_results = {}
    if test_recs_raw:
        all_results["raw_model"] = compute_metrics(*records_to_arrays(test_recs_raw))
    if test_recs_corr:
        all_results["probe_direction"] = compute_metrics(*records_to_arrays(test_recs_corr))

    print_results_table(all_results)

    # Alpha sweep plot
    if sweep_results:
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
        ax.set_title(f"Alpha sweep — {args.model} layer {layer}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "alpha_sweep.png"), dpi=150)
        plt.close(fig)
        print("Saved alpha_sweep.png")

    # Results subplot grid
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
    print("Saved results_subplots.png")

    out = {
        "method":              "probe_direction_intervention",
        "model":               args.model,
        "layer":               layer,
        "num_chunks":          args.num_chunks,
        "best_alpha_val":      best_alpha,
        "best_val_adv_paired": best_val_adv_paired,
        "alpha_sweep_val":     sweep_results,
        "results":             all_results,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results.json")

    metric_cols = ["overall", "gt_yes", "gt_no", "adversarial", "adv_paired", "adv_wo_pair"]
    with open(os.path.join(output_dir, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + metric_cols)
        writer.writeheader()
        for method, res in all_results.items():
            writer.writerow({
                "method": method,
                **{m: f"{res.get(m, float('nan')):.4f}" for m in metric_cols}
            })
    print("Saved results.csv")
    print(f"\nDone. Results in: {output_dir}")

    return all_results


def run_test_eval(args, output_dir, per_model_script, best_alpha,
                  best_val_adv_paired, sweep_results):
    """
    Run Phase 2 end-to-end: launch test chunks, wait, merge, save outputs.
    Returns all_results dict.
    """
    layer = args.layer

    print("=" * 60)
    print(f"Phase 2: Test eval — {args.model}  layer={layer}")
    print(f"Best alpha: {best_alpha:+.2f}  GPUs: {args.num_chunks}")
    print("=" * 60)

    procs = launch_test_chunks(args, output_dir, per_model_script, best_alpha)
    wait_for_chunks(procs, "test")
    check_chunks(procs, "test")

    print("\nMerging test results...")
    test_alphas = list({best_alpha, 0.0})
    test_merged = merge_test_chunks(output_dir, args.num_chunks, test_alphas)

    print("Computing final metrics and saving outputs...")
    return save_final_outputs(args, layer, output_dir, sweep_results,
                              best_alpha, best_val_adv_paired, test_merged)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: test eval with best alpha (multi-GPU)")
    parser.add_argument("--model", choices=sorted(MODEL_DEFAULTS), required=True)
    parser.add_argument("--layer",          type=int,   default=None)
    parser.add_argument("--num-chunks",     type=int,   default=4)
    parser.add_argument("--best-alpha",     type=float, default=None,
                        help="Override best alpha (default: read from best_alpha.json)")
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

    # Load best alpha from Phase 1 output (unless overridden)
    best_alpha_path = os.path.join(output_dir, "best_alpha.json")
    if args.best_alpha is not None:
        best_alpha = args.best_alpha
        best_val_adv_paired = float("nan")
        sweep_results = {}
    elif os.path.exists(best_alpha_path):
        with open(best_alpha_path) as f:
            data = json.load(f)
        best_alpha          = data["best_alpha"]
        best_val_adv_paired = data["best_val_adv_paired"]
        sweep_results       = data["sweep_results"]
        print(f"Loaded best_alpha={best_alpha:+.2f} from {best_alpha_path}")
    else:
        print(f"ERROR: best_alpha.json not found at {best_alpha_path}")
        print("Run run_val_sweep.py first, or pass --best-alpha manually.")
        sys.exit(1)

    run_test_eval(args, output_dir, per_model_script,
                  best_alpha, best_val_adv_paired, sweep_results)


if __name__ == "__main__":
    main()
