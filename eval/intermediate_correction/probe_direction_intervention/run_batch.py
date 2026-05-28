"""
Multi-GPU orchestrator for probe direction intervention.

Runs the full two-phase pipeline:
  Phase 1 (run_val_sweep.py)  — sweep all alphas on val, pick best alpha
  Phase 2 (run_test_eval.py)  — run test with best alpha, save final results

Usage:
    python run_batch.py --model llavamed [options]
    python run_batch.py --model chexagent --num-chunks 4 --layer 21
    python run_batch.py --model medgemma  --num-chunks 4 --layer 26
"""

import argparse
import os
import sys

import importlib.util

import numpy as np

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, EXPERIMENT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from intermediate_layer_correction import MODEL_DEFAULTS

# Import sibling modules by path so run_batch.py works regardless of cwd
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_val_sweep  = _load_module("run_val_sweep",  os.path.join(SCRIPT_DIR, "run_val_sweep.py"))
_test_eval  = _load_module("run_test_eval",  os.path.join(SCRIPT_DIR, "run_test_eval.py"))
run_val_sweep = _val_sweep.run_val_sweep
run_test_eval = _test_eval.run_test_eval


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU probe direction intervention")
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

    print("=" * 60)
    print(f"Probe Direction Intervention — {args.model}  layer={args.layer}")
    print(f"GPUs: {args.num_chunks}  alphas: {len(alphas)}")
    print("=" * 60)

    # Phase 1: val sweep
    best_alpha, best_val_adv_paired, sweep_results = run_val_sweep(
        args, output_dir, per_model_script, alphas
    )

    # Phase 2: test eval
    run_test_eval(
        args, output_dir, per_model_script,
        best_alpha, best_val_adv_paired, sweep_results
    )


if __name__ == "__main__":
    main()
