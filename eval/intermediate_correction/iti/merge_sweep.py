"""
Merge sharded sweep records into a single sweep_summary.csv.

Each worker (train_iti_probes.py --num-chunks N --chunk-idx k) writes
records_chunk{k}.json = {cfg_name: [per-question record, ...], "raw": [...]}.
This script concatenates records across shards per config, recomputes metrics
over the union, and writes sweep_summary.csv + per-config results.json/.csv.

Usage:
    python merge_sweep.py --sweep-dir results/llavamed/sweep --num-chunks 4
"""

import argparse
import csv
import glob
import json
import os

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, EXPERIMENT_DIR)
from train_intermediate_correction import compute_metrics

import numpy as np

METRIC_COLS = ["overall", "gt_yes", "gt_no", "adversarial", "adv_paired", "adv_wo_pair"]


def records_to_arrays(records):
    return (
        np.array([r["pred"]     for r in records]),
        np.array([r["gt_label"] for r in records]),
        np.array([r["qa_type"]  for r in records]),
        np.array([r["id"]       for r in records]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--num-chunks", type=int, default=None,
                    help="Expected shard count (for a completeness warning)")
    args = ap.parse_args()

    shard_files = sorted(glob.glob(os.path.join(args.sweep_dir, "records_chunk*.json")))
    if not shard_files:
        raise SystemExit(f"No records_chunk*.json found in {args.sweep_dir}")
    if args.num_chunks and len(shard_files) != args.num_chunks:
        print(f"WARNING: expected {args.num_chunks} shards, found {len(shard_files)}: {shard_files}")
    print(f"Merging {len(shard_files)} shards: {[os.path.basename(f) for f in shard_files]}")

    # cfg_name -> concatenated records across shards
    merged = {}
    for sf in shard_files:
        with open(sf) as f:
            d = json.load(f)
        for cfg, recs in d.items():
            merged.setdefault(cfg, []).extend(recs)

    # sanity: every config should have the same total question count
    counts = {cfg: len(recs) for cfg, recs in merged.items()}
    n_set = set(counts.values())
    if len(n_set) != 1:
        print(f"WARNING: configs have differing record counts: {counts}")
    else:
        print(f"All configs merged to {n_set.pop()} questions.")

    raw_metrics = compute_metrics(*records_to_arrays(merged["raw"]))

    # config names sort to a stable order; parse K/alpha from the name
    def parse_cfg(name):
        # iti_top{K}_alpha{A}[_com]
        body = name.replace("iti_top", "").replace("_com", "")
        k_str, a_str = body.split("_alpha")
        return int(k_str), float(a_str)

    cfg_names = sorted([c for c in merged if c != "raw"], key=parse_cfg)
    rows = []
    for cfg in cfg_names:
        K, alpha = parse_cfg(cfg)
        m = compute_metrics(*records_to_arrays(merged[cfg]))
        rows.append((K, alpha, m))
        # write per-config results too
        cfg_dir = os.path.join(args.sweep_dir, cfg)
        os.makedirs(cfg_dir, exist_ok=True)
        all_results = {"raw_model": raw_metrics, "iti": m}
        with open(os.path.join(cfg_dir, "results.json"), "w") as f:
            json.dump({"num_heads": K, "alpha": alpha, "results": all_results}, f, indent=2)
        with open(os.path.join(cfg_dir, "results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method"] + METRIC_COLS)
            w.writeheader()
            for method, res in all_results.items():
                w.writerow({"method": method,
                            **{c: f"{res.get(c, float('nan')):.4f}" for c in METRIC_COLS}})

    summary_path = os.path.join(args.sweep_dir, "sweep_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["num_heads", "alpha"] + METRIC_COLS)
        w.writeheader()
        w.writerow({"num_heads": "raw", "alpha": 0,
                    **{c: f"{raw_metrics.get(c, float('nan')):.4f}" for c in METRIC_COLS}})
        for K, alpha, m in rows:
            w.writerow({"num_heads": K, "alpha": alpha,
                        **{c: f"{m.get(c, float('nan')):.4f}" for c in METRIC_COLS}})

    print(f"\nBaseline adv_paired: {raw_metrics.get('adv_paired', float('nan')):.4f}")
    best = max(rows, key=lambda r: (r[2].get("adv_paired", -1)
                                    if r[2].get("adv_paired", -1) == r[2].get("adv_paired", -1) else -1))
    print(f"Best ITI config: K={best[0]} alpha={best[1]}  adv_paired={best[2].get('adv_paired'):.4f}")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
