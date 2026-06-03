"""Merge records_confirm_chunk*.json across shards, recompute metrics, print table."""
import argparse, glob, json, os, sys
import numpy as np
EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXPERIMENT_DIR)
from train_intermediate_correction import compute_metrics

COLS = ["overall", "gt_yes", "gt_no", "adversarial", "adv_paired", "adv_wo_pair"]

def to_arrays(recs):
    return (np.array([r["pred"] for r in recs]), np.array([r["gt_label"] for r in recs]),
            np.array([r["qa_type"] for r in recs]), np.array([r["id"] for r in recs]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sweep-dir", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.sweep_dir, "records_confirm_chunk*.json")))
    if not files: raise SystemExit("no records_confirm_chunk*.json")
    merged = {}
    for f in files:
        for k, recs in json.load(open(f)).items():
            merged.setdefault(k, []).extend(recs)
    n = {k: len(v) for k, v in merged.items()}
    print(f"merged {len(files)} shards; counts: {n}")

    def alpha_key(k): return -1e9 if k == "raw" else float(k.replace("alpha", ""))
    print(f"\n{'config':>12} " + " ".join(f"{c:>11}" for c in COLS))
    for k in sorted(merged, key=alpha_key):
        m = compute_metrics(*to_arrays(merged[k]))
        print(f"{k:>12} " + " ".join(f"{m.get(c, float('nan')):>11.4f}" for c in COLS))

    out = os.path.join(args.sweep_dir, "confirm_summary.csv")
    import csv
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config"] + COLS)
        for k in sorted(merged, key=alpha_key):
            m = compute_metrics(*to_arrays(merged[k]))
            w.writerow([k] + [f"{m.get(c, float('nan')):.4f}" for c in COLS])
    print(f"\nWrote {out}")

if __name__ == "__main__":
    main()
