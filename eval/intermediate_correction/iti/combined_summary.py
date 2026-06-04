"""
Combined cross-model summary of all intervention methods on adv_paired.

For each model (llavamed, chexagent, medgemma) pulls:
  - raw baseline                (offline_correction results.csv)
  - logistic_regression ceiling (offline_correction results.csv)  [offline decode upper bound]
  - offline correction model    (offline_correction results.csv)
  - vector_correction (learned) (vector_correction_layer*/results.csv) [fair vs raw]
  - best ITI config (+alpha)    (iti sweep_summary.csv)  [reports best + that it's <= raw]

Writes:
  iti/results/combined_summary.csv
  iti/results/combined_summary.png   (grouped bar chart of adv_paired per method per model)
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # intermediate_correction
ITI = os.path.dirname(os.path.abspath(__file__))                    # iti
MODELS = ["llavamed", "chexagent", "medgemma"]
LAYER = {"llavamed": 15, "chexagent": 21, "medgemma": 26}


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def offline_metric(model, method, col="adv_paired"):
    rows = read_csv_rows(os.path.join(
        EXP, model, "results", f"offline_correction_layer{LAYER[model]}", "results.csv"))
    for r in rows:
        if r.get("method") == method:
            try:
                return float(r[col])
            except (ValueError, KeyError):
                return float("nan")
    return float("nan")


def vector_correction_metric(model, col="adv_paired"):
    # prefer the non-"full1ep" dir if present, else any
    cands = sorted(glob.glob(os.path.join(EXP, model, "results", "vector_correction_layer*", "results.csv")))
    for path in cands:
        for r in read_csv_rows(path):
            if r.get("method") == "vector_correction":
                try:
                    return float(r[col]), os.path.basename(os.path.dirname(path))
                except (ValueError, KeyError):
                    pass
    return float("nan"), None


def best_iti(model, col="adv_paired"):
    """Return (best_value, K, alpha) over the ITI sweep, and the raw row for reference."""
    rows = read_csv_rows(os.path.join(ITI, "results", model, "sweep", "sweep_summary.csv"))
    best = (-1.0, None, None)
    raw = float("nan")
    for r in rows:
        if r["num_heads"] == "raw":
            raw = float(r[col])
            continue
        v = float(r[col])
        if v > best[0]:
            best = (v, int(r["num_heads"]), float(r["alpha"]))
    return best, raw


def main():
    cols = ["raw", "best_ITI(+a)", "vector_correction", "offline_correction", "LR_ceiling"]
    table = {}   # model -> {col: value}
    notes = {}

    for m in MODELS:
        raw = offline_metric(m, "raw_model")
        lr  = offline_metric(m, "logistic_regression")
        off = offline_metric(m, "correction")
        vc, vc_dir = vector_correction_metric(m)
        (iti_best, iti_K, iti_a), iti_raw = best_iti(m)

        table[m] = {
            "raw": raw,
            "best_ITI(+a)": iti_best if iti_best >= 0 else float("nan"),
            "vector_correction": vc,
            "offline_correction": off,
            "LR_ceiling": lr,
        }
        notes[m] = f"ITI best K={iti_K} a={iti_a} (sweep raw={iti_raw:.4f}); vc dir={vc_dir}"

    # ---- table ----
    print(f"\n{'model':>10} " + " ".join(f"{c:>18}" for c in cols))
    for m in MODELS:
        print(f"{m:>10} " + " ".join(f"{table[m][c]:>18.4f}" for c in cols))
    print()
    for m in MODELS:
        print(f"  {m}: {notes[m]}")

    out_csv = os.path.join(ITI, "results", "combined_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + cols + ["notes"])
        for m in MODELS:
            w.writerow([m] + [f"{table[m][c]:.4f}" for c in cols] + [notes[m]])
    print(f"\nWrote {out_csv}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(MODELS))
    width = 0.16
    colors = {"raw": "#888888", "best_ITI(+a)": "#d62728",
              "vector_correction": "#2ca02c", "offline_correction": "#1f77b4",
              "LR_ceiling": "#9467bd"}
    for i, c in enumerate(cols):
        vals = [table[m][c] for m in MODELS]
        bars = ax.bar(x + (i - len(cols)/2 + 0.5) * width, vals, width,
                      label=c, color=colors[c])
        for b, v in zip(bars, vals):
            if v == v:  # not nan
                ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("adv_paired (strict)")
    ax.set_ylim(0, 0.85)
    ax.set_title("ProbMed adv_paired by method — steering vs. offline vs. learned correction")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(ITI, "results", "combined_summary.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
