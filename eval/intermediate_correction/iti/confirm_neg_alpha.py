"""
Confirmation run: reuse already-built directions for one config and evaluate
baseline + a list of (typically negative) alphas on the FULL test set, sharded.

Reuses results/<model>/sweep/iti_top{K}_alpha{A}/directions.npz (any A — the
direction tensors are alpha-independent), so no probe retraining / re-derivation.

Each shard writes records_confirm_chunk{c}.json; merge with:
    python merge_sweep.py --sweep-dir <dir>   (records_confirm_chunk* are picked up too)

Usage (one shard):
    python confirm_neg_alpha.py --model llavamed --K 48 --alphas 0,-5,-10,-15 \\
        --num-chunks 4 --chunk-idx 0 --load-8bit
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
from tqdm import tqdm

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITI_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENT_DIR)
sys.path.insert(0, ITI_DIR)
from intermediate_layer_correction import MODEL_DEFAULTS, build_runner, load_questions
from extract_head_activations import get_o_proj


def load_directions(sweep_dir, K):
    # any alpha subdir for this K has the same directions.npz
    cands = sorted(glob.glob(os.path.join(sweep_dir, f"iti_top{K}_alpha*", "directions.npz")))
    if not cands:
        raise SystemExit(f"No directions.npz for K={K} in {sweep_dir}")
    d = np.load(cands[0])
    print(f"Loaded directions from {cands[0]}")
    return {int(k.split("_")[1]): torch.tensor(d[k]) for k in d.files}


def run(runner, questions, directions, alpha):
    recs, skipped = [], 0
    for q in tqdm(questions, desc=f"alpha={alpha}"):
        if not os.path.exists(q["image_path"]):
            skipped += 1; continue
        handles = []
        def mk(dt):
            def hook(_m, args):
                h = args[0]; h[0, -1, :] = h[0, -1, :] + alpha * dt.to(h.device, h.dtype)
            return hook
        for l, dt in directions.items():
            handles.append(get_o_proj(runner.layers[l]).register_forward_pre_hook(mk(dt)))
        try:
            with torch.inference_mode():
                lg = runner.forward_logits(
                    runner.prepare_image(q["image_path"], "real"), q["question"])
            recs.append({"id": int(q["id"]), "qa_type": q.get("qa_type", ""),
                         "gt_label": int(q["gt_label"]),
                         "pred": 1 if lg[0, runner.yes_token_id] > lg[0, runner.no_token_id] else 0})
        except Exception as e:
            skipped += 1
        finally:
            for h in handles: h.remove()
        if len(recs) % 100 == 0:
            torch.cuda.empty_cache()
    if skipped: print(f"  skipped {skipped}")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_DEFAULTS))
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--alphas", required=True, help="comma list incl 0 for baseline, e.g. 0,-5,-10,-15")
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--chunk-idx", type=int, default=0)
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    defaults = MODEL_DEFAULTS[args.model]
    sweep_dir = os.path.join(ITI_DIR, "results", args.model, "sweep")
    directions = load_directions(sweep_dir, args.K)

    # Same test split as the sweep (offline_correction reuse via glob).
    matches = sorted(glob.glob(os.path.join(
        EXPERIMENT_DIR, args.model, "results", "offline_correction_layer*", "test_image_ids.json")))
    test_ids = set(json.load(open(matches[0])))
    all_q = load_questions(os.path.join(EXPERIMENT_DIR, "..", "response_file", f"{args.model}.json"),
                           "/workspace/ProbMed-Dataset/test/test.json",
                           "/workspace/ProbMed-Dataset/test")
    test_qs = [q for q in all_q if q["id"] in test_ids]

    if args.num_chunks > 1:
        img_ids = sorted({q["id"] for q in test_qs})
        rng = np.random.RandomState(args.seed); rng.shuffle(img_ids)
        mine = set(img_ids[args.chunk_idx::args.num_chunks])
        test_qs = [q for q in test_qs if q["id"] in mine]
        print(f"[shard {args.chunk_idx}/{args.num_chunks}] {len(mine)} images, {len(test_qs)} qs")

    runner = build_runner(args.model, defaults["model_name"], args.load_8bit)
    runner.model.eval()

    out = {}
    for a in alphas:
        key = "raw" if a == 0 else f"alpha{a}"
        out[key] = run(runner, test_qs, {} if a == 0 else directions, a)

    suffix = f"_chunk{args.chunk_idx}" if args.num_chunks > 1 else ""
    path = os.path.join(sweep_dir, f"records_confirm{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
