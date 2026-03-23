"""
Multi-GPU Hidden State Extraction Runner
=========================================

Runs extract_hidden_states.py in parallel across multiple GPUs.

Usage:
    python run_extract_hidden_states_batch.py \
        --margin-scores-file results/margin_scores.json \
        --test-file /path/to/test.json \
        --image-folder /path/to/images \
        --output-dir results/hidden_states \
        --num-chunks 4
"""

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np


def run_chunk(chunk_idx, args):
    """Run hidden state extraction for a single chunk on one GPU."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    extract_script = os.path.join(script_dir, "extract_hidden_states.py")

    cmd = (
        f"CUDA_VISIBLE_DEVICES={chunk_idx} python {extract_script} "
        f"--margin-scores-file {args.margin_scores_file} "
        f"--test-file {args.test_file} "
        f"--image-folder {args.image_folder} "
        f"--output-dir {args.output_dir} "
        f"--model-name {args.model_name} "
        f"--num-chunks {args.num_chunks} "
        f"--chunk-idx {chunk_idx} "
    )

    if args.load_8bit:
        cmd += "--load-8bit "

    print(f"[Chunk {chunk_idx}] Running: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=False)

    if result.returncode != 0:
        print(f"[Chunk {chunk_idx}] FAILED with return code {result.returncode}")
        error_path = os.path.join(args.output_dir, f"chunk{chunk_idx}_FAILED.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(error_path, "w") as f:
            f.write(f"Chunk {chunk_idx} failed with return code {result.returncode}\n")
            f.write(f"Command: {cmd}\n")
        print(f"[Chunk {chunk_idx}] Error info written to {error_path}")
    else:
        print(f"[Chunk {chunk_idx}] Completed successfully")
        # Remove any stale failure file from a previous run
        error_path = os.path.join(args.output_dir, f"chunk{chunk_idx}_FAILED.txt")
        if os.path.exists(error_path):
            os.remove(error_path)

    return result.returncode


def merge_results(args):
    """Merge per-chunk .npz and metadata.json files into single output files."""

    all_hidden     = []
    all_yes_logits = []
    all_no_logits  = []
    all_gt_labels  = []
    all_image_ids  = []
    all_metadata   = []
    w_yes = w_no = None

    for idx in range(args.num_chunks):
        cache_file = os.path.join(args.output_dir, f"hidden_states_cache-chunk{idx}.npz")
        meta_file  = os.path.join(args.output_dir, f"metadata-chunk{idx}.json")

        if not os.path.exists(cache_file):
            print(f"Warning: Chunk cache not found: {cache_file}")
            continue

        data = np.load(cache_file)
        all_hidden.append(data["hidden_states"])
        all_yes_logits.append(data["yes_logits"])
        all_no_logits.append(data["no_logits"])
        all_gt_labels.append(data["gt_labels"])
        all_image_ids.append(data["image_ids"])

        # lm_head weights are identical across chunks; keep the first copy
        if w_yes is None:
            w_yes = data["w_yes"]
            w_no  = data["w_no"]

        if os.path.exists(meta_file):
            with open(meta_file) as f:
                all_metadata.extend(json.load(f))

        print(f"Merged chunk {idx}: {len(data['hidden_states'])} samples")
        # Optionally remove chunk files
        # os.remove(cache_file)
        # os.remove(meta_file)

    if not all_hidden:
        print("No chunk results found — nothing to merge.")
        return

    hidden_states = np.concatenate(all_hidden,     axis=0)
    yes_logits    = np.concatenate(all_yes_logits, axis=0)
    no_logits     = np.concatenate(all_no_logits,  axis=0)
    gt_labels     = np.concatenate(all_gt_labels,  axis=0)
    image_ids     = np.concatenate(all_image_ids,  axis=0)

    cache_path = os.path.join(args.output_dir, "hidden_states_cache.npz")
    np.savez(
        cache_path,
        hidden_states=hidden_states,
        yes_logits=yes_logits,
        no_logits=no_logits,
        gt_labels=gt_labels,
        image_ids=image_ids,
        w_yes=w_yes,
        w_no=w_no,
    )
    print(f"Saved merged cache: {cache_path}  shape={hidden_states.shape}")

    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"Saved merged metadata: {meta_path}")

    # Sanity check
    if len(hidden_states) >= 2:
        d    = (w_yes - w_no).astype(np.float32)
        H    = hidden_states[:200].astype(np.float32)
        corr = float(np.corrcoef(H @ d, (yes_logits - no_logits)[:200])[0, 1])
        print(f"Sanity check corr(d@h, logit_diff) = {corr:.4f}")
    else:
        print("Sanity check skipped: no samples extracted")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Hidden State Extraction")

    parser.add_argument("--model-name", type=str,
                        default="chaoyinshe/llava-med-v1.5-mistral-7b-hf")
    parser.add_argument("--margin-scores-file", type=str, required=True,
                        help="Path to margin scores JSON (output of VCD analysis)")
    parser.add_argument("--test-file", type=str, required=True,
                        help="Path to ProbMed test JSON")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to image folder")
    parser.add_argument("--output-dir", type=str, default="results/hidden_states",
                        help="Directory for output files")
    parser.add_argument("--load-8bit", action="store_true", default=True,
                        help="Load model in 8-bit")
    parser.add_argument("--num-chunks", type=int, default=4,
                        help="Number of GPUs/chunks to use")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Hidden State Extraction - Multi-GPU")
    print("=" * 60)
    print(f"Model:              {args.model_name}")
    print(f"Margin scores file: {args.margin_scores_file}")
    print(f"Test file:          {args.test_file}")
    print(f"Image folder:       {args.image_folder}")
    print(f"Output dir:         {args.output_dir}")
    print(f"Num GPUs/chunks:    {args.num_chunks}")
    print("=" * 60)

    run_chunk_with_args = partial(run_chunk, args=args)

    with ProcessPoolExecutor(max_workers=args.num_chunks) as executor:
        return_codes = list(executor.map(run_chunk_with_args, range(args.num_chunks)))

    failed = [i for i, rc in enumerate(return_codes) if rc != 0]
    if failed:
        print(f"\nWarning: Chunks {failed} failed!")

    print("\n" + "=" * 60)
    print("Merging results...")
    print("=" * 60)
    merge_results(args)

    print("\nDone!")


if __name__ == "__main__":
    main()
