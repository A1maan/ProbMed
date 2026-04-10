"""
Multi-GPU Hidden State Extraction Runner
=========================================

Runs extract_hidden_states.py in parallel across multiple GPUs.
Each chunk logs to {output_dir}/chunk{idx}.log. On failure a
chunk{idx}_FAILED.txt is written with the full log.

Usage:
    python run_extract_hidden_states_batch.py \
        --inference-file /path/to/chexagent.json \
        --image-folder /path/to/images \
        --output-dir results/hidden_states \
        --num-chunks 4
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np

POLL_INTERVAL = 30  # seconds between progress prints


def _last_line(path):
    """Return the last non-empty line of a file, or empty string."""
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


def build_cmd(chunk_idx, args, extract_script):
    cmd = (
        f"CUDA_VISIBLE_DEVICES={chunk_idx} /venv/main/bin/python3 {extract_script} "
        f"--inference-file {args.inference_file} "
        f"--test-file {args.test_file} "
        f"--image-folder {args.image_folder} "
        f"--output-dir {args.output_dir} "
        f"--model-name {args.model_name} "
        f"--num-chunks {args.num_chunks} "
        f"--chunk-idx {chunk_idx} "
    )
    return cmd


def launch_chunks(args):
    """Launch all chunks as subprocesses, each logging to its own file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extract_script = os.path.join(script_dir, "extract_hidden_states.py")

    procs = {}
    for chunk_idx in range(args.num_chunks):
        cmd = build_cmd(chunk_idx, args, extract_script)
        log_path = os.path.join(args.output_dir, f"chunk{chunk_idx}.log")

        print(f"[Chunk {chunk_idx}] Starting  (log: {log_path})")
        print(f"[Chunk {chunk_idx}] CMD: {cmd}")

        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT)
        procs[chunk_idx] = {"proc": proc, "log_path": log_path, "log_file": log_file}

    return procs


def wait_for_chunks(procs, output_dir):
    """Poll running chunks, printing last log line every POLL_INTERVAL seconds."""
    start = time.time()

    while True:
        running = [idx for idx, p in procs.items() if p["proc"].poll() is None]
        if not running:
            break

        elapsed = int(time.time() - start)
        print(f"\n[{elapsed}s elapsed] {len(running)} chunk(s) still running: {running}")
        for idx in running:
            last = _last_line(procs[idx]["log_path"])
            if last:
                print(f"  [Chunk {idx}] {last}")

        time.sleep(POLL_INTERVAL)

    # Close all log file handles
    for p in procs.values():
        p["log_file"].close()


def report_chunk_results(procs, output_dir):
    """Print success/failure for each chunk; write FAILED.txt on error."""
    failed = []
    for idx, p in procs.items():
        rc = p["proc"].returncode
        log_path = p["log_path"]

        if rc != 0:
            failed.append(idx)
            failed_path = os.path.join(output_dir, f"chunk{idx}_FAILED.txt")
            with open(log_path) as lf:
                log_content = lf.read()
            with open(failed_path, "w") as ff:
                ff.write(f"Chunk {idx} FAILED (exit code {rc})\n")
                ff.write(f"Log: {log_path}\n\n")
                ff.write(log_content)
            print(f"[Chunk {idx}] FAILED (rc={rc}) — details in {failed_path}")
        else:
            print(f"[Chunk {idx}] Completed successfully")
            # Remove stale failure file from a previous run
            stale = os.path.join(output_dir, f"chunk{idx}_FAILED.txt")
            if os.path.exists(stale):
                os.remove(stale)

    if failed:
        print(f"\nWarning: Chunks {failed} failed!")

    return failed


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
                        default="StanfordAIMI/CheXagent-2-3b")
    parser.add_argument("--inference-file", type=str, required=True,
                        help="Path to chexagent.json inference output")
    parser.add_argument("--test-file", type=str, required=True,
                        help="Path to test.json (used to look up image paths)")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Base image folder (image paths in test.json are relative to this)")
    parser.add_argument("--output-dir", type=str, default="results/hidden_states",
                        help="Directory for output files and logs")
    parser.add_argument("--load-8bit", action="store_true", default=False,
                        help="Unused for CheXagent (bfloat16 used instead)")
    parser.add_argument("--num-chunks", type=int, default=4,
                        help="Number of GPUs/chunks to use")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Hidden State Extraction - Multi-GPU")
    print("=" * 60)
    print(f"Model:              {args.model_name}")
    print(f"Inference file:     {args.inference_file}")
    print(f"Test file:          {args.test_file}")
    print(f"Image folder:       {args.image_folder}")
    print(f"Output dir:         {args.output_dir}")
    print(f"Num GPUs/chunks:    {args.num_chunks}")
    print(f"Progress poll:      every {POLL_INTERVAL}s")
    print("=" * 60)

    procs = launch_chunks(args)
    wait_for_chunks(procs, args.output_dir)
    report_chunk_results(procs, args.output_dir)

    print("\n" + "=" * 60)
    print("Merging results...")
    print("=" * 60)
    merge_results(args)

    print("\nDone!")


if __name__ == "__main__":
    main()
