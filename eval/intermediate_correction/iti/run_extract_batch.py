"""
Launch head-activation extraction across N GPUs, then merge chunks.

Usage:
    python run_extract_batch.py --model llavamed --num-chunks 4 --max-samples 10000
    python run_extract_batch.py --model chexagent --num-chunks 4
    python run_extract_batch.py --model medgemma --num-chunks 4
"""

import argparse
import os
import subprocess
import sys
import time

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITI_DIR = os.path.dirname(os.path.abspath(__file__))  # outputs live here, under iti/
sys.path.insert(0, EXPERIMENT_DIR)

from intermediate_layer_correction import MODEL_DEFAULTS

POLL_INTERVAL = 30
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_head_activations.py")


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


def launch_chunks(args, output_dir):
    python_bin = os.environ.get("PYTHON", "/venv/main/bin/python3")
    if not os.path.exists(python_bin):
        python_bin = "python3"

    procs = {}
    for chunk_idx in range(args.num_chunks):
        cmd = (
            f"CUDA_VISIBLE_DEVICES={chunk_idx} "
            f"{python_bin} {SCRIPT} "
            f"--model {args.model} "
            f"--results-file {args.results_file} "
            f"--test-file {args.test_file} "
            f"--image-folder {args.image_folder} "
            f"--output-dir {output_dir} "
            f"--num-chunks {args.num_chunks} "
            f"--chunk-idx {chunk_idx} "
            f"--max-samples {args.max_samples} "
            f"--seed {args.seed} "
        )
        if args.load_8bit:
            cmd += "--load-8bit "

        log_path = os.path.join(output_dir, f"extract_chunk{chunk_idx}.log")
        print(f"[Chunk {chunk_idx}] Starting  log={log_path}")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT)
        procs[chunk_idx] = {"proc": proc, "log_path": log_path, "log_file": log_file}

    return procs


def wait_and_check(procs):
    start = time.time()
    while True:
        running = [i for i, p in procs.items() if p["proc"].poll() is None]
        if not running:
            break
        elapsed = int(time.time() - start)
        print(f"\n[{elapsed}s] {len(running)} chunk(s) running: {running}")
        for i in running:
            last = _last_line(procs[i]["log_path"])
            if last:
                print(f"  [Chunk {i}] {last}")
        time.sleep(POLL_INTERVAL)

    for p in procs.values():
        p["log_file"].close()

    failed = [i for i, p in procs.items() if p["proc"].returncode != 0]
    for i in failed:
        print(f"[Chunk {i}] FAILED — see {procs[i]['log_path']}")
    if not failed:
        print("All chunks completed successfully.")
    return failed


def merge(output_dir, num_chunks):
    # Import and call directly to avoid argparse conflicts with the worker script.
    sys.path.insert(0, os.path.dirname(SCRIPT))
    from extract_head_activations import merge_chunks
    merge_chunks(output_dir, num_chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         choices=sorted(MODEL_DEFAULTS), required=True)
    parser.add_argument("--results-file",  required=True)
    parser.add_argument("--test-file",     required=True)
    parser.add_argument("--image-folder",  required=True)
    parser.add_argument("--output-dir",    default=None)
    parser.add_argument("--num-chunks",    type=int, default=4)
    parser.add_argument("--max-samples",   type=int, default=10000)
    parser.add_argument("--load-8bit",     action="store_true", default=False)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        ITI_DIR, "results", args.model, "iti_head_activations"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Extracting head activations for {args.model} across {args.num_chunks} GPUs")
    procs = launch_chunks(args, output_dir)
    failed = wait_and_check(procs)

    if failed:
        print(f"WARNING: {len(failed)} chunks failed. Merging available data anyway.")

    print("\nMerging chunks...")
    from extract_head_activations import merge_chunks
    merge_chunks(output_dir, args.num_chunks)
    print("Done.")


if __name__ == "__main__":
    main()
