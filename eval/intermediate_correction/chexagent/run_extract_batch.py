"""
Multi-GPU extraction of CheXagent intermediate-layer hidden states.
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np

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


def build_cmd(chunk_idx, args, extract_script):
    return (
        f"CUDA_VISIBLE_DEVICES={chunk_idx} /venv/chexagent/bin/python3 {extract_script} "
        f"--inference-file {args.inference_file} "
        f"--test-file {args.test_file} "
        f"--image-folder {args.image_folder} "
        f"--output-dir {args.output_dir} "
        f"--model-name {args.model_name} "
        f"--target-layer {args.target_layer} "
        f"--num-chunks {args.num_chunks} "
        f"--chunk-idx {chunk_idx} "
    )


def launch_chunks(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extract_script = os.path.join(script_dir, "extract_intermediate_hidden_states.py")
    procs = {}
    for chunk_idx in range(args.num_chunks):
        cmd = build_cmd(chunk_idx, args, extract_script)
        log_path = os.path.join(args.output_dir, f"chunk{chunk_idx}.log")
        print(f"[Chunk {chunk_idx}] Starting  (log: {log_path})")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT)
        procs[chunk_idx] = {"proc": proc, "log_path": log_path, "log_file": log_file}
    return procs


def wait_for_chunks(procs):
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


def report_chunk_results(procs, output_dir):
    failed = []
    for idx, p in procs.items():
        rc = p["proc"].returncode
        if rc != 0:
            failed.append(idx)
            with open(p["log_path"]) as lf:
                log_content = lf.read()
            with open(os.path.join(output_dir, f"chunk{idx}_FAILED.txt"), "w") as ff:
                ff.write(f"Chunk {idx} FAILED (exit code {rc})\n\n{log_content}")
            print(f"[Chunk {idx}] FAILED")
        else:
            print(f"[Chunk {idx}] OK")
    return failed


def merge_results(args):
    all_hidden, all_yes, all_no, all_labels, all_ids, all_meta = [], [], [], [], [], []
    w_yes = w_no = target_layer = None

    for idx in range(args.num_chunks):
        cache_file = os.path.join(args.output_dir, f"hidden_states_cache-chunk{idx}.npz")
        meta_file  = os.path.join(args.output_dir, f"metadata-chunk{idx}.json")
        if not os.path.exists(cache_file):
            print(f"Warning: missing chunk cache {cache_file}")
            continue
        data = np.load(cache_file)
        all_hidden.append(data["hidden_states"])
        all_yes.append(data["yes_logits"])
        all_no.append(data["no_logits"])
        all_labels.append(data["gt_labels"])
        all_ids.append(data["image_ids"])
        if w_yes is None:
            w_yes = data["w_yes"]
            w_no  = data["w_no"]
            target_layer = int(data["target_layer"])
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                all_meta.extend(json.load(f))
        print(f"Merged chunk {idx}: {len(data['hidden_states'])} samples")

    if not all_hidden:
        print("No chunk results found.")
        return

    hidden_states = np.concatenate(all_hidden)
    cache_path = os.path.join(args.output_dir, "hidden_states_cache.npz")
    np.savez(cache_path,
             hidden_states=hidden_states,
             yes_logits=np.concatenate(all_yes),
             no_logits=np.concatenate(all_no),
             gt_labels=np.concatenate(all_labels),
             image_ids=np.concatenate(all_ids),
             w_yes=w_yes, w_no=w_no,
             target_layer=np.array(target_layer))
    print(f"Saved merged cache: {cache_path}  shape={hidden_states.shape}")

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"Saved merged metadata  ({len(all_meta)} entries)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",    default="StanfordAIMI/CheXagent-2-3b")
    parser.add_argument("--inference-file", required=True)
    parser.add_argument("--test-file",     required=True)
    parser.add_argument("--image-folder",  required=True)
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--target-layer",  type=int, default=21)
    parser.add_argument("--num-chunks",    type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print(f"CheXagent Intermediate Hidden State Extraction  (layer {args.target_layer})")
    print(f"Model:   {args.model_name}")
    print(f"GPUs:    {args.num_chunks}")
    print("=" * 60)

    procs = launch_chunks(args)
    wait_for_chunks(procs)
    report_chunk_results(procs, args.output_dir)
    print("\nMerging results...")
    merge_results(args)
    print("Done!")


if __name__ == "__main__":
    main()
