"""
Multi-GPU Attention Analysis Runner
===================================

Runs attention_analysis.py in parallel across multiple GPUs.

Usage:
    python run_attention_batch.py \
        --margin-scores-file ../vcd/results/vcd_analysis/margin_scores.json \
        --image-folder /workspace/ProbMed-Dataset/test/ \
        --output-dir results/attention_analysis \
        --num-pairs 1000 \
        --num-chunks 4
"""

import argparse
import os
import subprocess
import json
from functools import partial
from concurrent.futures import ProcessPoolExecutor


def run_chunk(chunk_idx, args):
    """Run attention analysis for a single chunk on one GPU."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    attention_script = os.path.join(script_dir, "attention_analysis.py")
    
    # Output file for this chunk
    chunk_output_dir = os.path.join(args.output_dir, f"chunk{chunk_idx}")
    
    cmd = (
        f"CUDA_VISIBLE_DEVICES={chunk_idx} python {attention_script} "
        f"--model-name {args.model_name} "
        f"--margin-scores-file {args.margin_scores_file} "
        f"--image-folder {args.image_folder} "
        f"--output-dir {chunk_output_dir} "
        f"--num-chunks {args.num_chunks} "
        f"--chunk-idx {chunk_idx} "
        f"--seed {args.seed + chunk_idx} "
    )
    
    if args.test_file:
        cmd += f"--test-file {args.test_file} "
    
    if args.num_pairs:
        cmd += f"--num-pairs {args.num_pairs} "
    
    if args.load_8bit:
        cmd += "--load-8bit "
    
    print(f"[Chunk {chunk_idx}] Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"[Chunk {chunk_idx}] FAILED with return code {result.returncode}")
    else:
        print(f"[Chunk {chunk_idx}] Completed successfully")
    
    return result.returncode


def merge_results(args):
    """Merge chunk results into a single output file."""
    
    all_results = []
    
    for idx in range(args.num_chunks):
        chunk_dir = os.path.join(args.output_dir, f"chunk{idx}")
        chunk_file = os.path.join(chunk_dir, "attention_analysis.json")
        
        if os.path.exists(chunk_file):
            with open(chunk_file, 'r') as f:
                chunk_results = json.load(f)
                all_results.extend(chunk_results)
            print(f"Merged chunk {idx}: {len(chunk_results)} results")
        else:
            print(f"Warning: Chunk file not found: {chunk_file}")
    
    # Save merged results
    merged_file = os.path.join(args.output_dir, "attention_analysis.json")
    with open(merged_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nMerged {len(all_results)} total results to: {merged_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Attention Analysis')
    
    parser.add_argument("--model-name", type=str,
                        default="chaoyinshe/llava-med-v1.5-mistral-7b-hf")
    parser.add_argument("--margin-scores-file", type=str, required=True,
                        help="Path to margin_scores.json from VCD experiment")
    parser.add_argument("--test-file", type=str, default=None,
                        help="Path to original test.json (to get image paths)")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to image folder")
    parser.add_argument("--output-dir", type=str, default="results/attention_analysis",
                        help="Output directory for merged results")
    parser.add_argument("--num-pairs", type=int, default=None,
                        help="Total number of pairs to analyze (default: all pairs)")
    parser.add_argument("--num-chunks", type=int, default=4,
                        help="Number of GPUs/chunks to use")
    parser.add_argument("--load-8bit", action="store_true", default=True,
                        help="Load model in 8-bit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # pairs_per_chunk not needed when processing all - each chunk will get its portion
    args.pairs_per_chunk = args.num_pairs // args.num_chunks if args.num_pairs else None
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Attention Analysis - Multi-GPU")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Margin scores: {args.margin_scores_file}")
    print(f"Image folder: {args.image_folder}")
    print(f"Output dir: {args.output_dir}")
    print(f"Total pairs: {args.num_pairs if args.num_pairs else 'ALL'}")
    print(f"Num GPUs/chunks: {args.num_chunks}")
    print("=" * 60)
    
    # Run chunks in parallel
    run_chunk_with_args = partial(run_chunk, args=args)
    
    with ProcessPoolExecutor(max_workers=args.num_chunks) as executor:
        return_codes = list(executor.map(run_chunk_with_args, range(args.num_chunks)))
    
    # Check for failures
    failed = [i for i, rc in enumerate(return_codes) if rc != 0]
    if failed:
        print(f"\nWarning: Chunks {failed} failed!")
    
    # Merge results
    print("\n" + "=" * 60)
    print("Merging results...")
    print("=" * 60)
    merge_results(args)
    
    print("\nDone!")


if __name__ == "__main__":
    main()