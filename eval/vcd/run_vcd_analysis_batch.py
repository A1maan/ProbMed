"""
Multi-GPU VCD Margin Analysis Runner
====================================

Runs vcd_margin_analysis.py in parallel across multiple GPUs.

Usage:
    python run_vcd_analysis_batch.py \
        --question-file /path/to/test.json \
        --image-folder /path/to/images \
        --output-file results/margin_scores.json \
        --num-chunks 4 \
        --sample-ratio 0.3
"""

import argparse
import os
import subprocess
import json
from functools import partial
from concurrent.futures import ProcessPoolExecutor


def run_chunk(chunk_idx, args):
    """Run VCD analysis for a single chunk on one GPU."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vcd_script = os.path.join(script_dir, "vcd_margin_analysis.py")
    
    # Output file for this chunk
    base_name = args.output_file.replace('.json', '')
    chunk_output = f"{base_name}-chunk{chunk_idx}.json"
    
    cmd = (
        f"CUDA_VISIBLE_DEVICES={chunk_idx} python {vcd_script} "
        f"--model-name {args.model_name} "
        f"--question-file {args.question_file} "
        f"--image-folder {args.image_folder} "
        f"--output-file {chunk_output} "
        f"--sample-ratio {args.sample_ratio} "
        f"--downsample-scale {args.downsample_scale} "
        f"--seed {args.seed} "
        f"--num-chunks {args.num_chunks} "
        f"--chunk-idx {chunk_idx} "
    )
    
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
    
    base_name = args.output_file.replace('.json', '')
    all_results = []
    
    for idx in range(args.num_chunks):
        chunk_file = f"{base_name}-chunk{idx}.json"
        if os.path.exists(chunk_file):
            with open(chunk_file, 'r') as f:
                chunk_results = json.load(f)
                all_results.extend(chunk_results)
            print(f"Merged chunk {idx}: {len(chunk_results)} results")
            # Optionally remove chunk file
            # os.remove(chunk_file)
        else:
            print(f"Warning: Chunk file not found: {chunk_file}")
    
    # Save merged results
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nMerged {len(all_results)} total results to: {args.output_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU VCD Margin Analysis')
    
    parser.add_argument("--model-name", type=str,
                        default="chaoyinshe/llava-med-v1.5-mistral-7b-hf")
    parser.add_argument("--question-file", type=str, required=True,
                        help="Path to ProbMed question JSON")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to image folder")
    parser.add_argument("--output-file", type=str, default="results/margin_scores.json",
                        help="Output file for merged results")
    parser.add_argument("--sample-ratio", type=float, default=0.3,
                        help="Ratio of data to sample (0.3 = 30%%)")
    parser.add_argument("--downsample-scale", type=float, default=0.5,
                        help="Downsampling scale for image degradation")
    parser.add_argument("--load-8bit", action="store_true", default=True,
                        help="Load model in 8-bit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--num-chunks", type=int, default=4,
                        help="Number of GPUs/chunks to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print("=" * 60)
    print("VCD Margin Analysis - Multi-GPU")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Question file: {args.question_file}")
    print(f"Image folder: {args.image_folder}")
    print(f"Output file: {args.output_file}")
    print(f"Sample ratio: {args.sample_ratio}")
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