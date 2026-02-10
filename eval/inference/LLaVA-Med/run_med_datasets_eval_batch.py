"""
Replacement for: eval/inference/LLaVA-Med/run_med_datasets_eval_batch.py

Modified to work with HuggingFace LLaVA-Med weights.
Handles parallel chunk processing.

Usage:
    python run_med_datasets_eval_batch.py \
        --model-name chaoyinshe/llava-med-v1.5-mistral-7b-hf \
        --question-file /path/to/probmed.json \
        --image-folder /path/to/images \
        --answers-file /path/to/output.jsonl \
        --num-chunks 4
"""

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(description='Parallel LLaVA-Med HF evaluation script.')

    parser.add_argument("--model-name", type=str, 
                        default="chaoyinshe/llava-med-v1.5-mistral-7b-hf")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--load-8bit", action="store_true", default=False,
                        help="Load model in 8-bit quantization")
    parser.add_argument("--no-8bit", action="store_true", default=False,
                        help="Disable 8-bit quantization (use fp16)")
    
    # Kept for compatibility
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--answer-prompter", action="store_true")

    return parser.parse_args()


def run_job(chunk_idx, args):
    """Run inference for a single chunk."""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_vqa_script = os.path.join(script_dir, "model_vqa_med.py")
    
    cmd = (
        "CUDA_VISIBLE_DEVICES={chunk_idx} python {model_vqa_script} "
        "--model-name {model_name} "
        "--question-file {question_file} "
        "--image-folder {image_folder} "
        "--answers-file {experiment_name_with_split}-chunk{chunk_idx}.jsonl "
        "--num-chunks {chunks} "
        "--chunk-idx {chunk_idx} "
        "--temperature {temperature} "
        "--max-new-tokens {max_new_tokens} "
        "--batch-size {batch_size} "
    ).format(
        chunk_idx=chunk_idx,
        model_vqa_script=model_vqa_script,
        chunks=args.num_chunks,
        model_name=args.model_name,
        question_file=args.question_file,
        image_folder=args.image_folder,
        experiment_name_with_split=args.experiment_name_with_split,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    
    if args.load_8bit:
        cmd += "--load-8bit "
    
    if args.no_8bit:
        cmd += "--no-8bit "
    
    print(f"Running chunk {chunk_idx}:")
    print(cmd)
    
    subprocess.run(cmd, shell=True, check=True)


def main():
    args = parse_args()
    args.experiment_name_with_split = args.answers_file.split(".jsonl")[0]

    # For single GPU/chunk, just run directly
    if args.num_chunks == 1:
        run_job(0, args)
    else:
        # Run jobs in parallel
        run_job_with_args = partial(run_job, args=args)
        
        with ProcessPoolExecutor(max_workers=args.num_chunks) as executor:
            list(executor.map(run_job_with_args, range(args.num_chunks)))

    # Gather results from all chunks
    output_file = f"{args.experiment_name_with_split}.jsonl"
    with open(output_file, 'w') as outfile:
        for idx in range(args.num_chunks):
            chunk_file = f"{args.experiment_name_with_split}-chunk{idx}.jsonl"
            if os.path.exists(chunk_file):
                with open(chunk_file) as infile:
                    outfile.write(infile.read())
                # Clean up chunk file
                os.remove(chunk_file)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()