import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel MedGemma evaluation script.")

    parser.add_argument("--model-name", type=str, default="google/medgemma-1.5-4b-it")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1, help="Unused compatibility argument")

    # Kept for compatibility (ignored)
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--answer-prompter", action="store_true")

    return parser.parse_args()


def run_job(chunk_idx, args):
    """Run inference for a single chunk."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_vqa_script = os.path.join(script_dir, "model_vqa_med.py")

    cmd = (
        f"CUDA_VISIBLE_DEVICES={chunk_idx} {shlex.quote(sys.executable)} {shlex.quote(model_vqa_script)} "
        f"--model-name {shlex.quote(args.model_name)} "
        f"--question-file {shlex.quote(args.question_file)} "
        f"--image-folder {shlex.quote(args.image_folder)} "
        f"--answers-file {shlex.quote(f'{args.experiment_name_with_split}-chunk{chunk_idx}.jsonl')} "
        f"--num-chunks {args.num_chunks} "
        f"--chunk-idx {chunk_idx} "
        f"--temperature {args.temperature} "
        f"--max-new-tokens {args.max_new_tokens} "
        f"--batch-size {args.batch_size} "
    )

    print(f"Running chunk {chunk_idx}:")
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)


def precache_model(model_name):
    """Download model weights once before spawning parallel workers."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import torch
    print(f"Pre-caching model: {model_name}")
    AutoProcessor.from_pretrained(model_name)
    AutoModelForImageTextToText.from_pretrained(
        model_name, device_map="cpu", torch_dtype=torch.bfloat16
    )
    print("Model cached. Spawning parallel workers.")


def main():
    args = parse_args()
    args.experiment_name_with_split = args.answers_file.split(".jsonl")[0]

    if args.num_chunks > 1:
        precache_model(args.model_name)

    if args.num_chunks == 1:
        run_job(0, args)
    else:
        run_job_with_args = partial(run_job, args=args)
        with ProcessPoolExecutor(max_workers=args.num_chunks) as executor:
            list(executor.map(run_job_with_args, range(args.num_chunks)))

    output_file = f"{args.experiment_name_with_split}.jsonl"
    with open(output_file, "w") as outfile:
        for idx in range(args.num_chunks):
            chunk_file = f"{args.experiment_name_with_split}-chunk{idx}.jsonl"
            if os.path.exists(chunk_file):
                with open(chunk_file) as infile:
                    outfile.write(infile.read())
                os.remove(chunk_file)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
