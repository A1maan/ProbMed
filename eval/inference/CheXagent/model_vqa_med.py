import argparse
import json
import math
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks."""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k] if k < len(chunks) else []


def load_model(model_name):
    """Load the CheXagent-2 model."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    model = model.to(torch.bfloat16)
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


def run_inference_single(model, tokenizer, image_path, question, max_new_tokens=512):
    """Run inference for a single image-question pair."""
    query = tokenizer.from_list_format([
        {"image": image_path},
        {"text": question},
    ])
    conv = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": query},
    ]
    input_ids = tokenizer.apply_chat_template(
        conv, add_generation_prompt=True, return_tensors="pt"
    )
    device = next(model.parameters()).device
    with torch.inference_mode():
        output = model.generate(
            input_ids.to(device),
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )[0]
    response = tokenizer.decode(output[input_ids.size(1):-1])
    return response


def eval_model(args):
    model, tokenizer = load_model(args.model_name)

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions, desc=f"Chunk {args.chunk_idx}"):
            idx = line["id"]
            qa_type = line.get("qa_type", "unknown")
            image_type = line.get("image_type", "unknown")
            answer = line.get("answer", line.get("gt_ans", ""))
            question = line.get("question", "").replace("<image>", "").strip()

            image_file = line["image"]
            image_path = os.path.join(args.image_folder, image_file)

            try:
                response = run_inference_single(
                    model,
                    tokenizer,
                    image_path,
                    question,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                print(f"Error on id={idx}: {e}")
                response = f"ERROR: {str(e)}"

            ans_file.write(json.dumps({
                "id": idx,
                "qa_type": qa_type,
                "image_type": image_type,
                "question": question,
                "gt_ans": answer,
                "response": response,
            }) + "\n")
            ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="StanfordAIMI/CheXagent-2-3b",
        help="HuggingFace model name",
    )
    parser.add_argument("--image-folder", type=str, default="", help="Folder containing images")
    parser.add_argument("--question-file", type=str, default="tables/question.json", help="Path to question JSON file")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl", help="Output file path")
    parser.add_argument("--num-chunks", type=int, default=1, help="Number of chunks for parallel processing")
    parser.add_argument("--chunk-idx", type=int, default=0, help="Which chunk to process")
    parser.add_argument("--temperature", type=float, default=1.0, help="Unused for CheXagent-2 (greedy decoding)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--batch-size", type=int, default=1, help="Unused compatibility argument")

    # Kept for compatibility (ignored)
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--answer-prompter", action="store_true")

    args = parser.parse_args()
    eval_model(args)
