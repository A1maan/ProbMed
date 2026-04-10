import argparse
import json
import math
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k] if k < len(chunks) else []


def load_model(model_name):
    print(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print("Model loaded successfully!")
    return model, processor


def run_inference_single(model, processor, image, question, max_new_tokens=512):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generation = generation[0][input_len:]

    return processor.decode(generation, skip_special_tokens=True)


def eval_model(args):
    model, processor = load_model(args.model_name)

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions, desc=f"Chunk {args.chunk_idx}"):
            idx       = line["id"]
            qa_type   = line.get("qa_type", "unknown")
            image_type = line.get("image_type", "unknown")
            answer    = line.get("answer", line.get("gt_ans", ""))
            question  = line.get("question", "").replace("<image>", "").strip()

            image_file = line["image"]
            image_path = os.path.join(args.image_folder, image_file)

            try:
                image = Image.open(image_path).convert("RGB")
                response = run_inference_single(
                    model, processor, image, question,
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
    parser.add_argument("--model-name", type=str, default="google/medgemma-1.5-4b-it")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--answer-prompter", action="store_true")
    args = parser.parse_args()
    eval_model(args)
