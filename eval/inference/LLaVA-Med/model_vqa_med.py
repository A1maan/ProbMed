"""
Replacement for: eval/inference/LLaVA-Med/model_vqa_med.py

Modified to work with HuggingFace LLaVA-Med weights:
    chaoyinshe/llava-med-v1.5-mistral-7b-hf

Usage (same as original):
    python model_vqa_med.py \
        --model-name chaoyinshe/llava-med-v1.5-mistral-7b-hf \
        --question-file /path/to/probmed.json \
        --image-folder /path/to/images \
        --answers-file /path/to/output.jsonl
"""

import argparse
import torch
import os
import json
from tqdm import tqdm
import math

from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from PIL import Image


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_model(model_name, load_8bit=True):
    """Load HuggingFace LLaVA-Med model."""
    print(f"Loading model: {model_name}")
    
    if load_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Set pad token if not set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Set left padding for decoder-only models (important for batched generation)
    processor.tokenizer.padding_side = "left"
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, processor


def format_prompt(question, processor):
    """Format the prompt for LLaVA-Med HF model."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True
    )
    
    return prompt


def run_inference_batch(model, processor, images, questions, temperature=0.7, max_new_tokens=1024):
    """Run inference on a batch of image-question pairs."""
    
    # Format prompts
    prompts = [format_prompt(q, processor) for q in questions]
    
    # Process inputs as batch
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    
    # Decode responses
    responses = []
    input_len = inputs['input_ids'].shape[1]
    for i in range(len(questions)):
        response = processor.tokenizer.decode(
            output_ids[i][input_len:],
            skip_special_tokens=True
        ).strip()
        responses.append(response)
    
    return responses


def run_inference_single(model, processor, image, question, temperature=0.7, max_new_tokens=1024):
    """Run inference on a single image-question pair (fallback)."""
    
    prompt = format_prompt(question, processor)
    
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    
    input_len = inputs['input_ids'].shape[1]
    response = processor.tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True
    ).strip()
    
    return response


def eval_model(args):
    """Main evaluation function - matches original interface."""
    
    # Determine if using 8-bit
    use_8bit = args.load_8bit and not args.no_8bit
    
    # Load model
    model, processor = load_model(args.model_name, load_8bit=use_8bit)
    
    # Load questions
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    # Prepare output
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    batch_size = args.batch_size
    
    # Process in batches
    for batch_start in tqdm(range(0, len(questions), batch_size), desc=f"Chunk {args.chunk_idx}"):
        batch_end = min(batch_start + batch_size, len(questions))
        batch_items = questions[batch_start:batch_end]
        
        # Prepare batch data
        batch_images = []
        batch_questions = []
        batch_metadata = []
        valid_indices = []
        
        for i, line in enumerate(batch_items):
            idx = line["id"]
            qa_type = line.get("qa_type", "unknown")
            answer = line.get("answer", line.get("gt_ans", ""))
            image_type = line.get("image_type", "unknown")
            
            qs = line.get("question", "")
            qs = qs.replace('<image>', '').strip()
            
            if 'image' in line:
                image_file = line["image"]
                image_path = os.path.join(args.image_folder, image_file)
                try:
                    image = Image.open(image_path).convert('RGB')
                    batch_images.append(image)
                    batch_questions.append(qs)
                    batch_metadata.append({
                        "id": idx,
                        "qa_type": qa_type,
                        "image_type": image_type,
                        "question": qs,
                        "gt_ans": answer,
                    })
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue
        
        if not batch_images:
            continue
        
        # Run batch inference
        try:
            if len(batch_images) == 1:
                # Single item - use single inference
                responses = [run_inference_single(
                    model, processor, batch_images[0], batch_questions[0],
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens
                )]
            else:
                # Batch inference
                responses = run_inference_batch(
                    model, processor, batch_images, batch_questions,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens
                )
        except Exception as e:
            print(f"Batch inference failed, falling back to single: {e}")
            # Fallback to single inference
            responses = []
            for img, q in zip(batch_images, batch_questions):
                try:
                    resp = run_inference_single(
                        model, processor, img, q,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens
                    )
                    responses.append(resp)
                except Exception as e2:
                    print(f"Single inference error: {e2}")
                    responses.append(f"ERROR: {str(e2)}")
        
        # Write results
        for metadata, response in zip(batch_metadata, responses):
            metadata["response"] = response
            ans_file.write(json.dumps(metadata) + "\n")
            ans_file.flush()
    
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, 
                        default="chaoyinshe/llava-med-v1.5-mistral-7b-hf",
                        help="HuggingFace model name")
    parser.add_argument("--image-folder", type=str, default="",
                        help="Folder containing images")
    parser.add_argument("--question-file", type=str, default="tables/question.json",
                        help="Path to question JSON file")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl",
                        help="Output file path")
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Number of chunks for parallel processing")
    parser.add_argument("--chunk-idx", type=int, default=0,
                        help="Which chunk to process")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Maximum new tokens to generate")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--load-8bit", action="store_true", default=False,
                        help="Load model in 8-bit quantization")
    parser.add_argument("--no-8bit", action="store_true", default=False,
                        help="Disable 8-bit quantization (use fp16)")
    
    # Keep these for compatibility (ignored)
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--answer-prompter", action="store_true")
    
    args = parser.parse_args()
    
    eval_model(args)