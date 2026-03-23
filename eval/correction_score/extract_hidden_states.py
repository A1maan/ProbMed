import argparse
import json
import math
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks."""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """Get chunk k out of n chunks."""
    chunks = split_list(lst, n)
    return chunks[k] if k < len(chunks) else []


class HiddenStateExtractor:

    TARGET_LAYER = 32  # last transformer layer

    def __init__(self, model_name, load_8bit=True):
        if load_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch.float16
            )

        self.processor = AutoProcessor.from_pretrained(model_name)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"
        self.model.eval()

        self.yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id  = self.processor.tokenizer.encode("No",  add_special_tokens=False)[0]

    @property
    def device(self):
        return self.model.device

    def get_lm_head_weights(self):
        with torch.no_grad():
            w_yes = self.model.lm_head.weight[self.yes_token_id].cpu().float().numpy()
            w_no  = self.model.lm_head.weight[self.no_token_id].cpu().float().numpy()
        return w_yes, w_no

    def format_prompt(self, question):
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True)

    def extract(self, image, question):
        prompt = self.format_prompt(question)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        hidden    = outputs.hidden_states[self.TARGET_LAYER][0, -1, :].cpu().to(torch.float16).numpy()
        logits    = outputs.logits[0, -1, :]
        yes_logit = logits[self.yes_token_id].item()
        no_logit  = logits[self.no_token_id].item()

        return hidden, yes_logit, no_logit


def build_image_lookup(test_file):
    with open(test_file) as f:
        test_data = json.load(f)
    lookup = {}
    for item in test_data:
        item_id  = item.get("id")
        question = item.get("question", "").replace("<image>", "").strip()
        if item_id is not None and "image" in item:
            lookup[(item_id, question)] = item["image"]
    return lookup


def load_questions(margin_scores_file, image_lookup):
    with open(margin_scores_file) as f:
        results = json.load(f)

    questions = []
    missing   = 0
    for r in results:
        qid        = r.get("id")
        question   = r.get("question", "")
        image_path = image_lookup.get((qid, question))
        if image_path is None:
            missing += 1
            continue
        questions.append({
            "id":         qid,
            "question":   question,
            "gt_ans":     r.get("gt_ans", ""),
            "model_pred": r.get("model_pred", ""),
            "is_correct": bool(r.get("is_correct", False)),
            "qa_type":    r.get("qa_type", ""),
            "image_type": r.get("image_type", ""),
            "image_path": image_path,
        })

    print(f"Loaded {len(questions)} questions ({missing} skipped: no image mapping)")
    return questions


def run_extraction(extractor, questions, image_folder):
    hidden_list    = []
    yes_logit_list = []
    no_logit_list  = []
    gt_label_list  = []
    image_id_list  = []
    metadata_list  = []
    skipped        = 0

    for idx, q in enumerate(tqdm(questions, desc="Extracting")):
        image_path = os.path.join(image_folder, q["image_path"])
        if not os.path.exists(image_path):
            skipped += 1
            continue
        try:
            image  = Image.open(image_path).convert("RGB")
            hidden, yes_logit, no_logit = extractor.extract(image, q["question"])
        except Exception as e:
            print(f"  Error on id={q['id']}: {e}")
            skipped += 1
            continue

        hidden_list.append(hidden)
        yes_logit_list.append(yes_logit)
        no_logit_list.append(no_logit)
        gt_label_list.append(1 if q["gt_ans"] == "yes" else 0)
        image_id_list.append(int(q["id"]))
        metadata_list.append({
            "question":   q["question"],
            "qa_type":    q["qa_type"],
            "image_type": q["image_type"],
            "model_pred": q["model_pred"],
            "is_correct": q["is_correct"],
            "image_id":   int(q["id"]),
        })

        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    print(f"Extracted {len(hidden_list)} questions ({skipped} skipped)")

    return (
        np.array(hidden_list,    dtype=np.float16),
        np.array(yes_logit_list, dtype=np.float32),
        np.array(no_logit_list,  dtype=np.float32),
        np.array(gt_label_list,  dtype=np.int8),
        np.array(image_id_list,  dtype=np.int32),
        metadata_list,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin-scores-file", required=True)
    parser.add_argument("--test-file",          required=True)
    parser.add_argument("--image-folder",       required=True)
    parser.add_argument("--output-dir",         required=True)
    parser.add_argument("--model-name", default="chaoyinshe/llava-med-v1.5-mistral-7b-hf")
    parser.add_argument("--load-8bit",  action="store_true", default=True)
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Total number of chunks (GPUs) to split work across")
    parser.add_argument("--chunk-idx",  type=int, default=0,
                        help="Index of this chunk (0-based)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_lookup = build_image_lookup(args.test_file)
    questions    = load_questions(args.margin_scores_file, image_lookup)
    if not questions:
        print("No questions found.")
        return

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    print(f"Chunk {args.chunk_idx}/{args.num_chunks}: processing {len(questions)} questions")

    extractor   = HiddenStateExtractor(model_name=args.model_name, load_8bit=args.load_8bit)
    w_yes, w_no = extractor.get_lm_head_weights()

    hidden_states, yes_logits, no_logits, gt_labels, image_ids, metadata = run_extraction(
        extractor, questions, args.image_folder
    )

    # Use chunk-specific filenames when running in multi-GPU mode
    suffix = f"-chunk{args.chunk_idx}" if args.num_chunks > 1 else ""

    cache_path = os.path.join(args.output_dir, f"hidden_states_cache{suffix}.npz")
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
    print(f"Saved cache: {cache_path}  shape={hidden_states.shape}")

    meta_path = os.path.join(args.output_dir, f"metadata{suffix}.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    # Sanity check: d@h should correlate with yes_logit - no_logit
    if len(hidden_states) >= 2:
        d    = (w_yes - w_no).astype(np.float32)
        H    = hidden_states[:200].astype(np.float32)
        corr = float(np.corrcoef(H @ d, (yes_logits - no_logits)[:200])[0, 1])
        print(f"Sanity check corr(d@h, logit_diff) = {corr:.4f}")
    else:
        print("Sanity check skipped: no samples extracted")


if __name__ == "__main__":
    main()
