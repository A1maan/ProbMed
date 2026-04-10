import argparse
import json
import math
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k] if k < len(chunks) else []


def parse_model_pred(response):
    """Derive yes/no prediction from CheXagent response text."""
    r = response.strip()
    if "Yes" in r or r.lower() == "yes" or r.lower() == "yes.":
        return "yes"
    return "no"


class HiddenStateExtractor:

    def __init__(self, model_name):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        )
        self.model = self.model.to(torch.bfloat16)
        self.model.eval()

        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id  = self.tokenizer.encode("No",  add_special_tokens=False)[0]

        # Detect final transformer layer index
        self.target_layer = len(self.model.model.layers) - 1
        print(f"Model loaded! Target layer: {self.target_layer}, "
              f"Yes token: {self.yes_token_id}, No token: {self.no_token_id}")

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_lm_head_weights(self):
        with torch.no_grad():
            w_yes = self.model.lm_head.weight[self.yes_token_id].cpu().float().numpy()
            w_no  = self.model.lm_head.weight[self.no_token_id].cpu().float().numpy()
        return w_yes, w_no

    def extract(self, image_path, question):
        query = self.tokenizer.from_list_format([
            {"image": image_path},
            {"text": question},
        ])
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)

        hidden    = outputs.hidden_states[self.target_layer][0, -1, :].cpu().to(torch.float16).numpy()
        logits    = outputs.logits[0, -1, :]
        yes_logit = logits[self.yes_token_id].item()
        no_logit  = logits[self.no_token_id].item()

        return hidden, yes_logit, no_logit


def build_image_lookup(test_file):
    """Build (id, question) -> image_path lookup from test.json."""
    with open(test_file) as f:
        test_data = json.load(f)
    lookup = {}
    for item in test_data:
        item_id  = item.get("id")
        question = item.get("question", "").replace("<image>", "").strip()
        if item_id is not None and "image" in item:
            lookup[(item_id, question)] = item["image"]
    return lookup


def load_questions(inference_file, image_folder, test_file):
    """Load questions from chexagent.json, joining with test.json for image paths."""
    image_lookup = build_image_lookup(test_file)

    with open(inference_file) as f:
        data = json.load(f)

    questions = []
    missing = 0
    for r in data:
        gt_ans = r.get("gt_ans", "").lower().strip()
        if gt_ans not in ("yes", "no"):
            continue

        qid      = r.get("id")
        question = r.get("question", "").replace("<image>", "").strip()

        image_file = image_lookup.get((qid, question))
        if image_file is None:
            missing += 1
            continue

        response   = r.get("response", "")
        model_pred = parse_model_pred(response)
        is_correct = model_pred == gt_ans

        questions.append({
            "id":         qid,
            "question":   question,
            "gt_ans":     gt_ans,
            "model_pred": model_pred,
            "is_correct": is_correct,
            "qa_type":    r.get("qa_type", ""),
            "image_type": r.get("image_type", ""),
            "image_path": os.path.join(image_folder, image_file),
        })

    print(f"Loaded {len(questions)} yes/no questions ({missing} skipped: no image mapping)")
    return questions


def run_extraction(extractor, questions):
    hidden_list    = []
    yes_logit_list = []
    no_logit_list  = []
    gt_label_list  = []
    image_id_list  = []
    metadata_list  = []
    skipped        = 0

    for idx, q in enumerate(tqdm(questions, desc="Extracting")):
        image_path = q["image_path"]
        if not os.path.exists(image_path):
            skipped += 1
            continue
        try:
            hidden, yes_logit, no_logit = extractor.extract(image_path, q["question"])
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
    parser.add_argument("--inference-file", required=True,
                        help="Path to chexagent.json inference output")
    parser.add_argument("--test-file",      required=True,
                        help="Path to test.json (used to look up image paths)")
    parser.add_argument("--image-folder",   required=True)
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--model-name", default="StanfordAIMI/CheXagent-2-3b")
    parser.add_argument("--load-8bit",  action="store_true", default=False,
                        help="Unused for CheXagent (bfloat16 used instead)")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx",  type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    questions = load_questions(args.inference_file, args.image_folder, args.test_file)
    if not questions:
        print("No questions found.")
        return

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    print(f"Chunk {args.chunk_idx}/{args.num_chunks}: processing {len(questions)} questions")

    extractor   = HiddenStateExtractor(model_name=args.model_name)
    w_yes, w_no = extractor.get_lm_head_weights()

    hidden_states, yes_logits, no_logits, gt_labels, image_ids, metadata = run_extraction(
        extractor, questions
    )

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

    if len(hidden_states) >= 2:
        d    = (w_yes - w_no).astype(np.float32)
        H    = hidden_states[:200].astype(np.float32)
        corr = float(np.corrcoef(H @ d, (yes_logits - no_logits)[:200])[0, 1])
        print(f"Sanity check corr(d@h, logit_diff) = {corr:.4f}")
    else:
        print("Sanity check skipped: no samples extracted")


if __name__ == "__main__":
    main()
