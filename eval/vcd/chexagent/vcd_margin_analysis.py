import argparse
import json
import math
import os
import random
import tempfile

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k] if k < len(chunks) else []


class VCDMarginAnalyzer:
    """Computes VCD margin scores for hallucination detection using CheXagent-2."""

    def __init__(self, model_name="StanfordAIMI/CheXagent-2-3b"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        )
        self.model = self.model.to(torch.bfloat16)
        self.model.eval()

        self.yes_token_id = self._get_token_id("Yes")
        self.no_token_id = self._get_token_id("No")
        print(f"Model loaded! Yes token: {self.yes_token_id}, No token: {self.no_token_id}")

    def _get_token_id(self, word):
        tokens = self.tokenizer.encode(word, add_special_tokens=False)
        return tokens[0]

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _build_inputs(self, image_path, question):
        """Build input_ids for a given image path and question."""
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
        )
        return input_ids.to(self.device)

    def get_yes_no_logits(self, input_ids):
        """Get Yes/No logits from the last token position."""
        with torch.inference_mode():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            log_probs = F.log_softmax(logits[0], dim=-1)
            return {
                "yes_logit": logits[0, self.yes_token_id].item(),
                "no_logit": logits[0, self.no_token_id].item(),
                "log_p_yes": log_probs[self.yes_token_id].item(),
                "log_p_no": log_probs[self.no_token_id].item(),
            }

    def downsample_upsample(self, image, scale=0.5):
        """Degrade a PIL image by downsampling then upsampling."""
        w, h = image.size
        small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
        degraded = image.resize((small_w, small_h), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        return degraded

    def compute_margin_score(self, image_path, question, downsample_scale=0.5):
        """
        Compute VCD margin score.
        g = [log p(Yes|v,q) - log p(Yes|v',q)] - [log p(No|v,q) - log p(No|v',q)]
        """
        # Clean image forward pass
        input_ids_clean = self._build_inputs(image_path, question)
        logits_clean = self.get_yes_no_logits(input_ids_clean)

        # Degrade image and save to temp file (tokenizer needs a file path)
        image = Image.open(image_path).convert("RGB")
        degraded = self.downsample_upsample(image, scale=downsample_scale)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            degraded.save(tmp_path)

        try:
            input_ids_degraded = self._build_inputs(tmp_path, question)
            logits_degraded = self.get_yes_no_logits(input_ids_degraded)
        finally:
            os.unlink(tmp_path)

        yes_diff = logits_clean["log_p_yes"] - logits_degraded["log_p_yes"]
        no_diff = logits_clean["log_p_no"] - logits_degraded["log_p_no"]
        margin_g = yes_diff - no_diff

        return {
            "margin_g": margin_g,
            "yes_diff": yes_diff,
            "no_diff": no_diff,
            "log_p_yes_clean": logits_clean["log_p_yes"],
            "log_p_no_clean": logits_clean["log_p_no"],
            "log_p_yes_degraded": logits_degraded["log_p_yes"],
            "log_p_no_degraded": logits_degraded["log_p_no"],
            "yes_logit_clean": logits_clean["yes_logit"],
            "no_logit_clean": logits_clean["no_logit"],
        }


def filter_yes_no_questions(data):
    filtered = []
    for item in data:
        gt_ans = item.get("answer", item.get("gt_ans", "")).lower().strip()
        if gt_ans in ["yes", "no"]:
            filtered.append(item)
    return filtered


def run_analysis(args):
    with open(args.question_file, "r") as f:
        data = json.load(f)

    data = filter_yes_no_questions(data)
    print(f"Found {len(data)} yes/no questions")

    if args.sample_ratio < 1.0:
        sample_size = int(len(data) * args.sample_ratio)
        random.seed(args.seed)
        data = random.sample(data, sample_size)
        print(f"Sampled {len(data)} questions ({args.sample_ratio * 100:.0f}%)")

    data = get_chunk(data, args.num_chunks, args.chunk_idx)
    print(f"Chunk {args.chunk_idx}/{args.num_chunks}: processing {len(data)} questions")

    if not data:
        print("No data in this chunk, exiting.")
        return []

    analyzer = VCDMarginAnalyzer(model_name=args.model_name)

    results = []
    for item in tqdm(data, desc="Computing margin scores"):
        image_path = os.path.join(args.image_folder, item["image"])
        question = item.get("question", "").replace("<image>", "").strip()
        gt_ans = item.get("answer", item.get("gt_ans", "")).lower().strip()

        try:
            margin_result = analyzer.compute_margin_score(
                image_path, question, downsample_scale=args.downsample_scale
            )
            model_pred = "yes" if margin_result["yes_logit_clean"] > margin_result["no_logit_clean"] else "no"
            is_correct = model_pred == gt_ans

            results.append({
                "id": item.get("id"),
                "question": question,
                "gt_ans": gt_ans,
                "model_pred": model_pred,
                "is_correct": is_correct,
                "qa_type": item.get("qa_type", "unknown"),
                "image_type": item.get("image_type", "unknown"),
                "margin_g": margin_result["margin_g"],
                "yes_diff": margin_result["yes_diff"],
                "no_diff": margin_result["no_diff"],
                "log_p_yes_clean": margin_result["log_p_yes_clean"],
                "log_p_no_clean": margin_result["log_p_no_clean"],
                "log_p_yes_degraded": margin_result["log_p_yes_degraded"],
                "log_p_no_degraded": margin_result["log_p_no_degraded"],
            })
        except Exception as e:
            print(f"Error processing {item.get('id')}: {e}")
            continue

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to: {args.output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="VCD Margin Score Analysis for CheXagent-2")
    parser.add_argument("--model-name", type=str, default="StanfordAIMI/CheXagent-2-3b")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="results/chexagent/margin_scores.json")
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--downsample-scale", type=float, default=0.5)
    parser.add_argument("--load-8bit", action="store_true", default=False,
                        help="Unused for CheXagent (bfloat16 used instead)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    run_analysis(args)


if __name__ == "__main__":
    main()
