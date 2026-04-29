import argparse
import json
import math
import os
import tempfile
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


MODEL_DEFAULTS = {
    "llavamed": {
        "model_name": "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        "probe_dir": "../attention-analysis/results/representation_probing/image_mode_real",
        "layer": 15,
    },
    "chexagent": {
        "model_name": "StanfordAIMI/CheXagent-2-3b",
        "probe_dir": "../attention-analysis/results/representation_probing/chexagent",
        "layer": None,
    },
    "medgemma": {
        "model_name": "google/medgemma-1.5-4b-it",
        "probe_dir": "../attention-analysis/results/representation_probing/medgemma",
        "layer": None,
    },
}


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k] if k < len(chunks) else []


def load_json_or_jsonl(path):
    with open(path) as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def parse_yes_no(response):
    response = response.strip()
    if "Yes" in response or response.lower() in ("yes", "yes."):
        return "yes"
    return "no"


def build_image_lookup(test_file):
    with open(test_file) as f:
        test_data = json.load(f)
    lookup = {}
    for item in test_data:
        item_id = item.get("id")
        question = item.get("question", "").replace("<image>", "").strip()
        if item_id is not None and "image" in item:
            lookup[(item_id, question)] = item["image"]
    return lookup


def load_questions(results_file, test_file, image_folder, limit=None):
    image_lookup = build_image_lookup(test_file)
    rows = load_json_or_jsonl(results_file)
    questions = []
    missing = 0

    for row in rows:
        gt_ans = row.get("gt_ans", row.get("answer", "")).lower().strip()
        if gt_ans not in ("yes", "no"):
            continue

        qid = row.get("id")
        question = row.get("question", "").replace("<image>", "").strip()
        image_file = row.get("image") or image_lookup.get((qid, question))
        if not image_file:
            missing += 1
            continue

        response = row.get("response", "")
        model_pred = row["model_pred"] if row.get("model_pred") in ("yes", "no") else parse_yes_no(response)

        questions.append({
            "id": qid,
            "question": question,
            "gt_ans": gt_ans,
            "gt_label": 1 if gt_ans == "yes" else 0,
            "qa_type": row.get("qa_type", ""),
            "image_type": row.get("image_type", ""),
            "image_path": os.path.join(image_folder, image_file),
            "model_pred": model_pred,
            "is_correct": bool(row.get("is_correct", model_pred == gt_ans)),
        })

        if limit is not None and len(questions) >= limit:
            break

    print(f"Loaded {len(questions)} yes/no questions ({missing} skipped: no image mapping)")
    return questions


def load_probe(probe_dir, layer_arg=None):
    analysis_path = os.path.join(probe_dir, "analysis_summary.json")
    weights_path = os.path.join(probe_dir, "lr_weights.npz")

    if layer_arg is None:
        with open(analysis_path) as f:
            layer = int(json.load(f)["best_layer"])
    else:
        layer = int(layer_arg)

    data = np.load(weights_path, allow_pickle=True)
    weights = data["weights"]
    intercepts = data["intercepts"]
    if layer >= len(weights):
        raise ValueError(f"Layer {layer} requested, but {weights_path} has {len(weights)} weights")

    direction = weights[layer].astype(np.float32)
    intercept = float(intercepts[layer])
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    print(f"Using probe layer {layer} from {weights_path}; direction shape={direction.shape}")
    return layer, direction, intercept


def load_trained_direction(direction_file, layer):
    direction = np.load(direction_file).astype(np.float32)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    print(f"Loaded trained LR direction from {direction_file}  shape={direction.shape}")
    return int(layer), direction, 0.0


def replace_last_token(output, direction, strength, target_label=None, use_probe_sign=False, intercept=0.0):
    hidden = output[0] if isinstance(output, tuple) else output
    corrected = hidden.clone()
    last = corrected[:, -1, :]
    direction = direction.to(device=last.device, dtype=last.dtype)

    if target_label is not None:
        sign = 1.0 if int(target_label) == 1 else -1.0
    elif use_probe_sign:
        score = (last.float() @ direction.float()) + float(intercept)
        sign = torch.where(score >= 0, torch.ones_like(score), -torch.ones_like(score)).to(last.dtype)
        sign = sign.view(-1, 1)
    else:
        sign = 1.0

    corrected[:, -1, :] = last + float(strength) * sign * direction
    if isinstance(output, tuple):
        return (corrected,) + output[1:]
    return corrected


class BaseInterventionRunner:
    def __init__(self):
        self._temp_paths = set()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def cleanup(self, prepared_image):
        if isinstance(prepared_image, str) and prepared_image in self._temp_paths:
            try:
                os.remove(prepared_image)
            except OSError:
                pass
            self._temp_paths.discard(prepared_image)

    def prepare_image(self, image_path, image_mode):
        if image_mode == "real":
            return image_path if self.image_path_inputs else Image.open(image_path).convert("RGB")

        image = Image.open(image_path).convert("RGB")
        if image_mode == "black":
            image = Image.new("RGB", image.size, (0, 0, 0))
        elif image_mode == "random":
            arr = np.random.randint(0, 256, (image.size[1], image.size[0], 3), dtype=np.uint8)
            image = Image.fromarray(arr)

        if self.image_path_inputs:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            image.save(tmp.name)
            self._temp_paths.add(tmp.name)
            return tmp.name
        return image

    def register_hook(self, layer, direction, strength, mode, target_label=None, intercept=0.0):
        if layer <= 0:
            raise ValueError("Hook intervention expects layer >= 1 because layer 0 is embeddings")
        module = self.layers[layer - 1]
        direction_t = torch.tensor(direction, dtype=torch.float32)
        use_probe_sign = mode == "probe"
        oracle_label = target_label if mode == "oracle" else None

        def hook(_module, _inputs, output):
            return replace_last_token(
                output,
                direction_t,
                strength,
                target_label=oracle_label,
                use_probe_sign=use_probe_sign,
                intercept=intercept,
            )

        return module.register_forward_hook(hook)

    def yes_no_from_logits(self, logits):
        yes_logit = logits[0, self.yes_token_id].item()
        no_logit = logits[0, self.no_token_id].item()
        pred = "yes" if yes_logit > no_logit else "no"
        return yes_logit, no_logit, pred

    def run_one(self, prepared_image, question, layer, direction, strength, mode, target_label, intercept):
        with torch.inference_mode():
            baseline_logits = self.forward_logits(prepared_image, question)

        handle = self.register_hook(layer, direction, strength, mode, target_label, intercept)
        try:
            with torch.inference_mode():
                corrected_logits = self.forward_logits(prepared_image, question)
        finally:
            handle.remove()

        by, bn, bp = self.yes_no_from_logits(baseline_logits)
        cy, cn, cp = self.yes_no_from_logits(corrected_logits)
        return {
            "baseline_yes_logit": by,
            "baseline_no_logit": bn,
            "baseline_margin": by - bn,
            "baseline_pred": bp,
            "corrected_yes_logit": cy,
            "corrected_no_logit": cn,
            "corrected_margin": cy - cn,
            "corrected_pred": cp,
            "delta_margin": (cy - cn) - (by - bn),
        }


class LlavaMedRunner(BaseInterventionRunner):
    image_path_inputs = False

    def __init__(self, model_name, load_8bit):
        super().__init__()
        print(f"Loading LLaVA-Med: {model_name}")
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
        self.layers = self.model.model.language_model.layers
        self.yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]
        print(f"Loaded {len(self.layers)} language layers")

    def format_prompt(self, question):
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True)

    def forward_logits(self, image, question):
        prompt = self.format_prompt(question)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        return self.model(**inputs, return_dict=True).logits[:, -1, :]


class CheXagentRunner(BaseInterventionRunner):
    image_path_inputs = True

    def __init__(self, model_name, load_8bit):
        super().__init__()
        if load_8bit:
            print("Note: --load-8bit is ignored for CheXagent; loading in bfloat16.")
        print(f"Loading CheXagent: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        )
        self.model = self.model.to(torch.bfloat16)
        self.model.eval()
        self.layers = self.model.model.layers
        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        print(f"Loaded {len(self.layers)} language layers")

    def forward_logits(self, image_path, question):
        query = self.tokenizer.from_list_format([{"image": image_path}, {"text": question}])
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        return self.model(input_ids, return_dict=True).logits[:, -1, :]


class MedGemmaRunner(BaseInterventionRunner):
    image_path_inputs = False

    def __init__(self, model_name, load_8bit):
        super().__init__()
        if AutoModelForImageTextToText is None:
            raise ImportError(
                "AutoModelForImageTextToText not available in this transformers version. "
                "Upgrade to transformers>=4.45 to use MedGemma."
            )
        if load_8bit:
            print("Note: --load-8bit is ignored for MedGemma; loading in bfloat16.")
        print(f"Loading MedGemma: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()
        self.layers = self.model.model.language_model.layers
        self.yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]
        print(f"Loaded {len(self.layers)} language layers")

    def forward_logits(self, image, question):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)
        return self.model(**inputs, return_dict=True).logits[:, -1, :]


def build_runner(model_key, model_name, load_8bit):
    if model_key == "llavamed":
        return LlavaMedRunner(model_name, load_8bit)
    if model_key == "chexagent":
        return CheXagentRunner(model_name, load_8bit)
    if model_key == "medgemma":
        return MedGemmaRunner(model_name, load_8bit)
    raise ValueError(f"Unknown model: {model_key}")


def paired_metrics(records, pred_key):
    correct = {}
    by_image = defaultdict(list)
    for r in records:
        image_id = r["id"]
        qa_type = r.get("qa_type", "")
        correct[(image_id, qa_type)] = int(r[pred_key] == r["gt_ans"])
        by_image[image_id].append(qa_type)

    pairs = []
    for image_id, qa_types in by_image.items():
        qa_set = set(qa_types)
        if "modality_gt" in qa_set and "modality_hallu" in qa_set:
            pairs.append((correct[(image_id, "modality_gt")], correct[(image_id, "modality_hallu")]))
        if "body_part_gt" in qa_set and "body_part_hallu" in qa_set:
            pairs.append((correct[(image_id, "body_part_gt")], correct[(image_id, "body_part_hallu")]))
        for qt in qa_types:
            if qt.startswith("entity_gt_"):
                suffix = qt[len("entity_gt_"):]
                hallu = f"entity_hallu_{suffix}"
                if hallu in qa_set:
                    pairs.append((correct[(image_id, qt)], correct[(image_id, hallu)]))
            if qt.startswith("grounding_gt_"):
                suffix = qt[len("grounding_gt_"):]
                hallu = f"grounding_hallu_{suffix}"
                if hallu in qa_set:
                    pairs.append((correct[(image_id, qt)], correct[(image_id, hallu)]))

    if not pairs:
        return float("nan"), float("nan"), 0
    strict = sum(1 for gt_ok, hallu_ok in pairs if gt_ok and hallu_ok) / len(pairs)
    gt_only = sum(1 for gt_ok, _ in pairs if gt_ok) / len(pairs)
    return float(strict), float(gt_only), len(pairs)


def summarize(records):
    y_true = np.array([r["gt_label"] for r in records])

    def pred_array(key):
        return np.array([1 if r[key] == "yes" else 0 for r in records])

    summary = {}
    for name, key in [("baseline", "baseline_pred"), ("corrected", "corrected_pred")]:
        y_pred = pred_array(key)
        adv_mask = np.array(["hallu" in r.get("qa_type", "") for r in records])
        strict, gt_only, n_pairs = paired_metrics(records, key)
        summary[name] = {
            "overall": float(accuracy_score(y_true, y_pred)),
            "gt_yes": float(accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])) if (y_true == 1).sum() else float("nan"),
            "gt_no": float(accuracy_score(y_true[y_true == 0], y_pred[y_true == 0])) if (y_true == 0).sum() else float("nan"),
            "adversarial": float(accuracy_score(y_true[adv_mask], y_pred[adv_mask])) if adv_mask.sum() else float("nan"),
            "adv_paired": strict,
            "adv_wo_pair": gt_only,
            "n_pairs": n_pairs,
        }
    return summary


def main(
    default_model=None,
    default_results_file=None,
    default_test_file=None,
    default_image_folder=None,
    default_load_8bit=False,
):
    parser = argparse.ArgumentParser(description="Apply LR-probe hidden-state correction at an intermediate layer")
    parser.add_argument("--model", choices=sorted(MODEL_DEFAULTS), default=default_model, required=default_model is None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--results-file", default=default_results_file,
                        required=default_results_file is None,
                        help="Model response or margin-score JSON/JSONL")
    parser.add_argument("--test-file", default=default_test_file, required=default_test_file is None)
    parser.add_argument("--image-folder", default=default_image_folder, required=default_image_folder is None)
    parser.add_argument("--probe-dir", default=None,
                        help="Probing results dir (lr_weights.npz). Used when --trained-direction is absent.")
    parser.add_argument("--trained-direction", default=None,
                        help="Path to trained_lr_direction.npy from train_intermediate_correction.py. "
                             "When provided, --probe-dir is ignored.")
    parser.add_argument("--test-image-ids", default=None,
                        help="Path to test_image_ids.json from train_intermediate_correction.py. "
                             "When provided, only questions from the held-out test split are evaluated.")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--mode", choices=["probe", "oracle", "positive"], default="probe",
                        help="probe: push along LR-predicted sign; oracle: push toward GT sign; positive: always +direction")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--load-8bit", action="store_true", default=default_load_8bit)
    parser.add_argument("--image-mode", choices=["real", "black", "random"], default="real")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed + args.chunk_idx)

    defaults = MODEL_DEFAULTS[args.model]
    model_name = args.model_name or defaults["model_name"]
    layer_arg = args.layer if args.layer is not None else defaults["layer"]

    if args.trained_direction is not None:
        if layer_arg is None:
            raise ValueError("--layer is required when using --trained-direction")
        layer, direction, intercept = load_trained_direction(args.trained_direction, layer_arg)
        direction_label = "trained_lr"
    else:
        probe_dir = args.probe_dir or defaults["probe_dir"]
        layer, direction, intercept = load_probe(probe_dir, layer_arg)
        direction_label = "probe_lr"

    output_dir = args.output_dir or os.path.join(
        args.model,
        "results",
        f"layer_{layer}",
        f"{direction_label}_{args.mode}_strength_{args.strength:g}",
    )
    os.makedirs(output_dir, exist_ok=True)

    questions = load_questions(args.results_file, args.test_file, args.image_folder, limit=args.limit)

    if args.test_image_ids is not None:
        with open(args.test_image_ids) as f:
            test_ids = set(json.load(f))
        before = len(questions)
        questions = [q for q in questions if q["id"] in test_ids]
        print(f"Filtered to test split: {before} → {len(questions)} questions")
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    print(f"Chunk {args.chunk_idx}/{args.num_chunks}: evaluating {len(questions)} questions")

    runner = build_runner(args.model, model_name, args.load_8bit)
    if layer > len(runner.layers):
        raise ValueError(f"Layer {layer} is outside model with {len(runner.layers)} transformer layers")

    records = []
    for q in tqdm(questions, desc="Intervening"):
        if not os.path.exists(q["image_path"]):
            print(f"Missing image, skipping: {q['image_path']}")
            continue
        prepared = None
        try:
            prepared = runner.prepare_image(q["image_path"], args.image_mode)
            result = runner.run_one(
                prepared,
                q["question"],
                layer=layer,
                direction=direction,
                strength=args.strength,
                mode=args.mode,
                target_label=q["gt_label"],
                intercept=intercept,
            )
        except Exception as exc:
            print(f"Error on id={q['id']}: {exc}")
            continue
        finally:
            if prepared is not None:
                runner.cleanup(prepared)

        records.append({**q, **result})
        if len(records) % 50 == 0:
            torch.cuda.empty_cache()

    suffix = f"-chunk{args.chunk_idx}" if args.num_chunks > 1 else ""
    records_path = os.path.join(output_dir, f"intervention_results{suffix}.jsonl")
    with open(records_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    summary = {
        "model": args.model,
        "model_name": model_name,
        "direction_source": args.trained_direction or probe_dir,
        "direction_label": direction_label,
        "layer": layer,
        "mode": args.mode,
        "strength": args.strength,
        "image_mode": args.image_mode,
        "num_records": len(records),
        "metrics": summarize(records) if records else {},
    }
    summary_path = os.path.join(output_dir, f"summary{suffix}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved records: {records_path}")
    print(f"Saved summary: {summary_path}")
    if records:
        print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
