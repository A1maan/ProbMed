"""
Token-by-Token Attention Maps for Yes/No Tokens
================================================

Unlike attention_heatmaps.py which reads the last PROMPT token's attention
during the prefill phase, this script captures the attention of the actually
GENERATED token (Yes or No) attending to image patches.

Approach:
  1. Run a greedy forward pass to determine the model's first generated token.
  2. Append that token (Yes or No) to the input sequence.
  3. Run a second forward pass over the extended sequence.
  4. Read the attention of the appended token (last position) to image patches.

This gives the true per-token attention map at generation time.

Usage:
    python token_attention_maps.py \
        --margin-scores-file ../vcd/results/vcd_analysis/margin_scores.json \
        --test-file /workspace/ProbMed-Dataset/test/test.json \
        --image-folder /workspace/ProbMed-Dataset/test/ \
        --output-dir results/token_attention_maps \
        --num-pairs 50
"""

import argparse
import json
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


class TokenAttentionExtractor:
    """
    Extracts attention maps for the generated Yes/No token from LLaVA-Med.

    Strategy
    --------
    Two forward passes per question:
      Pass 1  – prompt only  →  reads logits to pick Yes/No
      Pass 2  – prompt + predicted token  →  reads attention at the token position
    """

    def __init__(self, model_name="chaoyinshe/llava-med-v1.5-mistral-7b-hf", load_8bit=True):
        print(f"Loading model: {model_name}")

        if load_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                attn_implementation="eager",  # required for output_attentions=True
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                attn_implementation="eager",
            )

        self.processor = AutoProcessor.from_pretrained(model_name)

        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"

        self.model.eval()

        # Model constants
        self.num_layers = self.model.config.text_config.num_hidden_layers
        self.num_heads  = self.model.config.text_config.num_attention_heads

        # LLaVA uses a 24×24 = 576 image patch grid
        self.image_grid_size  = 24
        self.num_image_tokens = self.image_grid_size ** 2

        # Pre-compute Yes/No token ids (single-token encoding)
        self.yes_token_id = self.processor.tokenizer.encode(
            "Yes", add_special_tokens=False
        )[0]
        self.no_token_id = self.processor.tokenizer.encode(
            "No", add_special_tokens=False
        )[0]

        print(f"Model loaded. Layers: {self.num_layers}, Heads: {self.num_heads}")
        print(f"Image grid: {self.image_grid_size}×{self.image_grid_size} = {self.num_image_tokens} tokens")
        print(f"Yes token id: {self.yes_token_id}, No token id: {self.no_token_id}")

    @property
    def device(self):
        return self.model.device

    def _build_inputs(self, image, question):
        """Tokenize prompt + image and return input dict on the correct device."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=True)
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def _find_image_token_range(self, input_ids):
        """
        Return (img_start, img_end) — the slice of input_ids that will be
        replaced by the 576 image patch embeddings inside LLaVA's forward pass.

        The processor inserts a single <image> placeholder token; the model's
        multi-modal projector expands it to num_image_tokens patches in-place,
        so the slice [img_start : img_start + num_image_tokens] addresses all
        image patch positions in the resulting hidden-state sequence.
        """
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
        positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            raise ValueError("No <image> token found in input_ids.")
        img_start = positions[0].item()
        img_end   = img_start + self.num_image_tokens
        return img_start, img_end

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_token_attention_maps(self, image, question, layers_to_extract=None):
        """
        Extract per-layer attention maps for the generated Yes/No token.

        Parameters
        ----------
        image            : PIL.Image
        question         : str
        layers_to_extract: list[int] | None  (default [0, 8, 16, 24, 31])

        Returns
        -------
        attention_maps : dict[int, np.ndarray]  layer_idx -> (24, 24) float32 array
        prediction     : str   "yes" | "no"
        confidence     : float softmax probability of the chosen token
        """
        if layers_to_extract is None:
            layers_to_extract = [0, 8, 16, 24, self.num_layers - 1]

        # ---- Pass 1: predict Yes or No --------------------------------
        inputs = self._build_inputs(image, question)
        with torch.inference_mode():
            out1 = self.model(**inputs, return_dict=True)

        logits = out1.logits[:, -1, :]   # (1, vocab_size) — last prompt position
        yes_logit = logits[0, self.yes_token_id].item()
        no_logit  = logits[0, self.no_token_id].item()

        prediction     = "yes" if yes_logit > no_logit else "no"
        chosen_token_id = self.yes_token_id if prediction == "yes" else self.no_token_id
        confidence = torch.softmax(
            torch.tensor([yes_logit, no_logit]), dim=0
        )[0 if prediction == "yes" else 1].item()

        # ---- Pass 2: append predicted token, extract its attention ----
        # input_ids shape: (1, prompt_len)  →  (1, prompt_len + 1) after append
        appended_ids = torch.cat(
            [
                inputs["input_ids"],
                torch.tensor([[chosen_token_id]], device=self.device),
            ],
            dim=1,
        )

        # pixel_values and attention_mask need to be updated too
        extended_inputs = dict(inputs)
        extended_inputs["input_ids"] = appended_ids

        if "attention_mask" in inputs:
            extended_inputs["attention_mask"] = torch.cat(
                [
                    inputs["attention_mask"],
                    torch.ones((1, 1), dtype=inputs["attention_mask"].dtype, device=self.device),
                ],
                dim=1,
            )

        # Locate image patch positions in the EXTENDED sequence.
        # The image token is still at the same absolute position;
        # the appended token comes after all prompt tokens.
        img_start, img_end = self._find_image_token_range(appended_ids)

        with torch.inference_mode():
            out2 = self.model(
                **extended_inputs,
                output_attentions=True,
                return_dict=True,
            )

        # Extract attention maps
        attention_maps = {}
        for layer_idx in layers_to_extract:
            if layer_idx >= len(out2.attentions):
                continue

            # attentions[layer]: (batch=1, heads, seq_len, seq_len)
            attn = out2.attentions[layer_idx][0]   # (heads, seq_len, seq_len)

            # Average across all attention heads
            attn_avg = attn.mean(dim=0)            # (seq_len, seq_len)

            # Row -1 = the appended Yes/No token attending to all prior tokens.
            # Columns [img_start : img_end] = image patch positions.
            token_to_img = attn_avg[-1, img_start:img_end]  # (num_image_tokens,)

            attn_2d = token_to_img.reshape(self.image_grid_size, self.image_grid_size)
            attention_maps[layer_idx] = attn_2d.cpu().float().numpy()

        return attention_maps, prediction, confidence

    def get_both_token_attention_maps(self, image, question, layers_to_extract=None):
        """
        Run two extra forward passes: one appending "Yes", one appending "No".
        Returns attention maps for BOTH tokens regardless of which was predicted.

        Returns
        -------
        yes_maps   : dict[int, np.ndarray]
        no_maps    : dict[int, np.ndarray]
        prediction : str "yes" | "no"
        confidence : float
        """
        if layers_to_extract is None:
            layers_to_extract = [0, 8, 16, 24, self.num_layers - 1]

        # ---- Pass 1: get prediction -----------------------------------
        inputs = self._build_inputs(image, question)
        with torch.inference_mode():
            out1 = self.model(**inputs, return_dict=True)

        logits      = out1.logits[:, -1, :]
        yes_logit   = logits[0, self.yes_token_id].item()
        no_logit    = logits[0, self.no_token_id].item()
        prediction  = "yes" if yes_logit > no_logit else "no"
        confidence  = torch.softmax(
            torch.tensor([yes_logit, no_logit]), dim=0
        )[0 if prediction == "yes" else 1].item()

        img_start, img_end = self._find_image_token_range(inputs["input_ids"])

        def _extract(token_id):
            ext = dict(inputs)
            ext["input_ids"] = torch.cat(
                [inputs["input_ids"], torch.tensor([[token_id]], device=self.device)],
                dim=1,
            )
            if "attention_mask" in inputs:
                ext["attention_mask"] = torch.cat(
                    [
                        inputs["attention_mask"],
                        torch.ones((1, 1), dtype=inputs["attention_mask"].dtype, device=self.device),
                    ],
                    dim=1,
                )
            with torch.inference_mode():
                out = self.model(**ext, output_attentions=True, return_dict=True)

            maps = {}
            for layer_idx in layers_to_extract:
                if layer_idx >= len(out.attentions):
                    continue
                attn_avg = out.attentions[layer_idx][0].mean(dim=0)
                token_to_img = attn_avg[-1, img_start:img_end]
                maps[layer_idx] = (
                    token_to_img
                    .reshape(self.image_grid_size, self.image_grid_size)
                    .cpu().float().numpy()
                )
            return maps

        yes_maps = _extract(self.yes_token_id)
        no_maps  = _extract(self.no_token_id)

        return yes_maps, no_maps, prediction, confidence


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def find_paired_questions(margin_scores_file, test_file, num_pairs=50):
    """Find pairs of questions on the same image (one correct, one wrong)."""
    print(f"Loading margin scores from: {margin_scores_file}")
    with open(margin_scores_file) as f:
        results = json.load(f)

    print(f"Loading test data from: {test_file}")
    with open(test_file) as f:
        test_data = json.load(f)

    id_q_to_image = {}
    for item in test_data:
        item_id = item.get("id")
        question = item.get("question", "").replace("<image>", "").strip()
        if item_id is not None and "image" in item:
            id_q_to_image[(item_id, question)] = item["image"]

    by_image = defaultdict(list)
    for r in results:
        key = (r.get("id"), r.get("question", ""))
        img_path = id_q_to_image.get(key)
        if img_path:
            r["image"] = img_path
            by_image[img_path].append(r)

    print(f"Unique images: {len(by_image)}")

    pairs = []
    for img_path, qs in by_image.items():
        correct = [q for q in qs if q.get("is_correct", False)]
        wrong   = [q for q in qs if not q.get("is_correct", False)]
        if correct and wrong:
            pairs.append({"image_path": img_path, "correct": correct[0], "wrong": wrong[0]})

    print(f"Found {len(pairs)} valid (correct, wrong) pairs")

    if len(pairs) > num_pairs:
        random.seed(42)
        pairs = random.sample(pairs, num_pairs)
        print(f"Sampled {num_pairs} pairs")

    return pairs


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _resize_attn(attn_2d, target_hw):
    """Resize a float32 2-D attention array to (H, W) using bilinear interpolation."""
    h, w = target_hw
    return np.array(
        Image.fromarray(attn_2d).resize((w, h), Image.BILINEAR)
    )


def visualize_yes_no_comparison(
    image,
    yes_maps,
    no_maps,
    prediction,
    confidence,
    question,
    gt_ans,
    layers,
    output_path,
):
    """
    Grid layout:
      Col 0  = "Yes" token attention per layer
      Col 1  = "No"  token attention per layer
      Row 0  = original image (both cols)
      Row i+1 = layer layers[i]

    The predicted token column is highlighted with a green border.
    """
    num_rows = len(layers) + 1
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows))

    img_np = np.array(image)

    # --- Row 0: image + question info ---
    token_labels = ["YES", "NO"]
    for col in range(2):
        token  = token_labels[col]
        is_predicted = (col == 0 and prediction == "yes") or (col == 1 and prediction == "no")
        pred_note = f"← model chose this  ({confidence:.0%} confident)" if is_predicted else "← not chosen"
        color = "green" if is_predicted else "#888888"

        axes[0, col].imshow(img_np)
        axes[0, col].set_title(
            f"Where does the '{token}' token look on the image?\n"
            f"{pred_note}\n"
            f"Question: {question[:55]}{'…' if len(question) > 55 else ''}\n"
            f"Ground truth answer: {gt_ans}",
            fontsize=8, color=color,
        )
        axes[0, col].axis("off")

        # Green border on the predicted column so it stands out
        if is_predicted:
            for spine in axes[0, col].spines.values():
                spine.set_edgecolor("green")
                spine.set_linewidth(3)

    # --- Rows 1+: per-layer heatmaps ---
    hw = (img_np.shape[0], img_np.shape[1])
    for i, layer_idx in enumerate(layers):
        row = i + 1
        for col, maps in enumerate([yes_maps, no_maps]):
            attn = maps.get(layer_idx)
            if attn is None:
                axes[row, col].axis("off")
                continue

            attn_resized = _resize_attn(attn, hw)

            axes[row, col].imshow(img_np)
            axes[row, col].imshow(
                attn_resized, cmap="hot", alpha=0.6,
                vmin=0, vmax=attn_resized.max() or 1e-6,
            )
            axes[row, col].set_title(
                f"Transformer layer {layer_idx}\n"
                f"(brighter = more attention to that image region)",
                fontsize=8,
            )
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_correct_vs_wrong(
    image,
    yes_maps_correct,
    no_maps_correct,
    yes_maps_wrong,
    no_maps_wrong,
    info_correct,
    info_wrong,
    layers,
    output_path,
):
    """
    Extended grid showing Yes/No token attention for both correct and wrong questions.

    Layout (4 columns):
      Col 0: correct Q — Yes token
      Col 1: correct Q — No  token
      Col 2: wrong   Q — Yes token
      Col 3: wrong   Q — No  token
    """
    num_rows = len(layers) + 1
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 4 * num_rows))

    img_np = np.array(image)
    hw = (img_np.shape[0], img_np.shape[1])

    # Col metadata: (which question, which token, base color)
    col_meta = [
        ("correct", "YES", "green"),
        ("correct", "NO",  "green"),
        ("wrong",   "YES", "red"),
        ("wrong",   "NO",  "red"),
    ]

    # Row 0: images + info
    for col, (q_type, token, base_color) in enumerate(col_meta):
        info = info_correct if q_type == "correct" else info_wrong
        pred     = info["prediction"]
        is_pred  = pred == token.lower()
        color    = base_color if is_pred else "#888888"
        chosen_note = "← model chose this" if is_pred else ""

        axes[0, col].imshow(img_np)
        axes[0, col].set_title(
            f"{'Correctly answered' if q_type == 'correct' else 'Incorrectly answered'} question\n"
            f"Attention of the '{token}' token  {chosen_note}\n"
            f"Q: {info['question'][:40]}{'…' if len(info['question']) > 40 else ''}\n"
            f"Ground truth: {info['gt_ans']}   Model said: {pred.upper()}",
            fontsize=7, color=color,
        )
        axes[0, col].axis("off")

    # Rows 1+: heatmaps per layer
    map_sets = [yes_maps_correct, no_maps_correct, yes_maps_wrong, no_maps_wrong]
    for i, layer_idx in enumerate(layers):
        row = i + 1
        for col, maps in enumerate(map_sets):
            attn = maps.get(layer_idx)
            if attn is None:
                axes[row, col].axis("off")
                continue
            attn_resized = _resize_attn(attn, hw)
            axes[row, col].imshow(img_np)
            axes[row, col].imshow(
                attn_resized, cmap="hot", alpha=0.6,
                vmin=0, vmax=attn_resized.max() or 1e-6,
            )
            axes[row, col].set_title(
                f"Transformer layer {layer_idx}\n"
                f"(brighter = more attention to that image region)",
                fontsize=7,
            )
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Token-by-token Yes/No attention maps")

    parser.add_argument("--model-name", type=str,
                        default="chaoyinshe/llava-med-v1.5-mistral-7b-hf")
    parser.add_argument("--margin-scores-file", type=str, required=True,
                        help="Path to margin_scores.json")
    parser.add_argument("--test-file", type=str, required=True,
                        help="Path to test.json")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to image folder")
    parser.add_argument("--output-dir", type=str,
                        default="results/token_attention_maps")
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--layers", type=str, default="0,8,16,24,31",
                        help="Comma-separated layer indices")
    parser.add_argument("--mode", type=str,
                        choices=["yes_no", "correct_wrong"],
                        default="yes_no",
                        help=(
                            "yes_no: show Yes vs No token maps for each question; "
                            "correct_wrong: 4-column comparison across question pairs"
                        ))
    parser.add_argument("--load-8bit", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    layers = [int(x) for x in args.layers.split(",")]
    print(f"Layers to extract: {layers}")
    print(f"Visualization mode: {args.mode}")

    os.makedirs(args.output_dir, exist_ok=True)

    pairs = find_paired_questions(args.margin_scores_file, args.test_file, args.num_pairs)
    if not pairs:
        print("No pairs found — exiting.")
        return

    extractor = TokenAttentionExtractor(
        model_name=args.model_name,
        load_8bit=args.load_8bit,
    )

    for idx, pair in enumerate(tqdm(pairs, desc="Processing pairs")):
        image_path = os.path.join(args.image_folder, pair["image_path"])
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")

            if args.mode == "yes_no":
                # --- Yes vs No maps for the CORRECT question ---
                yes_maps_c, no_maps_c, pred_c, conf_c = extractor.get_both_token_attention_maps(
                    image, pair["correct"]["question"], layers
                )
                out_path = os.path.join(args.output_dir, f"pair_{idx:04d}_correct.png")
                visualize_yes_no_comparison(
                    image, yes_maps_c, no_maps_c,
                    pred_c, conf_c,
                    pair["correct"]["question"],
                    pair["correct"]["gt_ans"],
                    layers, out_path,
                )

                # --- Yes vs No maps for the WRONG question ---
                yes_maps_w, no_maps_w, pred_w, conf_w = extractor.get_both_token_attention_maps(
                    image, pair["wrong"]["question"], layers
                )
                out_path = os.path.join(args.output_dir, f"pair_{idx:04d}_wrong.png")
                visualize_yes_no_comparison(
                    image, yes_maps_w, no_maps_w,
                    pred_w, conf_w,
                    pair["wrong"]["question"],
                    pair["wrong"]["gt_ans"],
                    layers, out_path,
                )

            elif args.mode == "correct_wrong":
                # 4-column: correct(yes/no) vs wrong(yes/no)
                yes_maps_c, no_maps_c, pred_c, conf_c = extractor.get_both_token_attention_maps(
                    image, pair["correct"]["question"], layers
                )
                yes_maps_w, no_maps_w, pred_w, conf_w = extractor.get_both_token_attention_maps(
                    image, pair["wrong"]["question"], layers
                )

                info_correct = {
                    "question": pair["correct"]["question"],
                    "gt_ans": pair["correct"]["gt_ans"],
                    "prediction": pred_c,
                    "confidence": conf_c,
                }
                info_wrong = {
                    "question": pair["wrong"]["question"],
                    "gt_ans": pair["wrong"]["gt_ans"],
                    "prediction": pred_w,
                    "confidence": conf_w,
                }

                out_path = os.path.join(args.output_dir, f"pair_{idx:04d}.png")
                visualize_correct_vs_wrong(
                    image,
                    yes_maps_c, no_maps_c,
                    yes_maps_w, no_maps_w,
                    info_correct, info_wrong,
                    layers, out_path,
                )

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error on pair {idx}: {e}")
            continue

    print(f"\nDone. Saved visualizations to: {args.output_dir}")


if __name__ == "__main__":
    main()
