"""
Extract per-attention-head activations at o_proj.input for ITI probe training.

Hooks self_attn.o_proj.input at every layer, grabs the last-token slice,
reshapes to (num_heads, head_dim), and saves the result.

Output per chunk:
    head_activations_chunk{K}.npz   shape: (N, num_layers, num_heads, head_dim)
    gt_labels_chunk{K}.npy          shape: (N,)
    metadata_chunk{K}.json

Merge chunks with merge_chunks.py or pass --merge after extraction.

Phase 1 — extract (one process per GPU):
    CUDA_VISIBLE_DEVICES=0 python extract_head_activations.py \\
        --model llavamed --num-chunks 4 --chunk-idx 0 --max-samples 10000

Phase 2 — merge chunks:
    python extract_head_activations.py --model llavamed --merge-only
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITI_DIR = os.path.dirname(os.path.abspath(__file__))  # outputs live here, under iti/
sys.path.insert(0, EXPERIMENT_DIR)

from intermediate_layer_correction import MODEL_DEFAULTS, build_runner, load_questions


def get_head_config(runner):
    """
    Resolve (num_heads, head_dim) for the model's attention output projection.

    head_dim must come from the config when present (e.g. MedGemma/Gemma3 set an
    explicit head_dim=256 with hidden_size=2560, num_heads=8, so
    hidden_size // num_heads = 320 is WRONG). Falls back to hidden_size // num_heads
    only when no explicit head_dim is configured (e.g. Mistral/LLaVA-Med).
    """
    cfg = getattr(runner.model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    num_heads = text_cfg.num_attention_heads
    hidden_size = text_cfg.hidden_size
    head_dim = getattr(text_cfg, "head_dim", None) or (hidden_size // num_heads)
    return num_heads, head_dim


def get_o_proj(layer):
    """
    Return the attention output-projection module for a decoder layer.

    Most models (Mistral/LLaVA-Med, Gemma/MedGemma) name it `o_proj`; CheXagent is
    Phi-based and names it `dense`. Both take input of shape
    (batch, seq, num_heads * head_dim) — the per-head concatenation we slice on.
    """
    attn = layer.self_attn
    proj = getattr(attn, "o_proj", None)
    if proj is None:
        proj = getattr(attn, "dense", None)
    if proj is None:
        raise AttributeError(
            f"No output projection (o_proj/dense) found on {type(attn).__name__}"
        )
    return proj


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k] if k < len(chunks) else []


def sample_questions(questions, max_samples, seed):
    """Stratified sample by image_id so we don't oversample any one image."""
    if max_samples is None or len(questions) <= max_samples:
        return questions
    rng = np.random.RandomState(seed)
    image_ids = list({q["id"] for q in questions})
    rng.shuffle(image_ids)
    sampled_ids = set(image_ids[:max_samples])
    sampled = [q for q in questions if q["id"] in sampled_ids]
    # if still over budget (multiple questions per image), truncate
    if len(sampled) > max_samples:
        sampled = sampled[:max_samples]
    return sampled


def extract_head_activations(runner, questions, output_dir, chunk_idx, num_chunks):
    """
    Run forward passes with hooks on every layer's o_proj.
    Returns arrays of shape (N, num_layers, num_heads, head_dim).
    """
    num_layers = len(runner.layers)

    # Infer num_heads and head_dim from the model config.
    # Works for LlavaForConditionalGeneration, CheXagent (Phi/CausalLM), MedGemma.
    num_heads, head_dim = get_head_config(runner)

    print(f"Model: {num_layers} layers, {num_heads} heads, head_dim={head_dim}")

    chunk = get_chunk(questions, num_chunks, chunk_idx)
    print(f"Chunk {chunk_idx}/{num_chunks}: {len(chunk)} questions")

    all_acts = []   # list of (num_layers, num_heads, head_dim) arrays
    gt_labels = []
    metadata = []
    skipped = 0

    for q in tqdm(chunk, desc=f"chunk{chunk_idx}"):
        if not os.path.exists(q["image_path"]):
            skipped += 1
            continue

        # Storage for this forward pass: one slot per layer
        layer_acts = [None] * num_layers

        def make_hook(layer_idx):
            # forward_pre_hook signature is (module, args); args[0] is the input to o_proj
            def hook_fn(_module, args):
                # args[0] shape: (batch, seq_len, num_heads * head_dim)
                h = args[0]
                last = h[0, -1, :].detach().float().cpu()   # (num_heads * head_dim,)
                layer_acts[layer_idx] = last.reshape(num_heads, head_dim).numpy()
            return hook_fn

        handles = []
        for li, layer in enumerate(runner.layers):
            h = get_o_proj(layer).register_forward_pre_hook(make_hook(li))
            handles.append(h)

        prepared = None
        try:
            prepared = runner.prepare_image(q["image_path"], "real")
            with torch.inference_mode():
                runner.forward_logits(prepared, q["question"])

            if any(a is None for a in layer_acts):
                skipped += 1
                continue

            all_acts.append(np.stack(layer_acts, axis=0))  # (L, H, D)
            gt_labels.append(q["gt_label"])
            metadata.append({
                "image_id": int(q["id"]),
                "qa_type": q.get("qa_type", ""),
                "image_type": q.get("image_type", ""),
                "is_correct": bool(q.get("is_correct", False)),
            })
        except Exception as e:
            print(f"  Error on id={q['id']}: {e}")
            skipped += 1
        finally:
            for h in handles:
                h.remove()
            if prepared is not None:
                runner.cleanup(prepared)

        if len(all_acts) % 100 == 0:
            torch.cuda.empty_cache()

    print(f"Extracted {len(all_acts)} questions, skipped {skipped}")

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_chunk{chunk_idx}" if num_chunks > 1 else ""

    acts_arr = np.stack(all_acts, axis=0).astype(np.float32)  # (N, L, H, D)
    np.savez_compressed(
        os.path.join(output_dir, f"head_activations{suffix}.npz"),
        activations=acts_arr,
        gt_labels=np.array(gt_labels, dtype=np.int8),
    )
    with open(os.path.join(output_dir, f"metadata{suffix}.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Saved: head_activations{suffix}.npz  shape={acts_arr.shape}")
    return acts_arr, np.array(gt_labels), metadata


def merge_chunks(output_dir, num_chunks):
    """Concatenate per-chunk npz files into a single head_activations.npz."""
    all_acts, all_labels, all_meta = [], [], []
    for k in range(num_chunks):
        npz = np.load(os.path.join(output_dir, f"head_activations_chunk{k}.npz"))
        all_acts.append(npz["activations"])
        all_labels.append(npz["gt_labels"])
        with open(os.path.join(output_dir, f"metadata_chunk{k}.json")) as f:
            all_meta.extend(json.load(f))

    acts = np.concatenate(all_acts, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    np.savez_compressed(
        os.path.join(output_dir, "head_activations.npz"),
        activations=acts,
        gt_labels=labels,
    )
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(all_meta, f)
    print(f"Merged: head_activations.npz  shape={acts.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_DEFAULTS), required=True)
    parser.add_argument("--model-name",    default=None)
    parser.add_argument("--results-file",  required=True)
    parser.add_argument("--test-file",     required=True)
    parser.add_argument("--image-folder",  required=True)
    parser.add_argument("--output-dir",    default=None)
    parser.add_argument("--max-samples",   type=int, default=10000,
                        help="Max questions to process (stratified by image_id)")
    parser.add_argument("--num-chunks",    type=int, default=1)
    parser.add_argument("--chunk-idx",     type=int, default=0)
    parser.add_argument("--merge-only",    action="store_true",
                        help="Skip extraction, just merge existing chunk files")
    parser.add_argument("--load-8bit",     action="store_true", default=False)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    defaults = MODEL_DEFAULTS[args.model]
    model_name = args.model_name or defaults["model_name"]
    output_dir = args.output_dir or os.path.join(
        ITI_DIR, "results", args.model, "iti_head_activations"
    )

    if args.merge_only:
        merge_chunks(output_dir, args.num_chunks)
        return

    all_questions = load_questions(args.results_file, args.test_file, args.image_folder)
    questions = sample_questions(all_questions, args.max_samples, args.seed)
    print(f"Using {len(questions)} questions (max_samples={args.max_samples})")

    runner = build_runner(args.model, model_name, args.load_8bit)
    runner.model.eval()

    extract_head_activations(runner, questions, output_dir, args.chunk_idx, args.num_chunks)

    if args.num_chunks == 1:
        pass  # single file, no merge needed
    elif args.chunk_idx == args.num_chunks - 1:
        print("Last chunk done — run with --merge-only to combine.")


if __name__ == "__main__":
    main()
