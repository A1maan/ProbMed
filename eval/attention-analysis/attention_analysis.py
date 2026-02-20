"""
Attention Analysis for Paired Questions (Multi-GPU Support)
============================================================

Extracts and compares attention patterns between correct and wrong predictions
on the same image to identify where the model's attention diverges.

Single GPU usage:
    python attention_analysis.py \
        --margin-scores-file ../vcd/results/vcd_analysis/margin_scores.json \
        --image-folder /workspace/ProbMed-Dataset/test/ \
        --output-dir results/attention_analysis \
        --num-pairs 50

Multi-GPU usage (called by run_attention_batch.py):
    CUDA_VISIBLE_DEVICES=0 python attention_analysis.py \
        --margin-scores-file ../vcd/results/vcd_analysis/margin_scores.json \
        --image-folder /workspace/ProbMed-Dataset/test/ \
        --output-dir results/attention_analysis/chunk0 \
        --num-pairs 250 \
        --num-chunks 4 \
        --chunk-idx 0
"""

import argparse
import json
import os
import random
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """Get chunk k out of n chunks"""
    chunks = split_list(lst, n)
    return chunks[k] if k < len(chunks) else []


class AttentionAnalyzer:
    """Extracts and analyzes attention patterns from LLaVA-Med."""
    
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
                attn_implementation="eager",  # Required for output_attentions=True
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                attn_implementation="eager",  # Required for output_attentions=True
            )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"
        
        self.model.eval()
        
        # Get model config
        self.num_layers = self.model.config.text_config.num_hidden_layers
        self.num_heads = self.model.config.text_config.num_attention_heads
        
        # Image tokens: LLaVA uses 24x24 = 576 patches
        self.image_grid_size = 24
        self.num_image_tokens = self.image_grid_size ** 2
        
        print(f"Model loaded! Layers: {self.num_layers}, Heads: {self.num_heads}")
        print(f"Image grid: {self.image_grid_size}x{self.image_grid_size} = {self.num_image_tokens} tokens")
    
    @property
    def device(self):
        return self.model.device
    
    def format_prompt(self, question):
        """Format prompt for the model."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    def _find_image_token_range(self, input_ids):
        """
        Find the start and end positions of image tokens in the sequence.
        
        The processor inserts a single <image> placeholder token; the model's
        multi-modal projector expands it to num_image_tokens patches in-place.
        """
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
        positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            raise ValueError("No <image> token found in input_ids.")
        img_start = positions[0].item()
        img_end = img_start + self.num_image_tokens
        return img_start, img_end
    
    def get_attention_maps(self, image, question):
        """
        Extract attention maps from all layers.
        
        Returns:
            attentions: tuple of (batch, num_heads, seq_len, seq_len) for each layer
            token_info: dict with token positions and text
        """
        prompt = self.format_prompt(question)
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Find image token range BEFORE forward pass (using placeholder position)
        img_start, img_end = self._find_image_token_range(inputs['input_ids'])
        
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )
        
        # Get attention weights from all layers
        # Each attention: (batch, num_heads, seq_len, seq_len)
        attentions = outputs.attentions
        
        # Get token information
        input_ids = inputs['input_ids'][0]
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Actual sequence length after image tokens are expanded
        actual_seq_len = attentions[0].shape[-1]
        
        token_info = {
            'seq_len': actual_seq_len,
            'num_image_tokens': self.num_image_tokens,
            'img_start': img_start,
            'img_end': img_end,
            'tokens': tokens,
            'input_ids': input_ids.cpu().tolist(),
        }
        
        # Get logits for Yes/No prediction
        logits = outputs.logits[:, -1, :]
        yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]
        
        pred = 'yes' if logits[0, yes_token_id] > logits[0, no_token_id] else 'no'
        
        return attentions, token_info, pred
    
    def compute_attention_stats(self, attentions, token_info):
        """
        Compute attention statistics for image vs text tokens.
        
        Returns:
            stats: dict with per-layer attention statistics
        """
        img_start = token_info['img_start']
        img_end = token_info['img_end']
        seq_len = token_info['seq_len']
        
        stats = {
            'per_layer': [],
            'summary': {}
        }
        
        for layer_idx, attn in enumerate(attentions):
            # attn: (batch, heads, seq, seq)
            attn = attn[0]  # Remove batch dim: (heads, seq, seq)
            
            # Average over heads
            attn_avg = attn.mean(dim=0)  # (seq, seq)
            
            # Last token attending to image tokens
            # (this is where the model looks when generating the answer)
            last_to_img = attn_avg[-1, img_start:img_end].mean().item()
            
            # Last token attending to non-image tokens (text before and after image)
            text_before = attn_avg[-1, :img_start].mean().item() if img_start > 0 else 0
            text_after = attn_avg[-1, img_end:].mean().item() if img_end < seq_len else 0
            last_to_text = (text_before + text_after) / 2 if (img_start > 0 or img_end < seq_len) else 0
            
            # All text tokens (after image) attending to image
            if img_end < seq_len:
                text_to_img = attn_avg[img_end:, img_start:img_end].mean().item()
            else:
                text_to_img = 0
            
            # Image attention entropy (how spread out is attention over image?)
            img_attn = attn_avg[-1, img_start:img_end]
            img_attn_normalized = img_attn / (img_attn.sum() + 1e-10)
            entropy = -(img_attn_normalized * torch.log(img_attn_normalized + 1e-10)).sum().item()
            
            stats['per_layer'].append({
                'layer': layer_idx,
                'last_to_img': last_to_img,
                'last_to_text': last_to_text,
                'text_to_img': text_to_img,
                'img_attention_entropy': entropy,
            })
        
        # Summary stats
        stats['summary'] = {
            'avg_last_to_img': np.mean([s['last_to_img'] for s in stats['per_layer']]),
            'avg_last_to_text': np.mean([s['last_to_text'] for s in stats['per_layer']]),
            'avg_entropy': np.mean([s['img_attention_entropy'] for s in stats['per_layer']]),
        }
        
        return stats


def find_paired_questions(margin_scores_file, test_file=None, max_pairs=None, num_chunks=1, chunk_idx=0):
    """
    Find pairs of questions on the same image where one is correct and one is wrong.
    
    Args:
        margin_scores_file: Path to margin_scores.json
        test_file: Path to original test.json (to get image paths)
        max_pairs: Maximum pairs to use (None = all pairs)
        num_chunks: Number of chunks for parallel processing
        chunk_idx: Which chunk to process
    """
    print(f"Loading margin scores from: {margin_scores_file}")
    with open(margin_scores_file, 'r') as f:
        results = json.load(f)
    
    print(f"Total samples: {len(results)}")
    
    # Load original test.json to get image paths
    # Create mapping from (id, question) to image path for accurate matching
    id_question_to_image = {}
    if test_file and os.path.exists(test_file):
        print(f"Loading image paths from: {test_file}")
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        for item in test_data:
            item_id = item.get('id')
            question = item.get('question', '').replace('<image>', '').strip()
            if item_id is not None and 'image' in item:
                id_question_to_image[(item_id, question)] = item['image']
        print(f"Loaded {len(id_question_to_image)} question-to-image mappings")
    
    # Group by IMAGE PATH (not ID) - this is the fix
    by_image = defaultdict(list)
    for r in results:
        # Match by both id and question to get correct image path
        key = (r.get('id'), r.get('question', ''))
        image_path = id_question_to_image.get(key)
        
        if image_path:
            r['image'] = image_path
            by_image[image_path].append(r)
        else:
            # Fallback: skip if we can't find image path
            pass
    
    print(f"Unique images: {len(by_image)}")
    
    # Find ALL pairs (one correct + one wrong per image)
    pairs = []
    for img_id, questions in by_image.items():
        correct = [q for q in questions if q.get('is_correct', False)]
        wrong = [q for q in questions if not q.get('is_correct', False)]
        
        if correct and wrong:
            # Take first of each
            pairs.append({
                'image_id': img_id,
                'correct': correct[0],
                'wrong': wrong[0],
            })
    
    print(f"Found {len(pairs)} image pairs with correct+wrong questions")
    
    # Only limit if max_pairs is specified
    if max_pairs is not None and len(pairs) > max_pairs:
        random.seed(42)  # Fixed seed for consistent sampling across chunks
        pairs = random.sample(pairs, max_pairs)
        print(f"Limited to {max_pairs} pairs")
    
    # Get this chunk's portion
    pairs = get_chunk(pairs, num_chunks, chunk_idx)
    print(f"Chunk {chunk_idx}/{num_chunks}: processing {len(pairs)} pairs")
    
    return pairs


def analyze_pairs(args):
    """Run attention analysis on paired questions."""
    
    # Find pairs (with chunking support)
    pairs = find_paired_questions(
        args.margin_scores_file, 
        test_file=args.test_file,
        max_pairs=args.num_pairs,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx
    )
    
    if not pairs:
        print("No pairs found! Check your margin_scores.json format.")
        return
    
    # Initialize analyzer
    analyzer = AttentionAnalyzer(
        model_name=args.model_name,
        load_8bit=args.load_8bit
    )
    
    # Analyze pairs
    results = []
    
    for pair in tqdm(pairs, desc="Analyzing pairs"):
        correct_q = pair['correct']
        wrong_q = pair['wrong']
        
        # Get image path from the correct question
        # Need to find image path - check if it's in the data
        image_file = correct_q.get('image', None)
        if not image_file:
            # Try to reconstruct from ID
            # This depends on your data format
            continue
        
        image_path = os.path.join(args.image_folder, image_file)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Get attention for correct question
            attn_correct, token_info_correct, pred_correct = analyzer.get_attention_maps(
                image, correct_q['question']
            )
            stats_correct = analyzer.compute_attention_stats(attn_correct, token_info_correct)
            
            # Get attention for wrong question
            attn_wrong, token_info_wrong, pred_wrong = analyzer.get_attention_maps(
                image, wrong_q['question']
            )
            stats_wrong = analyzer.compute_attention_stats(attn_wrong, token_info_wrong)
            
            results.append({
                'image_id': pair['image_id'],
                'correct_question': correct_q['question'],
                'wrong_question': wrong_q['question'],
                'correct_gt': correct_q['gt_ans'],
                'wrong_gt': wrong_q['gt_ans'],
                'correct_pred': pred_correct,
                'wrong_pred': pred_wrong,
                'correct_stats': stats_correct,
                'wrong_stats': stats_wrong,
            })
            
            # Clear GPU memory
            del attn_correct, attn_wrong
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing pair {pair['image_id']}: {e}")
            continue
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'attention_analysis.json')
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Saved {len(results)} pair analyses to: {output_file}")
    
    return results


def plot_attention_comparison(results, output_dir):
    """Plot attention statistics comparison between correct and wrong predictions."""
    
    if not results:
        print("No results to plot")
        return
    
    num_layers = len(results[0]['correct_stats']['per_layer'])
    
    # Aggregate per-layer stats
    correct_last_to_img = [[] for _ in range(num_layers)]
    wrong_last_to_img = [[] for _ in range(num_layers)]
    correct_entropy = [[] for _ in range(num_layers)]
    wrong_entropy = [[] for _ in range(num_layers)]
    
    for r in results:
        for layer_idx in range(num_layers):
            correct_last_to_img[layer_idx].append(
                r['correct_stats']['per_layer'][layer_idx]['last_to_img']
            )
            wrong_last_to_img[layer_idx].append(
                r['wrong_stats']['per_layer'][layer_idx]['last_to_img']
            )
            correct_entropy[layer_idx].append(
                r['correct_stats']['per_layer'][layer_idx]['img_attention_entropy']
            )
            wrong_entropy[layer_idx].append(
                r['wrong_stats']['per_layer'][layer_idx]['img_attention_entropy']
            )
    
    # Compute means
    correct_last_to_img_mean = [np.mean(x) for x in correct_last_to_img]
    wrong_last_to_img_mean = [np.mean(x) for x in wrong_last_to_img]
    correct_entropy_mean = [np.mean(x) for x in correct_entropy]
    wrong_entropy_mean = [np.mean(x) for x in wrong_entropy]
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    layers = list(range(num_layers))
    
    # 1. Last token attention to image tokens
    ax1 = axes[0, 0]
    ax1.plot(layers, correct_last_to_img_mean, 'g-o', label='Correct', markersize=4)
    ax1.plot(layers, wrong_last_to_img_mean, 'r-o', label='Wrong', markersize=4)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Attention Weight')
    ax1.set_title('Last Token → Image Attention (per layer)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Image attention entropy
    ax2 = axes[0, 1]
    ax2.plot(layers, correct_entropy_mean, 'g-o', label='Correct', markersize=4)
    ax2.plot(layers, wrong_entropy_mean, 'r-o', label='Wrong', markersize=4)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Image Attention Entropy (per layer)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Difference in attention to image
    ax3 = axes[1, 0]
    diff_last_to_img = [c - w for c, w in zip(correct_last_to_img_mean, wrong_last_to_img_mean)]
    ax3.bar(layers, diff_last_to_img, color=['green' if d > 0 else 'red' for d in diff_last_to_img])
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Difference (Correct - Wrong)')
    ax3.set_title('Attention Difference to Image')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary box plot
    ax4 = axes[1, 1]
    correct_summary = [r['correct_stats']['summary']['avg_last_to_img'] for r in results]
    wrong_summary = [r['wrong_stats']['summary']['avg_last_to_img'] for r in results]
    ax4.boxplot([correct_summary, wrong_summary], labels=['Correct', 'Wrong'])
    ax4.set_ylabel('Avg Attention to Image')
    ax4.set_title('Overall Image Attention Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'attention_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    
    plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Attention Analysis for Paired Questions')
    
    parser.add_argument("--model-name", type=str,
                        default="chaoyinshe/llava-med-v1.5-mistral-7b-hf")
    parser.add_argument("--margin-scores-file", type=str, required=True,
                        help="Path to margin_scores.json from VCD experiment")
    parser.add_argument("--test-file", type=str, default=None,
                        help="Path to original test.json (to get image paths)")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to image folder")
    parser.add_argument("--output-dir", type=str, default="results/attention_analysis",
                        help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=None,
                        help="Number of pairs to analyze (default: all pairs)")
    parser.add_argument("--load-8bit", action="store_true", default=True,
                        help="Load model in 8-bit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Number of chunks for parallel processing")
    parser.add_argument("--chunk-idx", type=int, default=0,
                        help="Which chunk to process (0-indexed)")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Run analysis
    results = analyze_pairs(args)
    
    # Plot results
    if results:
        plot_attention_comparison(results, args.output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()