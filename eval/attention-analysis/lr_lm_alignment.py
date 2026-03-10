import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


def load_lr_weights(weights_file):
    """
    Load saved logistic regression weights.
    
    Args:
        weights_file: path to lr_weights.npz file
    
    Returns:
        weights: list of numpy arrays, one per layer
        intercepts: list of intercepts
    """
    print(f"Loading LR weights from: {weights_file}")
    
    data = np.load(weights_file, allow_pickle=True)
    
    weights = list(data['weights'])
    intercepts = list(data['intercepts'])
    num_layers = int(data['num_layers'])
    
    print(f"  Loaded {len(weights)} layer weights")
    print(f"  Weight shape per layer: {weights[0].shape}")
    print(f"  Total layers (including None): {num_layers}")
    
    return weights, intercepts, num_layers


def extract_lm_head_direction(model_name, load_8bit=True):
    """
    Load model and extract LM head direction: W[yes] - W[no]
    
    Args:
        model_name: HuggingFace model name
        load_8bit: whether to load in 8-bit mode
    
    Returns:
        lm_direction: numpy array of shape (hidden_size,)
    """
    print(f"\nLoading model: {model_name}")
    
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
    
    yes_token_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]
    
    print(f"  Yes token ID: {yes_token_id}")
    print(f"  No token ID: {no_token_id}")
    
    lm_head = model.lm_head
    
    with torch.no_grad():
        w_yes = lm_head.weight[yes_token_id].cpu().float().numpy()
        w_no = lm_head.weight[no_token_id].cpu().float().numpy()
    
    lm_direction = w_yes - w_no
    
    print(f"\nExtracted LM head direction:")
    print(f"  W[yes] shape: {w_yes.shape}")
    print(f"  W[no] shape: {w_no.shape}")
    print(f"  Direction shape: {lm_direction.shape}")
    print(f"  Direction norm: {np.linalg.norm(lm_direction):.4f}")
    
    return lm_direction


def compute_alignment_with_lm_head(lr_weights, lm_direction):
    """
    Compute cosine similarity between each layer's LR weights and LM head direction.
    
    Args:
        lr_weights: list of numpy arrays (LR weights per layer)
        lm_direction: numpy array, the W[yes] - W[no] direction
    
    Returns:
        alignments: list of cosine similarities, one per layer
    """
    alignments = []
    
    print(f"\nComputing alignment for {len(lr_weights)} layers...")
    
    for layer_idx, w_l in enumerate(lr_weights):
        if w_l is None or len(w_l) == 0:
            alignments.append(0.0)
            continue
        
        w_l = np.array(w_l)
        
        sim = cosine_similarity(
            w_l.reshape(1, -1),
            lm_direction.reshape(1, -1)
        )[0][0]
        
        alignments.append(sim)
    
    print(f"  Min alignment: {min(alignments):.4f}")
    print(f"  Max alignment: {max(alignments):.4f}")
    print(f"  Mean alignment: {np.mean(alignments):.4f}")
    print(f"  Final layer alignment: {alignments[-1]:.4f}")
    
    return alignments


def plot_lm_head_alignment(alignments, output_dir, show_stats=True):
    """
    Plot the alignment between LR weights and LM head direction across layers.
    
    Args:
        alignments: list of cosine similarities
        output_dir: where to save the plot
        show_stats: whether to show statistics on the plot
    """
    layers = list(range(len(alignments)))
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(layers, alignments, 'b-o', markersize=6, linewidth=2.5, label='Cosine Similarity', alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Zero alignment')
    
    max_idx = int(np.argmax(alignments))
    max_val = alignments[max_idx]
    ax.plot(max_idx, max_val, 'r*', markersize=20, label=f'Peak (layer {max_idx})', zorder=5)
    
    final_val = alignments[-1]
    ax.plot(len(alignments)-1, final_val, 'gs', markersize=12, label=f'Final layer', zorder=5)
    
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cosine Similarity with LM Head (W[yes] - W[no])', fontsize=14, fontweight='bold')
    ax.set_title('LR Weight Alignment with LM Head Direction per Layer', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    ax.annotate(
        f'Peak: {max_val:.4f}\nat layer {max_idx}',
        xy=(max_idx, max_val),
        xytext=(max_idx + len(layers)*0.05, max_val + 0.05),
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=2)
    )
    
    if show_stats:
        stats_text = f'Statistics:\n'
        stats_text += f'Max: {max_val:.4f} (layer {max_idx})\n'
        stats_text += f'Final: {final_val:.4f}\n'
        stats_text += f'Mean: {np.mean(alignments):.4f}\n'
        stats_text += f'Std: {np.std(alignments):.4f}'
        
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'lm_head_alignment.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved LM head alignment plot to: {plot_path}")
    
    plt.close()
    
    return fig


def analyze_alignment_patterns(alignments):
    """
    Analyze patterns in the alignment across layers.
    
    Args:
        alignments: list of cosine similarities
    
    Returns:
        analysis: dict with key findings
    """
    alignments_arr = np.array(alignments)
    
    max_idx = int(np.argmax(alignments))
    max_val = float(alignments[max_idx])
    final_val = float(alignments[-1])
    
    diffs = np.diff(alignments_arr)
    increasing_ratio = np.sum(diffs > 0) / len(diffs) if len(diffs) > 0 else 0
    
    positive_layers = np.where(alignments_arr > 0)[0]
    first_positive = int(positive_layers[0]) if len(positive_layers) > 0 else None
    
    if first_positive is not None and first_positive < len(alignments) - 1:
        growth_rate = (alignments[-1] - alignments[first_positive]) / (len(alignments) - first_positive)
    else:
        growth_rate = 0
    
    print("\n" + "=" * 60)
    print("ALIGNMENT ANALYSIS")
    print("=" * 60)
    
    print(f"\nPeak alignment: {max_val:.4f} at layer {max_idx}")
    print(f"Final layer alignment: {final_val:.4f}")
    print(f"Mean alignment: {np.mean(alignments):.4f}")
    print(f"Std alignment: {np.std(alignments):.4f}")
    
    if first_positive is not None:
        print(f"\nFirst positive alignment at layer: {first_positive}")
        print(f"Alignment growth rate: {growth_rate:.6f} per layer")
    else:
        print("\nWARNING: No positive alignment found in any layer!")
    
    print(f"\nMonotonicity: {increasing_ratio*100:.1f}% of layer transitions are increasing")
    
    if increasing_ratio > 0.7:
        print("  Alignment generally increases across layers")
    elif increasing_ratio < 0.3:
        print("  Alignment generally decreases across layers")
    else:
        print("  Mixed pattern - alignment fluctuates")
    
    if max_idx == len(alignments) - 1:
        print("\nPeak alignment at final layer - probe aligns well with LM head")
    elif max_idx < len(alignments) // 2:
        print(f"\nWARNING: Peak alignment in early layer ({max_idx}) - alignment degrades in later layers")
    else:
        print(f"\nWARNING: Peak alignment at layer {max_idx}, drops by {max_val - final_val:.4f} at final layer")
    
    analysis = {
        'max_alignment': max_val,
        'max_alignment_layer': max_idx,
        'final_layer_alignment': final_val,
        'mean_alignment': float(np.mean(alignments)),
        'std_alignment': float(np.std(alignments)),
        'first_positive_layer': first_positive,
        'growth_rate': float(growth_rate),
        'increasing_ratio': float(increasing_ratio),
        'alignments_per_layer': [float(a) for a in alignments],
    }
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description='LR-LM Head Alignment Analysis')
    
    parser.add_argument("--weights-file", type=str, required=True,
                        help="Path to saved lr_weights.npz file")
    parser.add_argument("--model-name", type=str,
                        default="chaoyinshe/llava-med-v1.5-mistral-7b-hf",
                        help="Model name to extract LM head from")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for plots and results")
    parser.add_argument("--load-8bit", action="store_true", default=True,
                        help="Load model in 8-bit")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    lr_weights, intercepts, num_layers = load_lr_weights(args.weights_file)
    
    lm_direction = extract_lm_head_direction(args.model_name, args.load_8bit)
    
    if lr_weights[0].shape[0] != lm_direction.shape[0]:
        print(f"\nWARNING: Dimension mismatch!")
        print(f"  LR weights dimension: {lr_weights[0].shape[0]}")
        print(f"  LM direction dimension: {lm_direction.shape[0]}")
        return
    
    alignments = compute_alignment_with_lm_head(lr_weights, lm_direction)
    
    plot_lm_head_alignment(alignments, args.output_dir)
    
    analysis = analyze_alignment_patterns(alignments)
    
    output_file = os.path.join(args.output_dir, 'lm_alignment_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved alignment analysis to: {output_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
