"""
Analyze and Visualize VCD Margin Scores
=======================================

Loads margin scores from vcd_margin_analysis.py and:
1. Plots g distribution for GT vs adversarial
2. Computes AUC as binary hallucination detector
3. Analyzes by question type

Usage:
    python analyze_margin_scores.py --input-file results/margin_scores.json
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


def load_results(input_file):
    """Load margin score results."""
    with open(input_file, 'r') as f:
        return json.load(f)


def analyze_by_gt_answer(results):
    """Analyze margin scores by ground truth answer (yes vs no)."""
    
    # Separate by GT answer
    gt_yes = [r for r in results if r['gt_ans'] == 'yes']
    gt_no = [r for r in results if r['gt_ans'] == 'no']
    
    gt_yes_margins = [r['margin_g'] for r in gt_yes]
    gt_no_margins = [r['margin_g'] for r in gt_no]
    
    # Also separate by correctness
    correct = [r for r in results if r['is_correct']]
    incorrect = [r for r in results if not r['is_correct']]
    
    correct_margins = [r['margin_g'] for r in correct]
    incorrect_margins = [r['margin_g'] for r in incorrect]
    
    print("=" * 60)
    print("MARGIN SCORE ANALYSIS")
    print("=" * 60)
    
    print(f"\n--- By Ground Truth Answer ---")
    print(f"GT=Yes (genuine findings): n={len(gt_yes)}")
    print(f"  Mean margin g: {np.mean(gt_yes_margins):.4f} ± {np.std(gt_yes_margins):.4f}")
    print(f"  Median: {np.median(gt_yes_margins):.4f}")
    
    print(f"\nGT=No (false premise/adversarial): n={len(gt_no)}")
    print(f"  Mean margin g: {np.mean(gt_no_margins):.4f} ± {np.std(gt_no_margins):.4f}")
    print(f"  Median: {np.median(gt_no_margins):.4f}")
    
    print(f"\n--- By Model Correctness ---")
    print(f"Correct predictions: n={len(correct)}")
    print(f"  Mean margin g: {np.mean(correct_margins):.4f} ± {np.std(correct_margins):.4f}")
    
    print(f"\nIncorrect predictions: n={len(incorrect)}")
    print(f"  Mean margin g: {np.mean(incorrect_margins):.4f} ± {np.std(incorrect_margins):.4f}")
    
    return {
        'gt_yes': gt_yes_margins,
        'gt_no': gt_no_margins,
        'correct': correct_margins,
        'incorrect': incorrect_margins,
    }


def compute_auc(results):
    """
    Compute AUC for margin score as binary hallucination detector.
    
    Task: Distinguish GT=yes (genuine) from GT=no (adversarial/hallucinated)
    Higher margin g should indicate genuine findings.
    """
    
    # Labels: 1 = genuine (GT=yes), 0 = adversarial (GT=no)
    labels = []
    scores = []
    
    for r in results:
        if r['gt_ans'] == 'yes':
            labels.append(1)
        else:
            labels.append(0)
        scores.append(r['margin_g'])
    
    labels = np.array(labels)
    scores = np.array(scores)
    
    # Compute ROC AUC
    if len(np.unique(labels)) < 2:
        print("Warning: Only one class present, cannot compute AUC")
        return None
    
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Compute precision-recall AUC
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    
    print(f"\n--- AUC Analysis ---")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nOptimal threshold (Youden's J): {optimal_threshold:.4f}")
    print(f"  TPR at threshold: {tpr[optimal_idx]:.4f}")
    print(f"  FPR at threshold: {fpr[optimal_idx]:.4f}")
    
    return {
        'auc': auc,
        'ap': ap,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'optimal_threshold': optimal_threshold,
        'precision': precision,
        'recall': recall,
    }


def analyze_by_question_type(results):
    """Analyze margin scores by question type."""
    
    from collections import defaultdict
    
    by_type = defaultdict(list)
    for r in results:
        by_type[r['qa_type']].append(r)
    
    print(f"\n--- By Question Type ---")
    
    for qa_type, items in sorted(by_type.items()):
        margins = [r['margin_g'] for r in items]
        gt_yes = [r for r in items if r['gt_ans'] == 'yes']
        gt_no = [r for r in items if r['gt_ans'] == 'no']
        
        accuracy = sum(1 for r in items if r['is_correct']) / len(items) * 100
        
        print(f"\n{qa_type}: n={len(items)}, accuracy={accuracy:.1f}%")
        print(f"  Mean margin g: {np.mean(margins):.4f}")
        if gt_yes:
            print(f"  GT=yes margin: {np.mean([r['margin_g'] for r in gt_yes]):.4f}")
        if gt_no:
            print(f"  GT=no margin: {np.mean([r['margin_g'] for r in gt_no]):.4f}")


def plot_distributions(margins_dict, auc_results, output_dir):
    """Plot margin score distributions and ROC curve."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Distribution by GT answer
    ax1 = axes[0, 0]
    ax1.hist(margins_dict['gt_yes'], bins=50, alpha=0.6, 
             label=f"GT=Yes (genuine), n={len(margins_dict['gt_yes'])}", color='green')
    ax1.hist(margins_dict['gt_no'], bins=50, alpha=0.6,
             label=f"GT=No (adversarial), n={len(margins_dict['gt_no'])}", color='red')
    ax1.axvline(np.mean(margins_dict['gt_yes']), color='green', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(margins_dict['gt_no']), color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Margin Score (g)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Margin Score Distribution by Ground Truth', fontsize=14)
    ax1.legend(fontsize=10)
    
    # 2. Distribution by correctness
    ax2 = axes[0, 1]
    ax2.hist(margins_dict['correct'], bins=50, alpha=0.6,
             label=f"Correct, n={len(margins_dict['correct'])}", color='blue')
    ax2.hist(margins_dict['incorrect'], bins=50, alpha=0.6,
             label=f"Incorrect, n={len(margins_dict['incorrect'])}", color='orange')
    ax2.axvline(np.mean(margins_dict['correct']), color='blue', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(margins_dict['incorrect']), color='orange', linestyle='--', linewidth=2)
    ax2.set_xlabel('Margin Score (g)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Margin Score Distribution by Model Correctness', fontsize=14)
    ax2.legend(fontsize=10)
    
    # 3. ROC Curve
    ax3 = axes[1, 0]
    if auc_results:
        ax3.plot(auc_results['fpr'], auc_results['tpr'], 
                 color='darkorange', lw=2, 
                 label=f"ROC curve (AUC = {auc_results['auc']:.3f})")
        ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax3.scatter([auc_results['fpr'][np.argmax(auc_results['tpr'] - auc_results['fpr'])]],
                   [auc_results['tpr'][np.argmax(auc_results['tpr'] - auc_results['fpr'])]],
                   marker='o', color='red', s=100, label='Optimal threshold')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate', fontsize=12)
    ax3.set_ylabel('True Positive Rate', fontsize=12)
    ax3.set_title('ROC Curve: Margin Score as Hallucination Detector', fontsize=14)
    ax3.legend(loc="lower right", fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot comparison
    ax4 = axes[1, 1]
    box_data = [margins_dict['gt_yes'], margins_dict['gt_no']]
    bp = ax4.boxplot(box_data, labels=['GT=Yes\n(genuine)', 'GT=No\n(adversarial)'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax4.set_ylabel('Margin Score (g)', fontsize=12)
    ax4.set_title('Margin Score Comparison', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_path = f"{output_dir}/margin_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    
    plt.show()
    
    return fig


def save_summary(results, margins_dict, auc_results, output_dir):
    """Save analysis summary to JSON."""
    
    summary = {
        'total_samples': len(results),
        'gt_yes_count': len(margins_dict['gt_yes']),
        'gt_no_count': len(margins_dict['gt_no']),
        'correct_count': len(margins_dict['correct']),
        'incorrect_count': len(margins_dict['incorrect']),
        'accuracy': len(margins_dict['correct']) / len(results) * 100,
        'margin_stats': {
            'gt_yes_mean': float(np.mean(margins_dict['gt_yes'])),
            'gt_yes_std': float(np.std(margins_dict['gt_yes'])),
            'gt_no_mean': float(np.mean(margins_dict['gt_no'])),
            'gt_no_std': float(np.std(margins_dict['gt_no'])),
            'correct_mean': float(np.mean(margins_dict['correct'])),
            'incorrect_mean': float(np.mean(margins_dict['incorrect'])),
        },
        'auc_roc': float(auc_results['auc']) if auc_results else None,
        'average_precision': float(auc_results['ap']) if auc_results else None,
        'optimal_threshold': float(auc_results['optimal_threshold']) if auc_results else None,
    }
    
    output_path = f"{output_dir}/analysis_summary.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze VCD Margin Scores')
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to margin scores JSON from vcd_margin_analysis.py")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for plots and summary")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.input_file}")
    results = load_results(args.input_file)
    print(f"Loaded {len(results)} samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze by GT answer
    margins_dict = analyze_by_gt_answer(results)
    
    # Compute AUC
    auc_results = compute_auc(results)
    
    # Analyze by question type
    analyze_by_question_type(results)
    
    # Plot
    plot_distributions(margins_dict, auc_results, args.output_dir)
    
    # Save summary
    save_summary(results, margins_dict, auc_results, args.output_dir)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import os
    main()