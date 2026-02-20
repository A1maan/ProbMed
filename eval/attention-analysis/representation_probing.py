"""
Layer-wise Representation Probing
=================================

Extracts hidden representations from each layer and trains a logistic regression
classifier to predict correct Yes/No answer. This helps identify if early/mid
layers have correct information that gets destroyed in later layers.

Usage:
    python representation_probing.py \
        --margin-scores-file ../vcd/results/vcd_analysis/margin_scores.json \
        --image-folder /workspace/ProbMed-Dataset/test/ \
        --output-dir results/representation_probing \
        --num-samples 500
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


class RepresentationExtractor:
    """Extracts hidden representations from each layer of LLaVA-Med."""
    
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
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"
        
        self.model.eval()
        
        # Get model config
        self.num_layers = self.model.config.text_config.num_hidden_layers
        self.hidden_size = self.model.config.text_config.hidden_size
        
        print(f"Model loaded! Layers: {self.num_layers}, Hidden size: {self.hidden_size}")
    
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
    
    def extract_layer_representations(self, image, question):
        """
        Extract hidden representations from all layers.
        
        Returns:
            representations: list of tensors, one per layer (last token representation)
            prediction: model's yes/no prediction
        """
        prompt = self.format_prompt(question)
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get hidden states from all layers
        # hidden_states: tuple of (batch, seq_len, hidden_size) for each layer
        hidden_states = outputs.hidden_states
        
        # Extract last token representation from each layer
        representations = []
        for layer_idx, hidden in enumerate(hidden_states):
            # Take last token's representation
            last_token_repr = hidden[0, -1, :].cpu().numpy()
            representations.append(last_token_repr)
        
        # Get model prediction
        logits = outputs.logits[:, -1, :]
        yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]
        
        yes_logit = logits[0, yes_token_id].item()
        no_logit = logits[0, no_token_id].item()
        prediction = 'yes' if yes_logit > no_logit else 'no'
        
        return representations, prediction, yes_logit, no_logit


def load_samples(margin_scores_file, num_samples=500):
    """Load samples from margin scores file."""
    
    print(f"Loading samples from: {margin_scores_file}")
    with open(margin_scores_file, 'r') as f:
        results = json.load(f)
    
    print(f"Total samples: {len(results)}")
    
    # Filter samples that have image info
    valid_samples = [r for r in results if r.get('question') and r.get('gt_ans')]
    print(f"Valid samples: {len(valid_samples)}")
    
    # Sample if needed
    if len(valid_samples) > num_samples:
        random.seed(42)
        valid_samples = random.sample(valid_samples, num_samples)
        print(f"Sampled {num_samples} samples")
    
    return valid_samples


def extract_all_representations(extractor, samples, image_folder):
    """Extract representations for all samples."""
    
    # Storage for representations per layer
    num_layers = extractor.num_layers + 1  # +1 for embedding layer
    layer_representations = [[] for _ in range(num_layers)]
    labels = []  # Ground truth: 1 for yes, 0 for no
    sample_info = []
    
    for sample in tqdm(samples, desc="Extracting representations"):
        # Get image path
        image_file = sample.get('image')
        if not image_file:
            # Try to reconstruct from the original data
            # Skip if no image info
            continue
        
        image_path = os.path.join(image_folder, image_file)
        
        if not os.path.exists(image_path):
            continue
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            representations, pred, yes_logit, no_logit = extractor.extract_layer_representations(
                image, sample['question']
            )
            
            # Store representations for each layer
            for layer_idx, repr in enumerate(representations):
                layer_representations[layer_idx].append(repr)
            
            # Label: what the model SHOULD answer (ground truth)
            gt_label = 1 if sample['gt_ans'] == 'yes' else 0
            labels.append(gt_label)
            
            sample_info.append({
                'id': sample.get('id'),
                'question': sample['question'],
                'gt_ans': sample['gt_ans'],
                'model_pred': pred,
                'is_correct': (pred == sample['gt_ans']),
            })
            
            # Clear GPU memory periodically
            if len(labels) % 50 == 0:
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    # Convert to numpy arrays
    layer_representations = [np.array(reps) for reps in layer_representations]
    labels = np.array(labels)
    
    print(f"Extracted representations for {len(labels)} samples")
    print(f"Label distribution: {np.sum(labels)} yes, {len(labels) - np.sum(labels)} no")
    
    return layer_representations, labels, sample_info


def train_probing_classifiers(layer_representations, labels, test_size=0.2):
    """
    Train logistic regression classifier for each layer.
    
    Returns:
        results: list of dicts with accuracy and AUC per layer
    """
    
    results = []
    num_layers = len(layer_representations)
    
    # Split data
    train_idx, test_idx = train_test_split(
        range(len(labels)), 
        test_size=test_size, 
        random_state=42,
        stratify=labels
    )
    
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    
    print(f"\nTraining probing classifiers...")
    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    
    for layer_idx in tqdm(range(num_layers), desc="Training classifiers"):
        X = layer_representations[layer_idx]
        X_train = X[train_idx]
        X_test = X[test_idx]
        
        # Train logistic regression (re-initialized for each layer)
        clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            n_jobs=-1
        )
        
        try:
            clf.fit(X_train, y_train)
            
            # Predict
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            results.append({
                'layer': layer_idx,
                'accuracy': accuracy,
                'auc': auc,
            })
            
        except Exception as e:
            print(f"Error training layer {layer_idx}: {e}")
            results.append({
                'layer': layer_idx,
                'accuracy': 0.5,
                'auc': 0.5,
            })
    
    return results


def plot_layer_accuracy(results, output_dir):
    """Plot accuracy and AUC per layer."""
    
    layers = [r['layer'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    aucs = [r['auc'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1 = axes[0]
    ax1.plot(layers, accuracies, 'b-o', markersize=4)
    ax1.axhline(y=0.5, color='red', linestyle='--', label='Random baseline')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Probing Accuracy per Layer', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.4, 1.0])
    
    # AUC plot
    ax2 = axes[1]
    ax2.plot(layers, aucs, 'g-o', markersize=4)
    ax2.axhline(y=0.5, color='red', linestyle='--', label='Random baseline')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('Probing AUC per Layer', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'layer_probing_accuracy.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    
    plt.show()
    
    # Also create a bar chart
    fig2, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, aucs, width, label='AUC', color='forestgreen')
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Layer-wise Probing Performance', fontsize=14)
    ax.set_xticks(x[::2])  # Show every other layer for readability
    ax.set_xticklabels([str(l) for l in layers[::2]])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    
    bar_path = os.path.join(output_dir, 'layer_probing_bars.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    print(f"Saved bar plot to: {bar_path}")
    
    return fig, fig2


def analyze_results(results):
    """Analyze probing results to find insights."""
    
    accuracies = [r['accuracy'] for r in results]
    aucs = [r['auc'] for r in results]
    
    print("\n" + "=" * 60)
    print("PROBING ANALYSIS RESULTS")
    print("=" * 60)
    
    # Find best layer
    best_acc_layer = np.argmax(accuracies)
    best_auc_layer = np.argmax(aucs)
    
    print(f"\nBest accuracy: {accuracies[best_acc_layer]:.4f} at layer {best_acc_layer}")
    print(f"Best AUC: {aucs[best_auc_layer]:.4f} at layer {best_auc_layer}")
    
    # Check for accuracy drop
    max_acc = max(accuracies)
    final_acc = accuracies[-1]
    
    if max_acc > final_acc + 0.05:
        print(f"\n⚠️  FINDING: Accuracy drops from {max_acc:.4f} (layer {np.argmax(accuracies)}) "
              f"to {final_acc:.4f} (final layer)")
        print("   This suggests correct information exists in early/mid layers but gets corrupted later!")
    else:
        print(f"\n✓ No significant accuracy drop detected (max: {max_acc:.4f}, final: {final_acc:.4f})")
    
    # Check early vs late layers
    mid_point = len(results) // 2
    early_avg = np.mean(accuracies[:mid_point])
    late_avg = np.mean(accuracies[mid_point:])
    
    print(f"\nEarly layers (0-{mid_point-1}) avg accuracy: {early_avg:.4f}")
    print(f"Late layers ({mid_point}-{len(results)-1}) avg accuracy: {late_avg:.4f}")
    
    return {
        'best_acc_layer': best_acc_layer,
        'best_acc': accuracies[best_acc_layer],
        'best_auc_layer': best_auc_layer,
        'best_auc': aucs[best_auc_layer],
        'final_acc': final_acc,
        'early_avg': early_avg,
        'late_avg': late_avg,
        'accuracy_drop': max_acc - final_acc,
    }


def main():
    parser = argparse.ArgumentParser(description='Layer-wise Representation Probing')
    
    parser.add_argument("--model-name", type=str,
                        default="chaoyinshe/llava-med-v1.5-mistral-7b-hf")
    parser.add_argument("--margin-scores-file", type=str, required=True,
                        help="Path to margin_scores.json from VCD experiment")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to image folder")
    parser.add_argument("--output-dir", type=str, default="results/representation_probing",
                        help="Output directory")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of samples to use")
    parser.add_argument("--load-8bit", action="store_true", default=True,
                        help="Load model in 8-bit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load samples
    samples = load_samples(args.margin_scores_file, args.num_samples)
    
    if not samples:
        print("No samples found!")
        return
    
    # Initialize extractor
    extractor = RepresentationExtractor(
        model_name=args.model_name,
        load_8bit=args.load_8bit
    )
    
    # Extract representations
    layer_representations, labels, sample_info = extract_all_representations(
        extractor, samples, args.image_folder
    )
    
    if len(labels) < 50:
        print("Not enough samples extracted!")
        return
    
    # Train probing classifiers
    probing_results = train_probing_classifiers(layer_representations, labels)
    
    # Save results
    output_file = os.path.join(args.output_dir, 'probing_results.json')
    with open(output_file, 'w') as f:
        json.dump(probing_results, f, indent=2)
    print(f"Saved probing results to: {output_file}")
    
    # Plot results
    plot_layer_accuracy(probing_results, args.output_dir)
    
    # Analyze
    analysis = analyze_results(probing_results)
    
    # Save analysis
    analysis_file = os.path.join(args.output_dir, 'analysis_summary.json')
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("\nDone!")


if __name__ == "__main__":
    main()