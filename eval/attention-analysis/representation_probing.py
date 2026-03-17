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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


class PairedRepresentationExtractor:
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
        
        # Yes/No token IDs
        self.yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]
        
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
            representations: list of numpy arrays, one per layer (last token representation)
            prediction: model's yes/no prediction
            is_correct: whether prediction matches ground truth
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
        yes_logit = logits[0, self.yes_token_id].item()
        no_logit = logits[0, self.no_token_id].item()
        prediction = 'yes' if yes_logit > no_logit else 'no'
        
        return representations, prediction


def find_paired_questions(margin_scores_file, test_file, num_pairs=500):
    """
    Find pairs of questions on the same image where one is correct and one is wrong.
    """
    print(f"Loading margin scores from: {margin_scores_file}")
    with open(margin_scores_file, 'r') as f:
        results = json.load(f)
    
    print(f"Total samples: {len(results)}")
    
    # Load original test.json to get image paths
    print(f"Loading image paths from: {test_file}")
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    id_question_to_image = {}
    for item in test_data:
        item_id = item.get('id')
        question = item.get('question', '').replace('<image>', '').strip()
        if item_id is not None and 'image' in item:
            id_question_to_image[(item_id, question)] = item['image']
    
    print(f"Loaded {len(id_question_to_image)} question-to-image mappings")
    
    # Group by image path
    by_image = defaultdict(list)
    for r in results:
        key = (r.get('id'), r.get('question', ''))
        image_path = id_question_to_image.get(key)
        if image_path:
            r['image'] = image_path
            by_image[image_path].append(r)
    
    print(f"Unique images: {len(by_image)}")
    
    # Find pairs (one correct, one wrong per image)
    pairs = []
    for img_path, questions in by_image.items():
        correct = [q for q in questions if q.get('is_correct', False)]
        wrong = [q for q in questions if not q.get('is_correct', False)]
        
        if correct and wrong:
            pairs.append({
                'image_path': img_path,
                'correct': correct[0],
                'wrong': wrong[0],
            })
    
    print(f"Found {len(pairs)} valid pairs (images with both correct and wrong)")
    
    # Sample if needed
    if len(pairs) > num_pairs:
        random.seed(42)
        pairs = random.sample(pairs, num_pairs)
        print(f"Sampled {num_pairs} pairs")
    
    return pairs


def extract_paired_representations(extractor, pairs, image_folder, image_mode="real"):
    """
    Extract representations for all paired questions.

    For each pair (same image):
        - Extract representations for correct question
        - Extract representations for wrong question
    """
    assert image_mode in ("real", "black", "random")
    num_layers = extractor.num_layers + 1  # +1 for embedding layer
    
    # Storage: separate lists for correct and wrong questions
    correct_representations = [[] for _ in range(num_layers)]
    wrong_representations = [[] for _ in range(num_layers)]
    correct_labels = []  # Ground truth for correct questions
    wrong_labels = []    # Ground truth for wrong questions
    
    pair_info = []
    
    for pair in tqdm(pairs, desc="Extracting paired representations"):
        image_path = os.path.join(image_folder, pair['image_path'])
        
        if not os.path.exists(image_path):
            print(f"Image not found, skipping: {image_path}")
            continue
        
        try:
            image = Image.open(image_path).convert('RGB')
            if image_mode == 'black':
                image = Image.new('RGB', image.size, (0, 0, 0))
            elif image_mode == 'random':
                arr = np.random.randint(0, 256, (image.size[1], image.size[0], 3), dtype=np.uint8)
                image = Image.fromarray(arr)

            # Extract for CORRECT question
            repr_correct, pred_correct = extractor.extract_layer_representations(
                image, pair['correct']['question']
            )
            
            # Extract for WRONG question
            repr_wrong, pred_wrong = extractor.extract_layer_representations(
                image, pair['wrong']['question']
            )
            
            # Store representations
            for layer_idx in range(num_layers):
                correct_representations[layer_idx].append(repr_correct[layer_idx])
                wrong_representations[layer_idx].append(repr_wrong[layer_idx])
            
            # Store ground truth labels (yes=1, no=0)
            correct_labels.append(1 if pair['correct']['gt_ans'] == 'yes' else 0)
            wrong_labels.append(1 if pair['wrong']['gt_ans'] == 'yes' else 0)
            
            pair_info.append({
                'image': pair['image_path'],
                'correct_q': pair['correct']['question'],
                'wrong_q': pair['wrong']['question'],
                'correct_gt': pair['correct']['gt_ans'],
                'wrong_gt': pair['wrong']['gt_ans'],
                'correct_pred': pred_correct,
                'wrong_pred': pred_wrong,
            })
            
            # Clear GPU memory periodically
            if len(correct_labels) % 50 == 0:
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing pair: {e}")
            continue
    
    # Convert to numpy arrays
    correct_representations = [np.array(reps) for reps in correct_representations]
    wrong_representations = [np.array(reps) for reps in wrong_representations]
    correct_labels = np.array(correct_labels)
    wrong_labels = np.array(wrong_labels)
    
    print(f"\nExtracted representations for {len(correct_labels)} pairs")
    print(f"Correct questions - Yes: {correct_labels.sum()}, No: {len(correct_labels) - correct_labels.sum()}")
    print(f"Wrong questions - Yes: {wrong_labels.sum()}, No: {len(wrong_labels) - wrong_labels.sum()}")
    
    return correct_representations, wrong_representations, correct_labels, wrong_labels, pair_info


def train_probing_classifiers(correct_reps, wrong_reps, correct_labels, wrong_labels, test_size=0.2):
    """
    Train logistic regression classifier for each layer.

    Trains on COMBINED data from both correct and wrong questions.
    This tests: "Can layer X predict the ground truth answer?"

    The train/test split is done at the PAIR level so that both questions from
    the same image always end up in the same split (no image-feature leakage).

    Returns:
        results: dict with per-layer metrics for combined, correct-only, and wrong-only
        classifiers: list of trained LogisticRegression classifiers, one per layer
        test_pair_idx: indices into the original pairs list that form the test set
    """
    num_layers = len(correct_reps)
    num_pairs = len(correct_labels)

    # Combine correct and wrong for training
    combined_reps = [np.vstack([correct_reps[i], wrong_reps[i]]) for i in range(num_layers)]
    combined_labels = np.concatenate([correct_labels, wrong_labels])

    # Track which samples are from correct vs wrong questions
    is_correct_question = np.concatenate([
        np.ones(num_pairs),
        np.zeros(num_pairs)
    ])

    # Split at PAIR level so both questions from the same image stay together.
    # Correct questions occupy indices 0..num_pairs-1,
    # wrong questions occupy indices num_pairs..2*num_pairs-1.
    pair_indices = np.arange(num_pairs)
    train_pair_idx, test_pair_idx = train_test_split(
        pair_indices,
        test_size=test_size,
        random_state=42,
        stratify=correct_labels,  # balance yes/no across splits
    )

    train_idx = np.concatenate([train_pair_idx, train_pair_idx + num_pairs])
    test_idx = np.concatenate([test_pair_idx, test_pair_idx + num_pairs])

    y_train = combined_labels[train_idx]
    y_test = combined_labels[test_idx]
    is_correct_test = is_correct_question[test_idx]
    
    print(f"\nTraining probing classifiers...")
    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    
    results = {
        'per_layer': [],
        'num_train': len(train_idx),
        'num_test': len(test_idx),
    }
    classifiers = []
    
    for layer_idx in tqdm(range(num_layers), desc="Training classifiers"):
        X = combined_reps[layer_idx]
        X_train = X[train_idx]
        X_test = X[test_idx]
        
        # Train logistic regression
        clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
        )
        
        try:
            clf.fit(X_train, y_train)
            
            # Predictions
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            
            # Overall metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            # Metrics for correct questions only
            correct_mask = is_correct_test == 1
            if correct_mask.sum() > 0:
                acc_correct = accuracy_score(y_test[correct_mask], y_pred[correct_mask])
            else:
                acc_correct = 0
            
            # Metrics for wrong questions only (split by ground truth label)
            wrong_mask = is_correct_test == 0
            if wrong_mask.sum() > 0:
                acc_wrong = accuracy_score(y_test[wrong_mask], y_pred[wrong_mask])
            else:
                acc_wrong = 0

            wrong_gt_yes = wrong_mask & (y_test == 1)
            if wrong_gt_yes.sum() > 0:
                acc_wrong_gt_yes = accuracy_score(y_test[wrong_gt_yes], y_pred[wrong_gt_yes])
            else:
                acc_wrong_gt_yes = 0

            wrong_gt_no = wrong_mask & (y_test == 0)
            if wrong_gt_no.sum() > 0:
                acc_wrong_gt_no = accuracy_score(y_test[wrong_gt_no], y_pred[wrong_gt_no])
            else:
                acc_wrong_gt_no = 0

            results['per_layer'].append({
                'layer': layer_idx,
                'accuracy': accuracy,
                'auc': auc,
                'accuracy_correct_questions': acc_correct,
                'accuracy_wrong_questions': acc_wrong,
                'accuracy_wrong_gt_yes': acc_wrong_gt_yes,
                'accuracy_wrong_gt_no': acc_wrong_gt_no,
            })
            classifiers.append(clf)
            
        except Exception as e:
            print(f"Error training layer {layer_idx}: {e}")
            results['per_layer'].append({
                'layer': layer_idx,
                'accuracy': 0.5,
                'auc': 0.5,
                'accuracy_correct_questions': 0.5,
                'accuracy_wrong_questions': 0.5,
                'accuracy_wrong_gt_yes': 0.5,
                'accuracy_wrong_gt_no': 0.5,
            })
            classifiers.append(None)
    
    return results, classifiers, test_pair_idx


def save_classifier_weights(classifiers, output_dir):
    """
    Save logistic regression weights layer by layer.
    
    Args:
        classifiers: list of trained LogisticRegression classifiers
        output_dir: directory to save weights
    """
    weights_data = {
        'num_layers': len(classifiers),
        'weights': [],
        'intercepts': [],
    }
    
    for layer_idx, clf in enumerate(classifiers):
        if clf is not None:
            # Save coefficient and intercept
            weights = clf.coef_[0].tolist() if len(clf.coef_.shape) > 1 else clf.coef_.tolist()
            intercept = float(clf.intercept_[0]) if hasattr(clf.intercept_, '__len__') else float(clf.intercept_)
            
            weights_data['weights'].append(weights)
            weights_data['intercepts'].append(intercept)
        else:
            weights_data['weights'].append(None)
            weights_data['intercepts'].append(None)
    
    # Save as numpy arrays for easier loading
    weights_file = os.path.join(output_dir, 'lr_weights.npz')
    
    # Filter out None values for numpy
    valid_weights = [w for w in weights_data['weights'] if w is not None]
    valid_intercepts = [i for i in weights_data['intercepts'] if i is not None]
    
    np.savez(
        weights_file,
        weights=np.array(valid_weights),
        intercepts=np.array(valid_intercepts),
        num_layers=len(classifiers)
    )
    
    print(f"\nSaved classifier weights to: {weights_file}")
    print(f"  Number of layers: {len(classifiers)}")
    print(f"  Weight shape per layer: {np.array(valid_weights[0]).shape if valid_weights else 'N/A'}")
    
    return weights_file


def plot_layer_accuracy(results, output_dir):
    """Plot accuracy and AUC per layer with correct vs wrong breakdown."""
    
    layers = [r['layer'] for r in results['per_layer']]
    accuracies = [r['accuracy'] for r in results['per_layer']]
    aucs = [r['auc'] for r in results['per_layer']]
    acc_correct = [r['accuracy_correct_questions'] for r in results['per_layer']]
    acc_wrong = [r['accuracy_wrong_questions'] for r in results['per_layer']]
    acc_wrong_gt_yes = [r['accuracy_wrong_gt_yes'] for r in results['per_layer']]
    acc_wrong_gt_no = [r['accuracy_wrong_gt_no'] for r in results['per_layer']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overall accuracy
    ax1 = axes[0, 0]
    ax1.plot(layers, accuracies, 'b-o', markersize=4, label='Overall')
    ax1.axhline(y=0.5, color='red', linestyle='--', label='Random baseline')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Probing Accuracy per Layer (Combined)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.4, 1.0])
    
    # 2. AUC
    ax2 = axes[0, 1]
    ax2.plot(layers, aucs, 'g-o', markersize=4)
    ax2.axhline(y=0.5, color='red', linestyle='--', label='Random baseline')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('Probing AUC per Layer', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.4, 1.0])
    
    # 3. Correct vs Wrong questions accuracy
    ax3 = axes[1, 0]
    ax3.plot(layers, acc_correct, 'g-o', markersize=4, label='Correct questions')
    ax3.plot(layers, acc_wrong, 'r-o', markersize=4, label='Wrong questions')
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Layer', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Probing Accuracy: Correct vs Wrong Questions', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.4, 1.0])
    
    # 4. Difference (Correct - Wrong)
    ax4 = axes[1, 1]
    diff = [c - w for c, w in zip(acc_correct, acc_wrong)]
    colors = ['green' if d > 0 else 'red' for d in diff]
    ax4.bar(layers, diff, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Layer', fontsize=12)
    ax4.set_ylabel('Accuracy Difference', fontsize=12)
    ax4.set_title('Accuracy Difference (Correct - Wrong Questions)', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'paired_probing_accuracy.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")

    plt.close()

    # Separate plot: wrong questions split by GT label
    fig2, ax = plt.subplots(figsize=(9, 5))
    ax.plot(layers, acc_wrong_gt_yes, 'm-o', markersize=4, label='Wrong questions (GT=Yes)')
    ax.plot(layers, acc_wrong_gt_no, 'c-o', markersize=4, label='Wrong questions (GT=No)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Probing Accuracy for Incorrectly Answered Questions\nby Ground Truth Label', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    plt.tight_layout()

    split_plot_path = os.path.join(output_dir, 'wrong_questions_gt_split_accuracy.png')
    fig2.savefig(split_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved GT-split plot to: {split_plot_path}")

    plt.close()

    return fig


def analyze_results(results):
    """Analyze probing results to find insights."""
    
    per_layer = results['per_layer']
    accuracies = [r['accuracy'] for r in per_layer]
    acc_correct = [r['accuracy_correct_questions'] for r in per_layer]
    acc_wrong = [r['accuracy_wrong_questions'] for r in per_layer]
    acc_wrong_gt_yes = [r['accuracy_wrong_gt_yes'] for r in per_layer]
    acc_wrong_gt_no = [r['accuracy_wrong_gt_no'] for r in per_layer]
    
    print("\n" + "=" * 60)
    print("PAIRED PROBING ANALYSIS RESULTS")
    print("=" * 60)
    
    # Find best layer
    best_layer = int(np.argmax(accuracies))
    best_acc = accuracies[best_layer]
    
    print(f"\nBest overall accuracy: {best_acc:.4f} at layer {best_layer}")
    print(f"Final layer accuracy: {accuracies[-1]:.4f}")
    
    # Check for accuracy drop
    if best_acc > accuracies[-1] + 0.05:
        print(f"\n⚠️  FINDING: Accuracy drops from {best_acc:.4f} (layer {best_layer}) "
              f"to {accuracies[-1]:.4f} (final layer)")
        print("   This suggests correct information exists in earlier layers but gets corrupted!")
    else:
        print(f"\n✓ No significant accuracy drop detected")
    
    # Compare correct vs wrong questions
    avg_diff = np.mean([c - w for c, w in zip(acc_correct, acc_wrong)])
    print(f"\nAverage accuracy difference (correct - wrong): {avg_diff:.4f}")
    
    if avg_diff > 0.05:
        print("   → Model representations are more predictive for correctly answered questions")
    elif avg_diff < -0.05:
        print("   → Model representations are more predictive for wrongly answered questions (unexpected!)")
    else:
        print("   → Similar predictive power for both question types")
    
    # Find divergence layer
    diffs = [c - w for c, w in zip(acc_correct, acc_wrong)]
    max_diff_layer = int(np.argmax(np.abs(diffs)))
    print(f"\nLargest divergence at layer {max_diff_layer}: {diffs[max_diff_layer]:.4f}")

    # Wrong questions broken down by GT label
    best_wrong_gt_yes_layer = int(np.argmax(acc_wrong_gt_yes))
    best_wrong_gt_no_layer = int(np.argmax(acc_wrong_gt_no))
    print(f"\nWrong questions (GT=Yes) - best probing accuracy: "
          f"{acc_wrong_gt_yes[best_wrong_gt_yes_layer]:.4f} at layer {best_wrong_gt_yes_layer}, "
          f"final layer: {acc_wrong_gt_yes[-1]:.4f}")
    print(f"Wrong questions (GT=No)  - best probing accuracy: "
          f"{acc_wrong_gt_no[best_wrong_gt_no_layer]:.4f} at layer {best_wrong_gt_no_layer}, "
          f"final layer: {acc_wrong_gt_no[-1]:.4f}")

    return {
        'best_layer': best_layer,
        'best_accuracy': float(best_acc),
        'final_accuracy': float(accuracies[-1]),
        'accuracy_drop': float(best_acc - accuracies[-1]),
        'avg_correct_wrong_diff': float(avg_diff),
        'max_divergence_layer': max_diff_layer,
        'max_divergence': float(diffs[max_diff_layer]),
        'wrong_gt_yes_best_layer': best_wrong_gt_yes_layer,
        'wrong_gt_yes_best_accuracy': float(acc_wrong_gt_yes[best_wrong_gt_yes_layer]),
        'wrong_gt_yes_final_accuracy': float(acc_wrong_gt_yes[-1]),
        'wrong_gt_no_best_layer': best_wrong_gt_no_layer,
        'wrong_gt_no_best_accuracy': float(acc_wrong_gt_no[best_wrong_gt_no_layer]),
        'wrong_gt_no_final_accuracy': float(acc_wrong_gt_no[-1]),
    }


def main():
    parser = argparse.ArgumentParser(description='Paired Layer-wise Representation Probing')
    
    parser.add_argument("--model-name", type=str,
                        default="chaoyinshe/llava-med-v1.5-mistral-7b-hf")
    parser.add_argument("--margin-scores-file", type=str, required=True,
                        help="Path to margin_scores.json from VCD experiment")
    parser.add_argument("--test-file", type=str, required=True,
                        help="Path to test.json (for image paths)")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to image folder")
    parser.add_argument("--output-dir", type=str, default="results/paired_probing",
                        help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=500,
                        help="Number of pairs to use")
    parser.add_argument("--load-8bit", action="store_true", default=True,
                        help="Load model in 8-bit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--image-mode", type=str, default="real",
                        choices=["real", "black", "random"],
                        help="Image mode: 'real' uses actual images, 'black' replaces with black "
                             "images, 'random' replaces with random noise (ablation baselines)")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Namespace output dir for ablation modes so results don't overwrite the real run
    if args.image_mode != "real":
        args.output_dir = os.path.join(args.output_dir, f"image_mode_{args.image_mode}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find paired questions
    pairs = find_paired_questions(
        args.margin_scores_file,
        args.test_file,
        args.num_pairs
    )
    
    if not pairs:
        print("No pairs found!")
        return
    
    # Initialize extractor
    extractor = PairedRepresentationExtractor(
        model_name=args.model_name,
        load_8bit=args.load_8bit
    )
    
    # Extract representations for pairs
    correct_reps, wrong_reps, correct_labels, wrong_labels, pair_info = extract_paired_representations(
        extractor, pairs, args.image_folder, image_mode=args.image_mode
    )
    
    if len(correct_labels) < 50:
        print("Not enough pairs extracted!")
        return
    
    # Train probing classifiers
    probing_results, classifiers, test_pair_idx = train_probing_classifiers(
        correct_reps, wrong_reps, correct_labels, wrong_labels
    )

    # Compute model accuracy on the same held-out test pairs
    test_pairs_info = [pair_info[i] for i in test_pair_idx]
    model_acc_correct_q = np.mean([
        1 if p['correct_pred'] == p['correct_gt'] else 0 for p in test_pairs_info
    ])
    model_acc_wrong_q = np.mean([
        1 if p['wrong_pred'] == p['wrong_gt'] else 0 for p in test_pairs_info
    ])
    model_acc_overall = (model_acc_correct_q + model_acc_wrong_q) / 2

    print(f"\nModel accuracy on held-out test pairs ({len(test_pair_idx)} pairs):")
    print(f"  Overall:            {model_acc_overall:.4f}")
    print(f"  Correct questions:  {model_acc_correct_q:.4f}")
    print(f"  Wrong questions:    {model_acc_wrong_q:.4f}")

    probing_results['model_accuracy'] = {
        'overall': float(model_acc_overall),
        'correct_questions': float(model_acc_correct_q),
        'wrong_questions': float(model_acc_wrong_q),
        'num_test_pairs': len(test_pair_idx),
    }

    # Save results
    output_file = os.path.join(args.output_dir, 'probing_results.json')
    with open(output_file, 'w') as f:
        json.dump(probing_results, f, indent=2)
    print(f"Saved probing results to: {output_file}")
    
    # Save classifier weights
    save_classifier_weights(classifiers, args.output_dir)
    
    # Plot results
    plot_layer_accuracy(probing_results, args.output_dir)
    
    # Analyze
    analysis = analyze_results(probing_results)
    
    # Save analysis
    analysis_file = os.path.join(args.output_dir, 'analysis_summary.json')
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to: {analysis_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()