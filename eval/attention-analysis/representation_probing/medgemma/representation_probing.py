import argparse
import json
import math
import os
import random
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class MedGemmaRepresentationExtractor:
    """Extracts hidden representations from each layer of MedGemma."""

    def __init__(self, model_name="google/medgemma-1.5-4b-it", load_8bit=False):
        print(f"Loading model: {model_name}")
        if load_8bit:
            print("Note: --load-8bit is ignored for MedGemma; loading in bfloat16.")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()

        self.num_layers = len(self.model.model.language_model.layers)
        self.hidden_size = self.model.config.text_config.hidden_size
        self.yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]

        print(f"Model loaded! Layers: {self.num_layers}, Hidden size: {self.hidden_size}")

    @property
    def device(self):
        return next(self.model.parameters()).device

    def prepare_image(self, image_path, image_mode):
        image = Image.open(image_path).convert("RGB")
        if image_mode == "black":
            return Image.new("RGB", image.size, (0, 0, 0))
        if image_mode == "random":
            arr = np.random.randint(0, 256, (image.size[1], image.size[0], 3), dtype=np.uint8)
            return Image.fromarray(arr)
        return image

    def cleanup_image(self, prepared_image):
        pass

    def extract_layer_representations(self, image, question):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        representations = [
            h[0, -1, :].detach().cpu().to(torch.float32).numpy()
            for h in outputs.hidden_states
        ]

        logits = outputs.logits[:, -1, :]
        prediction = 'yes' if logits[0, self.yes_token_id].item() > logits[0, self.no_token_id].item() else 'no'

        return representations, prediction


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k] if k < len(chunks) else []


def get_score_binary(response, ans):
    response = response.strip()
    if ans == 'yes':
        return 1 if ('Yes' in response or response.lower() in ('yes', 'yes.')) else 0
    else:
        return 1 if ('No' in response or response.lower() in ('no', 'no.')) else 0


def load_results(margin_scores_file=None, response_file=None):
    if response_file:
        print(f"Loading response file: {response_file}")
        with open(response_file, 'r') as f:
            first = f.read(1)
            f.seek(0)
            results = json.load(f) if first == '[' else [json.loads(l) for l in f if l.strip()]
        for r in results:
            if 'gt_ans' in r:
                r['gt_ans'] = r['gt_ans'].lower().strip()
            if 'question' in r:
                r['question'] = r['question'].replace('<image>', '').strip()
            if 'is_correct' not in r:
                r['is_correct'] = get_score_binary(r.get('response', ''), r.get('gt_ans', '')) == 1
    else:
        print(f"Loading margin scores from: {margin_scores_file}")
        with open(margin_scores_file, 'r') as f:
            results = json.load(f)
        for r in results:
            if 'gt_ans' in r:
                r['gt_ans'] = r['gt_ans'].lower().strip()
            if 'question' in r:
                r['question'] = r['question'].replace('<image>', '').strip()
    return results


def find_paired_questions(results, test_file, num_pairs=500):
    print(f"Total samples: {len(results)}")
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    id_question_to_image = {}
    for item in test_data:
        q = item.get('question', '').replace('<image>', '').strip()
        if item.get('id') is not None and 'image' in item:
            id_question_to_image[(item['id'], q)] = item['image']
    print(f"Loaded {len(id_question_to_image)} question-to-image mappings")

    by_image = defaultdict(list)
    for r in results:
        if 'gt_ans' in r:
            r['gt_ans'] = r['gt_ans'].lower().strip()
        if 'question' in r:
            r['question'] = r['question'].replace('<image>', '').strip()
        image_path = id_question_to_image.get((r.get('id'), r.get('question', '')))
        if image_path:
            r['image'] = image_path
            by_image[image_path].append(r)
    print(f"Unique images: {len(by_image)}")

    pairs = []
    for img_path, questions in by_image.items():
        correct = [q for q in questions if q.get('is_correct', False)]
        wrong = [q for q in questions if not q.get('is_correct', False)]
        if correct and wrong:
            pairs.append({'image_path': img_path, 'correct': correct[0], 'wrong': wrong[0]})

    print(f"Found {len(pairs)} valid pairs")
    if len(pairs) > num_pairs:
        random.seed(42)
        pairs = random.sample(pairs, num_pairs)
        print(f"Sampled {num_pairs} pairs")
    return pairs


def extract_paired_representations(extractor, pairs, image_folder, image_mode="real"):
    assert image_mode in ("real", "black", "random")
    num_layers = extractor.num_layers + 1

    correct_representations = [[] for _ in range(num_layers)]
    wrong_representations = [[] for _ in range(num_layers)]
    correct_labels = []
    wrong_labels = []
    pair_info = []

    for pair in tqdm(pairs, desc="Extracting paired representations"):
        image_path = os.path.join(image_folder, pair['image_path'])
        if not os.path.exists(image_path):
            print(f"Image not found, skipping: {image_path}")
            continue

        prepared_image = None
        try:
            prepared_image = extractor.prepare_image(image_path, image_mode)
            repr_correct, pred_correct = extractor.extract_layer_representations(
                prepared_image, pair['correct']['question'])
            repr_wrong, pred_wrong = extractor.extract_layer_representations(
                prepared_image, pair['wrong']['question'])

            for i in range(num_layers):
                correct_representations[i].append(repr_correct[i])
                wrong_representations[i].append(repr_wrong[i])

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
            if len(correct_labels) % 50 == 0:
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing pair: {e}")
        finally:
            if prepared_image is not None:
                extractor.cleanup_image(prepared_image)

    correct_representations = [np.array(r) for r in correct_representations]
    wrong_representations = [np.array(r) for r in wrong_representations]
    correct_labels = np.array(correct_labels)
    wrong_labels = np.array(wrong_labels)
    print(f"\nExtracted {len(correct_labels)} pairs")
    return correct_representations, wrong_representations, correct_labels, wrong_labels, pair_info


def save_extraction_chunk(output_dir, chunk_idx, correct_reps, wrong_reps,
                          correct_labels, wrong_labels, pair_info):
    chunk_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    arrays = {
        "correct_labels": correct_labels,
        "wrong_labels": wrong_labels,
        "num_layers": np.array([len(correct_reps)], dtype=np.int32),
    }
    for i, reps in enumerate(correct_reps):
        arrays[f"correct_layer_{i}"] = reps
        arrays[f"wrong_layer_{i}"] = wrong_reps[i]
    np.savez(os.path.join(chunk_dir, f"chunk{chunk_idx}.npz"), **arrays)
    with open(os.path.join(chunk_dir, f"chunk{chunk_idx}_pair_info.json"), "w") as f:
        json.dump(pair_info, f, indent=2)
    print(f"Saved chunk {chunk_idx} to {chunk_dir}")


def load_extraction_chunks(output_dir, num_chunks):
    chunk_dir = os.path.join(output_dir, "chunks")
    correct_by_layer = None
    wrong_by_layer = None
    correct_labels, wrong_labels, pair_info = [], [], []

    for idx in range(num_chunks):
        chunk_file = os.path.join(chunk_dir, f"chunk{idx}.npz")
        info_file = os.path.join(chunk_dir, f"chunk{idx}_pair_info.json")
        if not os.path.exists(chunk_file):
            raise FileNotFoundError(f"Missing chunk file: {chunk_file}")
        data = np.load(chunk_file)
        num_layers = int(data["num_layers"][0])
        if correct_by_layer is None:
            correct_by_layer = [[] for _ in range(num_layers)]
            wrong_by_layer = [[] for _ in range(num_layers)]
        for i in range(num_layers):
            correct_by_layer[i].append(data[f"correct_layer_{i}"])
            wrong_by_layer[i].append(data[f"wrong_layer_{i}"])
        correct_labels.append(data["correct_labels"])
        wrong_labels.append(data["wrong_labels"])
        if os.path.exists(info_file):
            with open(info_file) as f:
                pair_info.extend(json.load(f))
        print(f"Loaded chunk {idx}: {len(data['correct_labels'])} pairs")

    correct_reps = [np.concatenate(c, axis=0) for c in correct_by_layer]
    wrong_reps = [np.concatenate(w, axis=0) for w in wrong_by_layer]
    correct_labels = np.concatenate(correct_labels, axis=0)
    wrong_labels = np.concatenate(wrong_labels, axis=0)
    print(f"Merged: {len(correct_labels)} pairs, {len(correct_reps)} layers")
    return correct_reps, wrong_reps, correct_labels, wrong_labels, pair_info


def train_probing_classifiers(correct_reps, wrong_reps, correct_labels, wrong_labels, test_size=0.2):
    num_layers = len(correct_reps)
    num_pairs = len(correct_labels)
    combined_reps = [np.vstack([correct_reps[i], wrong_reps[i]]) for i in range(num_layers)]
    combined_labels = np.concatenate([correct_labels, wrong_labels])
    is_correct_question = np.concatenate([np.ones(num_pairs), np.zeros(num_pairs)])

    train_pair_idx, test_pair_idx = train_test_split(
        np.arange(num_pairs), test_size=test_size, random_state=42, stratify=correct_labels)
    train_idx = np.concatenate([train_pair_idx, train_pair_idx + num_pairs])
    test_idx = np.concatenate([test_pair_idx, test_pair_idx + num_pairs])
    y_train = combined_labels[train_idx]
    y_test = combined_labels[test_idx]
    is_correct_test = is_correct_question[test_idx]

    print(f"\nTraining probing classifiers... train={len(train_idx)}, test={len(test_idx)}")
    results = {'per_layer': [], 'num_train': len(train_idx), 'num_test': len(test_idx)}
    classifiers = []

    for layer_idx in tqdm(range(num_layers), desc="Training classifiers"):
        X = combined_reps[layer_idx]
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
            clf.fit(X[train_idx], y_train)
            y_pred = clf.predict(X[test_idx])
            y_prob = clf.predict_proba(X[test_idx])[:, 1]
            cm = is_correct_test == 1
            wm = is_correct_test == 0
            results['per_layer'].append({
                'layer': layer_idx,
                'accuracy': accuracy_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob),
                'accuracy_correct_questions': accuracy_score(y_test[cm], y_pred[cm]) if cm.sum() > 0 else 0,
                'accuracy_wrong_questions': accuracy_score(y_test[wm], y_pred[wm]) if wm.sum() > 0 else 0,
                'accuracy_wrong_gt_yes': accuracy_score(y_test[wm & (y_test==1)], y_pred[wm & (y_test==1)]) if (wm & (y_test==1)).sum() > 0 else 0,
                'accuracy_wrong_gt_no': accuracy_score(y_test[wm & (y_test==0)], y_pred[wm & (y_test==0)]) if (wm & (y_test==0)).sum() > 0 else 0,
            })
            classifiers.append(clf)
        except Exception as e:
            print(f"Error at layer {layer_idx}: {e}")
            results['per_layer'].append({'layer': layer_idx, 'accuracy': 0.5, 'auc': 0.5,
                'accuracy_correct_questions': 0.5, 'accuracy_wrong_questions': 0.5,
                'accuracy_wrong_gt_yes': 0.5, 'accuracy_wrong_gt_no': 0.5})
            classifiers.append(None)

    return results, classifiers, test_pair_idx


def save_classifier_weights(classifiers, output_dir):
    valid_weights = [clf.coef_[0].tolist() for clf in classifiers if clf is not None]
    valid_intercepts = [float(clf.intercept_[0]) for clf in classifiers if clf is not None]
    weights_file = os.path.join(output_dir, 'lr_weights.npz')
    np.savez(weights_file, weights=np.array(valid_weights),
             intercepts=np.array(valid_intercepts), num_layers=len(classifiers))
    print(f"Saved classifier weights to: {weights_file}")
    return weights_file


def plot_layer_accuracy(results, output_dir):
    layers = [r['layer'] for r in results['per_layer']]
    accuracies = [r['accuracy'] for r in results['per_layer']]
    aucs = [r['auc'] for r in results['per_layer']]
    acc_correct = [r['accuracy_correct_questions'] for r in results['per_layer']]
    acc_wrong = [r['accuracy_wrong_questions'] for r in results['per_layer']]
    acc_wrong_gt_yes = [r['accuracy_wrong_gt_yes'] for r in results['per_layer']]
    acc_wrong_gt_no = [r['accuracy_wrong_gt_no'] for r in results['per_layer']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, y, title, color in [
        (axes[0,0], accuracies, 'Probing Accuracy per Layer (Combined)', 'b'),
        (axes[0,1], aucs, 'Probing AUC per Layer', 'g'),
    ]:
        ax.plot(layers, y, f'{color}-o', markersize=4)
        ax.axhline(y=0.5, color='red', linestyle='--', label='Random baseline')
        ax.set_title(title, fontsize=14); ax.set_ylim([0.4, 1.0]); ax.grid(True, alpha=0.3); ax.legend()

    axes[1,0].plot(layers, acc_correct, 'g-o', markersize=4, label='Correct questions')
    axes[1,0].plot(layers, acc_wrong, 'r-o', markersize=4, label='Wrong questions')
    axes[1,0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1,0].set_title('Probing Accuracy: Correct vs Wrong', fontsize=14)
    axes[1,0].set_ylim([0.4, 1.0]); axes[1,0].grid(True, alpha=0.3); axes[1,0].legend()

    diff = [c - w for c, w in zip(acc_correct, acc_wrong)]
    axes[1,1].bar(layers, diff, color=['green' if d > 0 else 'red' for d in diff], alpha=0.7)
    axes[1,1].axhline(y=0, color='black', linewidth=0.5)
    axes[1,1].set_title('Accuracy Difference (Correct - Wrong)', fontsize=14)
    axes[1,1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paired_probing_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig2, ax = plt.subplots(figsize=(9, 5))
    ax.plot(layers, acc_wrong_gt_yes, 'm-o', markersize=4, label='Wrong (GT=Yes)')
    ax.plot(layers, acc_wrong_gt_no, 'c-o', markersize=4, label='Wrong (GT=No)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.set_title('Probing Accuracy for Wrong Questions by GT Label', fontsize=14)
    ax.set_ylim([0.4, 1.0]); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wrong_questions_gt_split_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()


def analyze_results(results):
    per_layer = results['per_layer']
    accuracies = [r['accuracy'] for r in per_layer]
    acc_correct = [r['accuracy_correct_questions'] for r in per_layer]
    acc_wrong = [r['accuracy_wrong_questions'] for r in per_layer]
    acc_wrong_gt_yes = [r['accuracy_wrong_gt_yes'] for r in per_layer]
    acc_wrong_gt_no = [r['accuracy_wrong_gt_no'] for r in per_layer]

    best_layer = int(np.argmax(accuracies))
    best_acc = accuracies[best_layer]
    avg_diff = np.mean([c - w for c, w in zip(acc_correct, acc_wrong)])
    diffs = [c - w for c, w in zip(acc_correct, acc_wrong)]
    max_diff_layer = int(np.argmax(np.abs(diffs)))

    print(f"\nBest accuracy: {best_acc:.4f} at layer {best_layer}, final: {accuracies[-1]:.4f}")
    print(f"Avg correct-wrong diff: {avg_diff:.4f}, max divergence at layer {max_diff_layer}: {diffs[max_diff_layer]:.4f}")

    bwgy = int(np.argmax(acc_wrong_gt_yes))
    bwgn = int(np.argmax(acc_wrong_gt_no))
    return {
        'best_layer': best_layer, 'best_accuracy': float(best_acc),
        'final_accuracy': float(accuracies[-1]),
        'accuracy_drop': float(best_acc - accuracies[-1]),
        'avg_correct_wrong_diff': float(avg_diff),
        'max_divergence_layer': max_diff_layer, 'max_divergence': float(diffs[max_diff_layer]),
        'wrong_gt_yes_best_layer': bwgy, 'wrong_gt_yes_best_accuracy': float(acc_wrong_gt_yes[bwgy]),
        'wrong_gt_yes_final_accuracy': float(acc_wrong_gt_yes[-1]),
        'wrong_gt_no_best_layer': bwgn, 'wrong_gt_no_best_accuracy': float(acc_wrong_gt_no[bwgn]),
        'wrong_gt_no_final_accuracy': float(acc_wrong_gt_no[-1]),
    }


def train_and_save_results(correct_reps, wrong_reps, correct_labels, wrong_labels, pair_info, output_dir):
    if len(correct_labels) < 50:
        print("Not enough pairs extracted!")
        return

    probing_results, classifiers, test_pair_idx = train_probing_classifiers(
        correct_reps, wrong_reps, correct_labels, wrong_labels)

    test_pairs_info = [pair_info[i] for i in test_pair_idx]
    acc_correct_q = np.mean([1 if p['correct_pred'] == p['correct_gt'] else 0 for p in test_pairs_info])
    acc_wrong_q = np.mean([1 if p['wrong_pred'] == p['wrong_gt'] else 0 for p in test_pairs_info])
    print(f"\nModel accuracy — correct Q: {acc_correct_q:.4f}, wrong Q: {acc_wrong_q:.4f}")

    probing_results['model_accuracy'] = {
        'overall': float((acc_correct_q + acc_wrong_q) / 2),
        'correct_questions': float(acc_correct_q),
        'wrong_questions': float(acc_wrong_q),
        'num_test_pairs': len(test_pair_idx),
    }

    with open(os.path.join(output_dir, 'probing_results.json'), 'w') as f:
        json.dump(probing_results, f, indent=2)
    save_classifier_weights(classifiers, output_dir)
    plot_layer_accuracy(probing_results, output_dir)
    analysis = analyze_results(probing_results)
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    print("Done!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MedGemma Representation Probing')
    parser.add_argument("--model-name", type=str, default="google/medgemma-1.5-4b-it")

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--margin-scores-file", type=str)
    grp.add_argument("--response-file", type=str)

    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/representation_probing/medgemma")
    parser.add_argument("--num-pairs", type=int, default=500)
    parser.add_argument("--load-8bit", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-mode", type=str, default="real", choices=["real", "black", "random"])
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=None)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--train-from-chunks", action="store_true")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.image_mode != "real":
        args.output_dir = os.path.join(args.output_dir, f"image_mode_{args.image_mode}")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.train_from_chunks:
        correct_reps, wrong_reps, correct_labels, wrong_labels, pair_info = load_extraction_chunks(
            args.output_dir, args.num_chunks)
        train_and_save_results(correct_reps, wrong_reps, correct_labels, wrong_labels,
                               pair_info, args.output_dir)
        return

    results = load_results(margin_scores_file=args.margin_scores_file, response_file=args.response_file)
    pairs = find_paired_questions(results, args.test_file, args.num_pairs)
    if not pairs:
        print("No pairs found!")
        return

    if args.num_chunks > 1:
        if args.chunk_idx is None:
            raise ValueError("--chunk-idx required when --num-chunks > 1")
        pairs = get_chunk(pairs, args.num_chunks, args.chunk_idx)
        print(f"Chunk {args.chunk_idx}/{args.num_chunks}: processing {len(pairs)} pairs")
        np.random.seed(args.seed + args.chunk_idx)

    extractor = MedGemmaRepresentationExtractor(model_name=args.model_name, load_8bit=args.load_8bit)
    correct_reps, wrong_reps, correct_labels, wrong_labels, pair_info = extract_paired_representations(
        extractor, pairs, args.image_folder, image_mode=args.image_mode)

    if args.extract_only:
        if args.chunk_idx is None:
            raise ValueError("--extract-only requires --chunk-idx")
        save_extraction_chunk(args.output_dir, args.chunk_idx, correct_reps, wrong_reps,
                              correct_labels, wrong_labels, pair_info)
        return

    train_and_save_results(correct_reps, wrong_reps, correct_labels, wrong_labels,
                           pair_info, args.output_dir)


if __name__ == "__main__":
    main()
