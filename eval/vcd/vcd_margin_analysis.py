"""
VCD Margin Score Analysis for ProbMed (Multi-GPU Support)
==========================================================

Computes margin scores to detect hallucinations:
g = [log p(Yes|v,q) - log p(Yes|v',q)] - [log p(No|v,q) - log p(No|v',q)]

- Large g → visually grounded (genuine)
- Small g → model prior bias (hallucinated)

Single GPU usage:
    python vcd_margin_analysis.py \
        --question-file /path/to/probmed.json \
        --image-folder /path/to/images \
        --output-file results/margin_scores.json \
        --sample-ratio 0.3

Multi-GPU usage (called by run_vcd_analysis_batch.py):
    CUDA_VISIBLE_DEVICES=0 python vcd_margin_analysis.py \
        --question-file /path/to/probmed.json \
        --image-folder /path/to/images \
        --output-file results/margin_scores-chunk0.json \
        --num-chunks 4 \
        --chunk-idx 0 \
        --sample-ratio 0.3
"""

import argparse
import json
import os
import random
import math
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """Get chunk k out of n chunks"""
    chunks = split_list(lst, n)
    return chunks[k] if k < len(chunks) else []


class VCDMarginAnalyzer:
    """Computes VCD margin scores for hallucination detection."""
    
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
        
        # Get Yes/No token IDs
        self.yes_token_id = self._get_token_id("Yes")
        self.no_token_id = self._get_token_id("No")
        
        print(f"Model loaded! Yes token: {self.yes_token_id}, No token: {self.no_token_id}")
    
    def _get_token_id(self, word):
        """Get token ID for a word."""
        tokens = self.processor.tokenizer.encode(word, add_special_tokens=False)
        return tokens[0]
    
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
    
    def downsample_upsample(self, pixel_values, scale=0.5):
        """Degrade image by downsampling then upsampling."""
        original_size = pixel_values.shape[-2:]
        
        # Downsample
        small = F.interpolate(pixel_values, scale_factor=scale, mode='bilinear', align_corners=False)
        
        # Upsample back
        degraded = F.interpolate(small, size=original_size, mode='bilinear', align_corners=False)
        
        return degraded
    
    def get_yes_no_logits(self, inputs):
        """Get Yes/No logits from model outputs."""
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last token logits
            
            log_probs = F.log_softmax(logits[0], dim=-1)
            
            return {
                'yes_logit': logits[0, self.yes_token_id].item(),
                'no_logit': logits[0, self.no_token_id].item(),
                'log_p_yes': log_probs[self.yes_token_id].item(),
                'log_p_no': log_probs[self.no_token_id].item(),
            }
    
    def compute_margin_score(self, image, question, downsample_scale=0.5):
        """
        Compute VCD margin score.
        
        g = [log p(Yes|v,q) - log p(Yes|v',q)] - [log p(No|v,q) - log p(No|v',q)]
        """
        prompt = self.format_prompt(question)
        
        # Process image
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        pixel_values_clean = inputs['pixel_values']
        pixel_values_degraded = self.downsample_upsample(pixel_values_clean, scale=downsample_scale)
        
        # Move to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        pixel_values_degraded = pixel_values_degraded.to(self.device)
        
        # Forward pass with clean image
        logits_clean = self.get_yes_no_logits(inputs)
        
        # Forward pass with degraded image
        inputs_degraded = {k: v.clone() if hasattr(v, 'clone') else v for k, v in inputs.items()}
        inputs_degraded['pixel_values'] = pixel_values_degraded
        logits_degraded = self.get_yes_no_logits(inputs_degraded)
        
        # Compute margin score
        # g = [log p(Yes|v,q) - log p(Yes|v',q)] - [log p(No|v,q) - log p(No|v',q)]
        yes_diff = logits_clean['log_p_yes'] - logits_degraded['log_p_yes']
        no_diff = logits_clean['log_p_no'] - logits_degraded['log_p_no']
        margin_g = yes_diff - no_diff
        
        return {
            'margin_g': margin_g,
            'yes_diff': yes_diff,
            'no_diff': no_diff,
            'log_p_yes_clean': logits_clean['log_p_yes'],
            'log_p_no_clean': logits_clean['log_p_no'],
            'log_p_yes_degraded': logits_degraded['log_p_yes'],
            'log_p_no_degraded': logits_degraded['log_p_no'],
            'yes_logit_clean': logits_clean['yes_logit'],
            'no_logit_clean': logits_clean['no_logit'],
        }
    
    def get_model_prediction(self, logits_clean):
        """Determine if model predicts Yes or No."""
        if logits_clean['yes_logit'] > logits_clean['no_logit']:
            return 'yes'
        else:
            return 'no'


def filter_yes_no_questions(data):
    """Filter questions that have yes/no ground truth answers."""
    filtered = []
    for item in data:
        gt_ans = item.get('answer', item.get('gt_ans', '')).lower().strip()
        if gt_ans in ['yes', 'no']:
            filtered.append(item)
    return filtered


def run_analysis(args):
    """Run VCD margin analysis on ProbMed data."""
    
    # Load data
    print(f"Loading questions from: {args.question_file}")
    with open(args.question_file, 'r') as f:
        data = json.load(f)
    
    # Filter to yes/no questions only
    data = filter_yes_no_questions(data)
    print(f"Found {len(data)} yes/no questions")
    
    # Sample if needed (do this BEFORE chunking for consistency)
    if args.sample_ratio < 1.0:
        sample_size = int(len(data) * args.sample_ratio)
        random.seed(args.seed)
        data = random.sample(data, sample_size)
        print(f"Sampled {len(data)} questions ({args.sample_ratio*100:.0f}%)")
    
    # Get this chunk's portion
    data = get_chunk(data, args.num_chunks, args.chunk_idx)
    print(f"Chunk {args.chunk_idx}/{args.num_chunks}: processing {len(data)} questions")
    
    if len(data) == 0:
        print("No data in this chunk, exiting.")
        return []
    
    # Initialize analyzer
    analyzer = VCDMarginAnalyzer(
        model_name=args.model_name,
        load_8bit=args.load_8bit
    )
    
    # Process questions
    results = []
    
    for item in tqdm(data, desc="Computing margin scores"):
        image_path = os.path.join(args.image_folder, item['image'])
        question = item.get('question', '').replace('<image>', '').strip()
        gt_ans = item.get('answer', item.get('gt_ans', '')).lower().strip()
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            margin_result = analyzer.compute_margin_score(
                image, question,
                downsample_scale=args.downsample_scale
            )
            
            # Determine model's prediction
            model_pred = 'yes' if margin_result['yes_logit_clean'] > margin_result['no_logit_clean'] else 'no'
            is_correct = (model_pred == gt_ans)
            
            results.append({
                'id': item.get('id'),
                'question': question,
                'gt_ans': gt_ans,
                'model_pred': model_pred,
                'is_correct': is_correct,
                'qa_type': item.get('qa_type', 'unknown'),
                'image_type': item.get('image_type', 'unknown'),
                'margin_g': margin_result['margin_g'],
                'yes_diff': margin_result['yes_diff'],
                'no_diff': margin_result['no_diff'],
                'log_p_yes_clean': margin_result['log_p_yes_clean'],
                'log_p_no_clean': margin_result['log_p_no_clean'],
                'log_p_yes_degraded': margin_result['log_p_yes_degraded'],
                'log_p_no_degraded': margin_result['log_p_no_degraded'],
            })
            
        except Exception as e:
            print(f"Error processing {item.get('id')}: {e}")
            continue
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to: {args.output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='VCD Margin Score Analysis')
    
    parser.add_argument("--model-name", type=str,
                        default="chaoyinshe/llava-med-v1.5-mistral-7b-hf")
    parser.add_argument("--question-file", type=str, required=True,
                        help="Path to ProbMed question JSON")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to image folder")
    parser.add_argument("--output-file", type=str, default="results/margin_scores.json",
                        help="Output file for results")
    parser.add_argument("--sample-ratio", type=float, default=0.3,
                        help="Ratio of data to sample (0.3 = 30%)")
    parser.add_argument("--downsample-scale", type=float, default=0.5,
                        help="Downsampling scale for image degradation")
    parser.add_argument("--load-8bit", action="store_true", default=True,
                        help="Load model in 8-bit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Number of chunks for parallel processing")
    parser.add_argument("--chunk-idx", type=int, default=0,
                        help="Which chunk to process (0-indexed)")
    
    args = parser.parse_args()
    
    run_analysis(args)


if __name__ == "__main__":
    main()