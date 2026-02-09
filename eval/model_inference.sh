#!/bin/bash
set -e

model_name=$1

# ============================================
# CONFIGURATION - Update these paths
# ============================================
question_file="./test.json"  # Path to ProbMed question file
image_folder="./probmed"         # Path to image folder
answer_file="./response_file/${model_name}"
answer_file_json="./response_file/${model_name}.json"

# HuggingFace model names
LLAVAMED_HF="chaoyinshe/llava-med-v1.5-mistral-7b-hf"

# ============================================

echo "=========================================="
echo "Running inference for: ${model_name}"
echo "=========================================="

if [ "${model_name}" == "llavamed" ]; then
    # Using HuggingFace model - no need to cd to LLaVA-Med repo
    python run_med_datasets_eval_batch.py \
        --num-chunks 1 \
        --model-name ${LLAVAMED_HF} \
        --question-file ${question_file} \
        --image-folder ${image_folder} \
        --answers-file ${answer_file}.jsonl \
        --load-8bit
    
    # Convert JSONL to JSON for calculate_score.py
    python -c "
import json
with open('${answer_file}.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${answer_file_json}', 'w') as f:
    json.dump(data, f, indent=2)
print('Converted to JSON: ${answer_file_json}')
"

elif [ "${model_name}" == "llavamed_hf" ]; then
    # Alternative name for HF version
    python run_med_datasets_eval_batch.py \
        --num-chunks 1 \
        --model-name ${LLAVAMED_HF} \
        --question-file ${question_file} \
        --image-folder ${image_folder} \
        --answers-file ${answer_file}.jsonl \
        --load-8bit
    
    python -c "
import json
with open('${answer_file}.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${answer_file_json}', 'w') as f:
    json.dump(data, f, indent=2)
"

# ============================================
# Keep other models as-is (or comment out if not needed)
# ============================================

elif [ "${model_name}" == "llava_v1" ]; then
    cd ../LLaVA
    python llava/eval/run_eval_batch.py \
        --num-chunks 4 \
        --model-name /model_weights/llava/llava_v1 \
        --image-folder ${image_folder} \
        --question-file ${question_file} \
        --answers-file ${answer_file}
    rm ${answer_file}-*

elif [ "${model_name}" == "llava_v1.6" ]; then
    cd ../LLaVA
    python llava/eval/run_eval_batch.py \
        --num-chunks 4 \
        --model-name /model_weights/llava/llava-v1.6-vicuna-7b \
        --image-folder ${image_folder} \
        --question-file ${question_file} \
        --answers-file ${answer_file}
    rm ${answer_file}-*

elif [ "${model_name}" == "minigptv2" ]; then
    cd ../MiniGPT-4
    python run_eval_batch.py \
        --num-chunks 4 \
        --cfg-path eval_configs/minigptv2_eval.yaml \
        --image-folder ${image_folder} \
        --question-file ${question_file} \
        --answers-file ${answer_file}
    rm ${answer_file}-*

elif [ "${model_name}" == "chexagent" ]; then
    cd ../CheXagent
    python run_eval_batch.py \
        --num-chunks 4 \
        --image-folder ${image_folder} \
        --question-file ${question_file} \
        --answers-file ${answer_file}
    rm ${answer_file}-*

elif [ "${model_name}" == "gpt4v" ]; then
    cd ../gpt4V
    python gpt4v.py \
        --image-folder ${image_folder} \
        --question-file ${question_file} \
        --answers-file ${answer_file_json}

elif [ "${model_name}" == "gemini" ]; then
    cd ../gemini
    python run_eval_batch.py \
        --num-chunks 4 \
        --image-folder ${image_folder} \
        --question-file ${question_file} \
        --answers-file ${answer_file}
    rm ${answer_file}-*
    rm ${answer_file}_*

elif [ "${model_name}" == "gpt4o" ]; then
    cd ../gpt4V
    python gpt4v.py \
        --image-folder ${image_folder} \
        --question-file ${question_file} \
        --answers-file ${answer_file_json}

elif [ "${model_name}" == "med-flamingo" ]; then
    cd ../med-flamingo
    python scripts/run_eval_batch.py \
        --num-chunks 3 \
        --question-file ${question_file} \
        --answers-file ${answer_file}
    rm ${answer_file}-*

elif [ "${model_name}" == "biomedgpt" ]; then
    cd ../BiomedGPT
    python evaluate.py \
        ablation.tsv \
        --path /model_weights/biomedgpt_base.pt \
        --user-dir module \
        --task vqa_gen \
        --batch-size 64 \
        --log-format simple --log-interval 10 \
        --seed 7 \
        --gen-subset ablation \
        --results-path ../ablation \
        --fp16 \
        --beam-search-vqa-eval \
        --ema-eval \
        --unnormalized \
        --temperature 1.0 \
        --num-workers 0

else
    echo "Unknown model: ${model_name}"
    echo "Available models: llavamed, llavamed_hf, llava_v1, llava_v1.6, minigptv2, chexagent, gpt4v, gemini, gpt4o, med-flamingo, biomedgpt"
    exit 1
fi

echo "=========================================="
echo "Done: ${model_name}"
echo "=========================================="