#!/bin/bash
set -e

model_name=$1

# ============================================
# INSTALL DEPENDENCIES
# ============================================
echo "Installing dependencies..."
pip install -q transformers accelerate bitsandbytes pillow tqdm

# ============================================
# CONFIGURATION - Update these paths
# ============================================
question_file="/workspace/ProbMed-Dataset/test/test.json"
image_folder="/workspace/ProbMed-Dataset/test/"
answer_file="./response_file/${model_name}"
answer_file_json="./response_file/${model_name}.json"

# HuggingFace model names
LLAVAMED_HF="chaoyinshe/llava-med-v1.5-mistral-7b-hf"
CHEXAGENT_HF="StanfordAIMI/CheXagent"

# Path to inference scripts
LLAVAMED_INFERENCE_DIR="./inference/LLaVA-Med"
CHEXAGENT_INFERENCE_DIR="./inference/CheXagent"

# ============================================

echo "=========================================="
echo "Running inference for: ${model_name}"
echo "=========================================="

if [ "${model_name}" == "llavamed" ]; then
    # Using HuggingFace model
    python ${LLAVAMED_INFERENCE_DIR}/run_med_datasets_eval_batch.py \
        --num-chunks 4 \
        --model-name ${LLAVAMED_HF} \
        --question-file ${question_file} \
        --image-folder ${image_folder} \
        --answers-file ${answer_file}.jsonl \
        --batch-size 1 \
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
    python ${LLAVAMED_INFERENCE_DIR}/run_med_datasets_eval_batch.py \
        --num-chunks 4 \
        --model-name ${LLAVAMED_HF} \
        --question-file ${question_file} \
        --image-folder ${image_folder} \
        --answers-file ${answer_file}.jsonl \
        --batch-size 1 \
        --load-8bit
    
    python -c "
import json
with open('${answer_file}.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${answer_file_json}', 'w') as f:
    json.dump(data, f, indent=2)
"

elif [ "${model_name}" == "chexagent" ]; then
    python ${CHEXAGENT_INFERENCE_DIR}/run_eval_batch.py \
        --num-chunks 4 \
        --model-name ${CHEXAGENT_HF} \
        --question-file ${question_file} \
        --image-folder ${image_folder} \
        --answers-file ${answer_file}.jsonl

    # Convert JSONL to JSON for calculate_score.py
    python -c "
import json
with open('${answer_file}.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${answer_file_json}', 'w') as f:
    json.dump(data, f, indent=2)
print('Converted to JSON: ${answer_file_json}')
"

else
    echo "Unknown model: ${model_name}"
    echo "Available models: llavamed, llavamed_hf, chexagent"
    exit 1
fi

echo "=========================================="
echo "Done: ${model_name}"
echo "=========================================="