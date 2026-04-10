#!/bin/bash
set -e

model_name=$1

# Use venv Python if available (has torch/transformers installed)
PYTHON=/venv/main/bin/python3
if [ ! -f "$PYTHON" ]; then
    PYTHON=python3
fi

# ============================================
# INSTALL DEPENDENCIES
# ============================================
echo "Installing dependencies..."
$PYTHON -m pip install -q transformers accelerate bitsandbytes pillow tqdm einops

# ============================================
# CONFIGURATION - Update these paths
# ============================================
question_file="/workspace/ProbMed-Dataset/test/test.json"
image_folder="/workspace/ProbMed-Dataset/test/"
answer_file="./response_file/${model_name}"
answer_file_json="./response_file/${model_name}.json"

# HuggingFace model names
LLAVAMED_HF="chaoyinshe/llava-med-v1.5-mistral-7b-hf"
CHEXAGENT_HF="StanfordAIMI/CheXagent-2-3b"
MEDGEMMA_HF="google/medgemma-1.5-4b-it"

# Path to inference scripts
LLAVAMED_INFERENCE_DIR="./inference/LLaVA-Med"
CHEXAGENT_INFERENCE_DIR="./inference/CheXagent"
MEDGEMMA_INFERENCE_DIR="./inference/MedGemma"

# ============================================

echo "=========================================="
echo "Running inference for: ${model_name}"
echo "=========================================="

if [ "${model_name}" == "llavamed" ]; then
    # Using HuggingFace model
    $PYTHON ${LLAVAMED_INFERENCE_DIR}/run_med_datasets_eval_batch.py \
        --num-chunks 4 \
        --model-name ${LLAVAMED_HF} \
        --question-file ${question_file} \
        --image-folder ${image_folder} \
        --answers-file ${answer_file}.jsonl \
        --batch-size 1 \
        --load-8bit
    
    # Convert JSONL to JSON for calculate_score.py
    $PYTHON -c "
import json
with open('${answer_file}.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${answer_file_json}', 'w') as f:
    json.dump(data, f, indent=2)
print('Converted to JSON: ${answer_file_json}')
"

elif [ "${model_name}" == "llavamed_hf" ]; then
    # Alternative name for HF version
    $PYTHON ${LLAVAMED_INFERENCE_DIR}/run_med_datasets_eval_batch.py \
        --num-chunks 4 \
        --model-name ${LLAVAMED_HF} \
        --question-file ${question_file} \
        --image-folder ${image_folder} \
        --answers-file ${answer_file}.jsonl \
        --batch-size 1 \
        --load-8bit
    
    $PYTHON -c "
import json
with open('${answer_file}.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${answer_file_json}', 'w') as f:
    json.dump(data, f, indent=2)
"

elif [ "${model_name}" == "chexagent" ]; then
    $PYTHON ${CHEXAGENT_INFERENCE_DIR}/run_eval_batch.py \
        --num-chunks 4 \
        --model-name ${CHEXAGENT_HF} \
        --question-file ${question_file} \
        --image-folder ${image_folder} \
        --answers-file ${answer_file}.jsonl \
        --batch-size 1

    $PYTHON -c "
import json
with open('${answer_file}.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${answer_file_json}', 'w') as f:
    json.dump(data, f, indent=2)
print('Converted to JSON: ${answer_file_json}')
"

elif [ "${model_name}" == "medgemma" ]; then
    # Load HF_TOKEN from .env if present
    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    if [ -f "${REPO_ROOT}/.env" ]; then
        set -a; source "${REPO_ROOT}/.env"; set +a
    fi

    $PYTHON ${MEDGEMMA_INFERENCE_DIR}/run_eval_batch.py \
        --num-chunks 4 \
        --model-name ${MEDGEMMA_HF} \
        --question-file ${question_file} \
        --image-folder ${image_folder} \
        --answers-file ${answer_file}.jsonl \
        --batch-size 1

    $PYTHON -c "
import json
with open('${answer_file}.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${answer_file_json}', 'w') as f:
    json.dump(data, f, indent=2)
print('Converted to JSON: ${answer_file_json}')
"

else
    echo "Unknown model: ${model_name}"
    echo "Available models: llavamed, llavamed_hf, chexagent, medgemma"
    exit 1
fi

echo "=========================================="
echo "Done: ${model_name}"
echo "=========================================="
