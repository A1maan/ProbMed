#!/bin/bash
# Representation Probing Pipeline for CheXagent and MedGemma
# =================================================================
#
# Usage:
#   ./run_attention_probing.sh
#
# Prerequisites:
#   - CheXagent and MedGemma response files must exist in ../response_file/

set -e

# Load shared environment variables (HF_TOKEN, etc.) from ProbMed repo root
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

# ============================================
# CONFIGURATION
# ============================================
TEST_FILE="/workspace/ProbMed-Dataset/test/test.json"
IMAGE_FOLDER="/workspace/ProbMed-Dataset/test/"
OUTPUT_DIR_PROBING="./results/representation_probing"
NUM_PAIRS=500       # Number of paired samples for representation probing
NUM_GPUS=4          # Number of GPUs / extraction chunks to use

# CheXagent requires transformers==4.40.0 — use its own venv
PYTHON_CHEXAGENT=/venv/chexagent/bin/python3
# MedGemma works with the main venv (transformers 5.x)
PYTHON_MEDGEMMA=/venv/main/bin/python3

CHEXAGENT_MODEL="StanfordAIMI/CheXagent-2-3b"
CHEXAGENT_RESPONSE_FILE="../response_file/chexagent.json"

MEDGEMMA_MODEL="google/medgemma-1.5-4b-it"
MEDGEMMA_RESPONSE_FILE="../response_file/medgemma.json"

# ============================================
# Install MedGemma dependencies into main venv
# ============================================
echo "Installing MedGemma dependencies..."
/venv/main/bin/pip install -q transformers accelerate pillow tqdm matplotlib scikit-learn

# ============================================
# Create output directories
# ============================================
mkdir -p ${OUTPUT_DIR_PROBING}

run_probe() {
    local model_name="$1"
    local model_family="$2"
    local response_file="$3"
    local output_dir="$4"
    local python_bin="$5"

    if [ ! -f "${response_file}" ]; then
        echo "Missing response file: ${response_file}"
        exit 1
    fi

    for IMAGE_MODE in real black random; do
        echo ""
        echo "=========================================="
        echo "Representation Probing: ${model_family}, image-mode=${IMAGE_MODE} (${NUM_PAIRS} pairs)"
        echo "=========================================="

        local run_output_dir="${output_dir}"
        if [ "${IMAGE_MODE}" != "real" ]; then
            run_output_dir="${output_dir}/image_mode_${IMAGE_MODE}"
        fi
        local log_dir="${run_output_dir}/logs"
        mkdir -p "${log_dir}"

        local pids=()
        for CHUNK_IDX in $(seq 0 $((NUM_GPUS - 1))); do
            local log_file="${log_dir}/chunk${CHUNK_IDX}.log"
            echo "[${model_family}/${IMAGE_MODE}] Starting chunk ${CHUNK_IDX}/${NUM_GPUS} on GPU ${CHUNK_IDX} (log: ${log_file})"
            CUDA_VISIBLE_DEVICES=${CHUNK_IDX} ${python_bin} representation_probing/${model_family}/representation_probing.py \
                --model-name "${model_name}" \
                --response-file "${response_file}" \
                --test-file "${TEST_FILE}" \
                --image-folder "${IMAGE_FOLDER}" \
                --output-dir "${output_dir}" \
                --num-pairs "${NUM_PAIRS}" \
                --image-mode "${IMAGE_MODE}" \
                --num-chunks "${NUM_GPUS}" \
                --chunk-idx "${CHUNK_IDX}" \
                --extract-only \
                > "${log_file}" 2>&1 &
            pids+=("$!")
        done

        local failed=0
        for i in "${!pids[@]}"; do
            if ! wait "${pids[$i]}"; then
                echo "[${model_family}/${IMAGE_MODE}] Chunk ${i} failed. See ${log_dir}/chunk${i}.log"
                failed=1
            else
                echo "[${model_family}/${IMAGE_MODE}] Chunk ${i} completed"
            fi
        done

        if [ "${failed}" -ne 0 ]; then
            echo "One or more chunks failed for ${model_family}/${IMAGE_MODE}; skipping merge."
            exit 1
        fi

        echo "[${model_family}/${IMAGE_MODE}] Merging chunks and training probes"
        ${python_bin} representation_probing/${model_family}/representation_probing.py \
            --model-name "${model_name}" \
            --response-file "${response_file}" \
            --test-file "${TEST_FILE}" \
            --image-folder "${IMAGE_FOLDER}" \
            --output-dir "${output_dir}" \
            --num-pairs "${NUM_PAIRS}" \
            --image-mode "${IMAGE_MODE}" \
            --num-chunks "${NUM_GPUS}" \
            --train-from-chunks

        echo "Done: ${model_family}, image-mode=${IMAGE_MODE}"
    done
}

run_probe "${CHEXAGENT_MODEL}" "chexagent" "${CHEXAGENT_RESPONSE_FILE}" "${OUTPUT_DIR_PROBING}/chexagent" "${PYTHON_CHEXAGENT}"
run_probe "${MEDGEMMA_MODEL}" "medgemma" "${MEDGEMMA_RESPONSE_FILE}" "${OUTPUT_DIR_PROBING}/medgemma" "${PYTHON_MEDGEMMA}"

echo ""
echo "=========================================="
echo "DONE!"
echo "CheXagent probing results: ${OUTPUT_DIR_PROBING}/chexagent"
echo "MedGemma probing results: ${OUTPUT_DIR_PROBING}/medgemma"
echo "=========================================="
