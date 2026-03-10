set -e

MARGIN_SCORES_FILE="../vcd/results/vcd_analysis/margin_scores.json"
TEST_FILE="/workspace/ProbMed-Dataset/test/test.json"
IMAGE_FOLDER="/workspace/ProbMed-Dataset/test/"
OUTPUT_DIR_ATTENTION="./results/attention_analysis"
OUTPUT_DIR_PROBING="./results/lr_lm_alignment"
NUM_PAIRS=500       # Number of paired samples for representation probing
NUM_GPUS=4          # Number of GPUs to use


echo "Installing dependencies..."
pip install -q transformers accelerate bitsandbytes pillow tqdm matplotlib scikit-learn

mkdir -p ${OUTPUT_DIR_ATTENTION}
mkdir -p ${OUTPUT_DIR_PROBING}

echo ""
echo "=========================================="
echo "Step 1: Representation Probing (${NUM_PAIRS} pairs)"
echo "=========================================="

python representation_probing.py \
    --margin-scores-file ${MARGIN_SCORES_FILE} \
    --test-file ${TEST_FILE} \
    --image-folder ${IMAGE_FOLDER} \
    --output-dir ${OUTPUT_DIR_PROBING} \
    --num-pairs ${NUM_PAIRS} \
    --load-8bit

echo ""
echo "=========================================="
echo "Step 2: LR-LM Alignment Analysis"
echo "=========================================="

python lr_lm_alignment.py \
    --weights-file ${OUTPUT_DIR_PROBING}/lr_weights.npz \
    --output-dir ${OUTPUT_DIR_PROBING} \
    --load-8bit

echo ""
echo "=========================================="
echo "DONE!"
echo "Probing results: ${OUTPUT_DIR_PROBING}"
echo "Alignment results: ${OUTPUT_DIR_PROBING}/lm_alignment_analysis.json"
echo "=========================================="