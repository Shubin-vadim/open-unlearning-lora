#!/bin/bash

# Full Pipeline Script for WMDP Benchmark with LoRA
# This script demonstrates a complete workflow:
# 0. Evaluating original model (baseline)
# 1. Unlearning using RMU (default trainer for WMDP)
# 2. Evaluating unlearned model
#
# NOTE: WMDP typically uses pre-trained models from HuggingFace.
# This script assumes you're starting from a base model and doing unlearning.

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
MODEL="Qwen2.5-3B-Instruct"  # Original model from HuggingFace, will use LoRA for training
# NOTE: WMDP default uses zephyr-7b-beta, but this script uses Qwen2.5-3B-Instruct with LoRA
#       You can change MODEL to "zephyr-7b-beta" if you prefer, but make sure to update
#       the model config path accordingly
DATA_SPLIT="cyber"  # Options: cyber, bio
TRAINER="RMU"  # Default trainer for WMDP

# RMU-specific configuration (adjust based on model architecture)
# For Qwen with PEFT LoRA: use "base_model.model.layers.7" (PEFT wraps base_model)
# For Qwen with custom LoRA wrapper: try "model.model.layers.7"
# For Zephyr/Mistral without LoRA: use "model.layers.7"
# Adjust layer number based on model size (e.g., 7 for 7B models, 3-4 for smaller models)
# If you get "No module matched" error, the improved error message will show available modules
RMU_MODULE_REGEX="base_model\\.model\\.model\\.layers\\.7"  # For Qwen with PEFT LoRA (default - three model levels!)
# Alternative options (uncomment one if default doesn't work):
# RMU_MODULE_REGEX="base_model\\.model\\.layers\\.7"  # For PEFT with two model levels
# RMU_MODULE_REGEX="model\\.model\\.layers\\.7"  # For custom LoRA wrapper
# RMU_MODULE_REGEX="model\\.layers\\.7"  # For models without LoRA wrapper
# RMU_MODULE_REGEX="base_model\\.model\\.model\\.layers\\.3"  # For smaller models, try lower layer numbers
RMU_TRAINABLE_PARAMS_REGEX=".*"  # Update all parameters, or use specific regex like "base_model\\.model\\.layers\\.(5|6|7)\\.mlp\\.down_proj\\.weight"

# Training parameters
# Memory optimization: reduce batch size and increase gradient accumulation
PER_DEVICE_TRAIN_BATCH_SIZE=1  # Reduced to 1 to save VRAM (minimum batch size)
GRADIENT_ACCUMULATION_STEPS=16  # Increased to maintain effective batch size
NUM_GPUS=1  # Number of GPUs to use
GPU_IDS=0  # GPU device IDs (use 0 for single GPU, 0,1 for multi-GPU)

# Additional memory-saving options
USE_8BIT_OPTIMIZER=true  # Use 8-bit optimizer to save memory (paged_adamw_32bit)

# Set master port for distributed training
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Set PyTorch CUDA memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "WMDP Full Pipeline (LoRA) - ${DATA_SPLIT}"
echo "=========================================="
echo "Model: $MODEL"
echo "Data Split: $DATA_SPLIT"
echo "Trainer: $TRAINER"
echo "Batch Size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective Batch Size: $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Gradient Checkpointing: Enabled (saves VRAM)"
echo "=========================================="
echo ""
echo "Memory Optimization Settings:"
echo "- Base Model: $MODEL (original from HuggingFace)"
echo "- Using LoRA: Yes (parameter-efficient fine-tuning)"
echo "- Batch Size: $PER_DEVICE_TRAIN_BATCH_SIZE (minimum)"
echo "- Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "- 8-bit Optimizer: ${USE_8BIT_OPTIMIZER:-false}"
echo "- Gradient Checkpointing: Enabled"
echo ""
echo "LoRA Benefits:"
echo "- Uses original model from HuggingFace"
echo "- Trains only ~1% of model parameters (LoRA adapters)"
echo "- Significantly reduced memory usage"
echo "- Faster training and smaller checkpoints"
echo ""
echo "NOTE: WMDP typically uses pre-trained models. If you have a pre-trained model,"
echo "      you can set BASE_MODEL_PATH to skip the baseline evaluation."
echo ""

########################################################################################################################
########################################### Data Setup Check ##########################################################
########################################################################################################################

echo "Checking WMDP data availability..."
echo "-------------------------------------------"

WMDP_DATA_DIR="data/wmdp/wmdp-corpora"
FORGET_FILE="${WMDP_DATA_DIR}/${DATA_SPLIT}-forget-corpus.jsonl"
RETAIN_FILE="${WMDP_DATA_DIR}/${DATA_SPLIT}-retain-corpus.jsonl"

if [ ! -f "$FORGET_FILE" ] || [ ! -f "$RETAIN_FILE" ]; then
    echo "⚠️  WMDP data files not found!"
    echo "   Missing: $FORGET_FILE or $RETAIN_FILE"
    echo ""
    echo "Downloading WMDP dataset..."
    echo "This will download and extract the WMDP corpora."
    echo ""
    
    # Check if setup_data.py exists
    if [ ! -f "setup_data.py" ]; then
        echo "❌ Error: setup_data.py not found. Please run this script from the project root directory."
        exit 1
    fi
    
    # Download WMDP data
    python setup_data.py --wmdp
    
    # Verify download
    if [ ! -f "$FORGET_FILE" ] || [ ! -f "$RETAIN_FILE" ]; then
        echo "❌ Error: Failed to download WMDP data files."
        echo "   Please manually run: python setup_data.py --wmdp"
        echo "   Or download from: https://cais-wmdp.s3.us-west-1.amazonaws.com/wmdp-corpora.zip"
        echo "   Password: wmdpcorpora"
        exit 1
    fi
    
    echo "✓ WMDP data downloaded successfully"
else
    echo "✓ WMDP data files found"
fi

echo ""

########################################################################################################################
########################################### Step 0: Evaluate Original Model ###########################################
########################################################################################################################

echo "Step 0: Evaluating original model (baseline)..."
echo "-------------------------------------------"
echo "Note: WMDP evaluation uses lm-evaluation-harness"
echo "      This evaluates on wmdp_${DATA_SPLIT} and mmlu tasks"
echo ""

ORIGINAL_TASK_NAME="wmdp_${MODEL}_${DATA_SPLIT}_original"

# Check if we should use a pre-trained model
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    --config-name=eval \
    experiment=eval/wmdp/default \
    model=${MODEL} \
    task_name=${ORIGINAL_TASK_NAME} \
    data_split=${DATA_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${BASE_MODEL_PATH}

echo "✓ Original model evaluation completed"
echo "Evaluation results saved to: saves/eval/${ORIGINAL_TASK_NAME}/LM_EVAL.json"
echo ""

########################################################################################################################
########################################### Step 1: Unlearning #######################################################
########################################################################################################################

echo "Step 1: Unlearning using ${TRAINER}..."
echo "-------------------------------------------"
echo "Note: WMDP unlearning uses forget and retain corpora"
echo "      Forget corpus: data/wmdp/wmdp-corpora/${DATA_SPLIT}-forget-corpus.jsonl"
echo "      Retain corpus: data/wmdp/wmdp-corpora/${DATA_SPLIT}-retain-corpus.jsonl"
echo ""
echo "RMU Configuration:"
echo "  Using default config: module_regex='base_model.model.model.layers.7' (for Qwen with PEFT LoRA)"
echo "  Note: Qwen with PEFT LoRA has THREE 'model' levels: base_model.model.model.layers.X"
echo "  If you get 'No module matched' error:"
echo "    - The error message will show available layer modules"
echo "    - You can override in the script by uncommenting RMU_MODULE_REGEX override lines"
echo "    - Or edit configs/experiment/unlearn/wmdp/lora.yaml"
echo "  Common patterns:"
echo "    - For Qwen with PEFT LoRA: 'base_model.model.model.layers.7' (default - three model levels)"
echo "    - For PEFT with two levels: 'base_model.model.layers.7'"
echo "    - For Zephyr/Mistral: 'model.layers.7'"
echo "    - For smaller models: try lower layer numbers (e.g., 'base_model.model.model.layers.3')"
echo ""

UNLEARN_TASK_NAME="wmdp_${MODEL}_${DATA_SPLIT}_${TRAINER}"

TRAIN_CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python src/train.py \
    --config-name=unlearn \
    experiment=unlearn/wmdp/lora \
    model=${MODEL} \
    +model.use_lora=true \
    +model.lora_config.target_modules='[\"q_proj\",\"v_proj\",\"k_proj\",\"o_proj\",\"gate_proj\",\"down_proj\",\"up_proj\",\"lm_head\"]' \
    +model.lora_config.lora_alpha=128 \
    +model.lora_config.lora_dropout=0.05 \
    +model.lora_config.r=128 \
    +model.lora_config.bias=none \
    +model.lora_config.task_type=CAUSAL_LM \
    trainer=${TRAINER} \
    task_name=${UNLEARN_TASK_NAME} \
    data_split=${DATA_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.gradient_checkpointing=True"

# Override RMU module_regex if custom value is set (uncomment to use custom regex)
# TRAIN_CMD="${TRAIN_CMD} trainer.method_args.module_regex='${RMU_MODULE_REGEX}'"
# TRAIN_CMD="${TRAIN_CMD} trainer.method_args.trainable_params_regex='[\"${RMU_TRAINABLE_PARAMS_REGEX}\"]'"

if [ "${USE_8BIT_OPTIMIZER:-false}" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} trainer.args.optim=paged_adamw_32bit"
fi

# Clear GPU cache before training
echo "Clearing GPU cache before training..."
python -c "import torch; torch.cuda.empty_cache()" || true

eval $TRAIN_CMD

echo "✓ Unlearning completed"
echo "Unlearned model saved to: saves/unlearn/${UNLEARN_TASK_NAME}"
echo ""

########################################################################################################################
########################################### Step 2: Evaluate Unlearned Model #########################################
########################################################################################################################

echo "Step 2: Evaluating unlearned model..."
echo "-------------------------------------------"

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    --config-name=eval \
    experiment=eval/wmdp/default \
    model=${MODEL} \
    task_name=${UNLEARN_TASK_NAME} \
    data_split=${DATA_SPLIT} \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/${UNLEARN_TASK_NAME} \
    paths.output_dir=saves/unlearn/${UNLEARN_TASK_NAME}/evals

echo "✓ Unlearned model evaluation completed"
echo "Evaluation results saved to: saves/unlearn/${UNLEARN_TASK_NAME}/evals/LM_EVAL.json"
echo ""

########################################################################################################################
########################################### Summary ####################################################################
########################################################################################################################

echo "=========================================="
echo "Pipeline Summary"
echo "=========================================="
echo "✓ Original model evaluation: saves/eval/${ORIGINAL_TASK_NAME}/LM_EVAL.json"
echo "✓ Unlearned model: saves/unlearn/${UNLEARN_TASK_NAME}"
echo "✓ Unlearned evaluation: saves/unlearn/${UNLEARN_TASK_NAME}/evals/LM_EVAL.json"
echo "=========================================="
echo ""
echo "Pipeline completed successfully! 🎉"
echo ""
echo "NOTE: WMDP evaluation includes:"
echo "  - wmdp_${DATA_SPLIT}: WMDP-specific evaluation"
echo "  - mmlu: General knowledge benchmark"

