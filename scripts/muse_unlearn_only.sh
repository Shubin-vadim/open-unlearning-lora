#!/bin/bash

# Script to run only the unlearning step for MUSE benchmark with LoRA
# This assumes that:
# 1. Fine-tuned model exists at: saves/finetune/muse_${MODEL}_${DATA_SPLIT}_full
# 2. Retain model evaluation exists at: saves/eval/muse_${MODEL}_${DATA_SPLIT}_${RETAIN_SPLIT}/MUSE_EVAL.json

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration - adjust these values as needed
MODEL="Qwen2.5-3B-Instruct"  # Original model from HuggingFace, will use LoRA for training
DATA_SPLIT="News"  # Options: News, Books
FORGET_SPLIT="forget"
RETAIN_SPLIT="retain1"  # MUSE dataset has 'retain1' and 'retain2' splits, not 'retain'
TRAINER="GradAscent"

# Training parameters
# Memory optimization: reduce batch size and increase gradient accumulation
PER_DEVICE_TRAIN_BATCH_SIZE=1  # Reduced to 1 to save VRAM (minimum batch size)
GRADIENT_ACCUMULATION_STEPS=16  # Increased to maintain effective batch size
NUM_GPUS=1  # Number of GPUs to use
GPU_IDS=0  # GPU device IDs (use 0 for single GPU, 0,1 for multi-GPU)

# Additional memory-saving options
MAX_LENGTH=2048  # MUSE uses longer sequences
USE_8BIT_OPTIMIZER=true  # Use 8-bit optimizer to save memory (paged_adamw_32bit)

# Set master port for distributed training
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Set PyTorch CUDA memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Calculate task names
FULL_TASK_NAME="muse_${MODEL}_${DATA_SPLIT}_full"
RETAIN_TASK_NAME="muse_${MODEL}_${DATA_SPLIT}_${RETAIN_SPLIT}"
UNLEARN_TASK_NAME="muse_${MODEL}_${DATA_SPLIT}_${FORGET_SPLIT}_${TRAINER}"

echo "=========================================="
echo "MUSE Unlearning Only (LoRA) - ${DATA_SPLIT}"
echo "=========================================="
echo "Model: $MODEL"
echo "Data Split: $DATA_SPLIT"
echo "Forget Split: $FORGET_SPLIT"
echo "Retain Split: $RETAIN_SPLIT"
echo "Trainer: $TRAINER"
echo "Batch Size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective Batch Size: $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "=========================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."
echo "-------------------------------------------"

# Check if fine-tuned model exists
FULL_MODEL_PATH="saves/finetune/${FULL_TASK_NAME}"
if [ ! -d "$FULL_MODEL_PATH" ]; then
    echo "❌ Error: Fine-tuned model not found at: $FULL_MODEL_PATH"
    echo "   Please run the full pipeline or fine-tune the model first."
    exit 1
fi
echo "✓ Fine-tuned model found: $FULL_MODEL_PATH"

# Check if retain evaluation exists
RETAIN_EVAL_PATH="saves/eval/${RETAIN_TASK_NAME}/MUSE_EVAL.json"
if [ ! -f "$RETAIN_EVAL_PATH" ]; then
    echo "⚠️  Warning: Retain evaluation not found at: $RETAIN_EVAL_PATH"
    echo "   Some evaluation metrics may not work correctly."
    echo "   You can set retain_logs_path=null to skip this requirement."
    RETAIN_LOGS_PATH="null"
else
    echo "✓ Retain evaluation found: $RETAIN_EVAL_PATH"
    RETAIN_LOGS_PATH="$RETAIN_EVAL_PATH"
fi

echo ""
echo "Prerequisites check completed!"
echo ""

########################################################################################################################
########################################### Unlearning #######################################################
########################################################################################################################

echo "Starting unlearning using ${TRAINER}..."
echo "-------------------------------------------"

TRAIN_CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python src/train.py \
    --config-name=unlearn \
    experiment=unlearn/muse/lora \
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
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${FULL_MODEL_PATH} \
    retain_logs_path=${RETAIN_LOGS_PATH} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.gradient_checkpointing=True"

if [ -n "${MAX_LENGTH:-}" ]; then
    TRAIN_CMD="${TRAIN_CMD} data.forget.MUSE_forget.args.max_length=${MAX_LENGTH} data.retain.MUSE_retain.args.max_length=${MAX_LENGTH}"
fi

if [ "${USE_8BIT_OPTIMIZER:-false}" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} trainer.args.optim=paged_adamw_32bit"
fi

# Clear GPU cache before training
echo "Clearing GPU cache before training..."
python -c "import torch; torch.cuda.empty_cache()" || true

echo ""
echo "Running unlearning command..."
echo "-------------------------------------------"
eval $TRAIN_CMD

echo ""
echo "✓ Unlearning completed"
echo "Unlearned model saved to: saves/unlearn/${UNLEARN_TASK_NAME}"
echo ""

########################################################################################################################
########################################### Optional: Evaluate Unlearned Model #########################################
########################################################################################################################

read -p "Do you want to evaluate the unlearned model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Evaluating unlearned model..."
    echo "-------------------------------------------"

    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        --config-name=eval \
        experiment=eval/muse/default \
        model=${MODEL} \
        task_name=${UNLEARN_TASK_NAME} \
        data_split=${DATA_SPLIT} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${UNLEARN_TASK_NAME} \
        paths.output_dir=saves/unlearn/${UNLEARN_TASK_NAME}/evals \
        retain_logs_path=${RETAIN_LOGS_PATH}

    echo "✓ Unlearned model evaluation completed"
    echo "Evaluation results saved to: saves/unlearn/${UNLEARN_TASK_NAME}/evals/MUSE_EVAL.json"
    echo ""
fi

echo "=========================================="
echo "Unlearning completed successfully! 🎉"
echo "=========================================="
echo "Unlearned model: saves/unlearn/${UNLEARN_TASK_NAME}"
if [ -f "saves/unlearn/${UNLEARN_TASK_NAME}/evals/MUSE_EVAL.json" ]; then
    echo "Evaluation results: saves/unlearn/${UNLEARN_TASK_NAME}/evals/MUSE_EVAL.json"
fi
echo "=========================================="

