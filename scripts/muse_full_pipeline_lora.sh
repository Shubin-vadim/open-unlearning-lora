#!/bin/bash

# Full Pipeline Script for MUSE Benchmark with LoRA
# This script demonstrates a complete workflow:
# 0. Evaluating original model (baseline)
# 1. Fine-tuning on MUSE full dataset
# 2. Fine-tuning retain model
# 3. Evaluating retain model
# 4. Unlearning using GradAscent
# 5. Evaluating unlearned model

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
MODEL="Qwen2.5-3B-Instruct"  # Original model from HuggingFace, will use LoRA for training
DATA_SPLIT="News"  # Options: News, Books
FORGET_SPLIT="forget"
RETAIN_SPLIT="retain1"
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

echo "=========================================="
echo "MUSE Full Pipeline (LoRA) - ${DATA_SPLIT}"
echo "=========================================="
echo "Model: $MODEL"
echo "Data Split: $DATA_SPLIT"
echo "Forget Split: $FORGET_SPLIT"
echo "Retain Split: $RETAIN_SPLIT"
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
echo "- Max Length: ${MAX_LENGTH:-2048}"
echo "- 8-bit Optimizer: ${USE_8BIT_OPTIMIZER:-false}"
echo "- Gradient Checkpointing: Enabled"
echo ""
echo "LoRA Benefits:"
echo "- Uses original model from HuggingFace"
echo "- Trains only ~1% of model parameters (LoRA adapters)"
echo "- Significantly reduced memory usage"
echo "- Faster training and smaller checkpoints"
echo ""

########################################################################################################################
########################################### Data Setup Check ##########################################################
########################################################################################################################

echo "Checking MUSE evaluation data availability..."
echo "-------------------------------------------"

# Check if eval logs directory exists and has content
EVAL_LOGS_DIR="saves/eval"
EVAL_LOGS_CHECK=false

# Check for some common MUSE eval log files
if [ -d "$EVAL_LOGS_DIR" ] && [ "$(ls -A $EVAL_LOGS_DIR 2>/dev/null)" ]; then
    # Check if there are any MUSE eval files
    if find "$EVAL_LOGS_DIR" -name "*MUSE*" -o -name "*muse*" 2>/dev/null | grep -q .; then
        EVAL_LOGS_CHECK=true
    fi
fi

if [ "$EVAL_LOGS_CHECK" = false ]; then
    echo "⚠️  MUSE evaluation logs not found!"
    echo "   Evaluation logs are needed for proper evaluation metrics."
    echo ""
    echo "Downloading evaluation logs..."
    echo "This will download eval logs for TOFU, MUSE retain and finetuned models."
    echo ""
    
    # Check if setup_data.py exists
    if [ ! -f "setup_data.py" ]; then
        echo "❌ Error: setup_data.py not found. Please run this script from the project root directory."
        exit 1
    fi
    
    # Download eval logs
    python setup_data.py --eval_logs
    
    # Verify download
    if [ ! -d "$EVAL_LOGS_DIR" ] || [ -z "$(ls -A $EVAL_LOGS_DIR 2>/dev/null)" ]; then
        echo "⚠️  Warning: Evaluation logs may not have been downloaded successfully."
        echo "   You can manually run: python setup_data.py --eval_logs"
        echo "   The pipeline will continue, but some evaluation metrics may not work correctly."
    else
        echo "✓ Evaluation logs downloaded successfully"
    fi
else
    echo "✓ Evaluation logs found"
fi

echo ""
echo "Note: MUSE dataset will be automatically downloaded from HuggingFace when needed."
echo ""

########################################################################################################################
########################################### Step 0: Evaluate Original Model ###########################################
########################################################################################################################

echo "Step 0: Evaluating original model (baseline)..."
echo "-------------------------------------------"

ORIGINAL_TASK_NAME="muse_${MODEL}_${DATA_SPLIT}_original"

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    --config-name=eval \
    experiment=eval/muse/default \
    model=${MODEL} \
    task_name=${ORIGINAL_TASK_NAME} \
    data_split=${DATA_SPLIT} \
    model.model_args.pretrained_model_name_or_path=Qwen/Qwen2.5-3B-Instruct

echo "✓ Original model evaluation completed"
echo "Evaluation results saved to: saves/eval/${ORIGINAL_TASK_NAME}/MUSE_EVAL.json"
echo ""

########################################################################################################################
########################################### Step 1: Fine-tune on MUSE Full ###########################################
########################################################################################################################
# NOTE: Full dataset contains ALL data (forget + retain). This model knows everything and serves as the starting point
# for unlearning. The unlearning process will try to "forget" the forget data while preserving retain knowledge.

echo "Step 1: Fine-tuning on MUSE ${DATA_SPLIT} full dataset..."
echo "-------------------------------------------"
echo "Note: Full dataset = forget data + retain data (all data together)"
echo "      This model will serve as the starting point for unlearning."
echo ""

FULL_TASK_NAME="muse_${MODEL}_${DATA_SPLIT}_full"

TRAIN_CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python src/train.py \
    --config-name=train \
    experiment=finetune/muse/lora \
    model=${MODEL} \
    +model.use_lora=true \
    +model.lora_config.target_modules='[\"q_proj\",\"v_proj\",\"k_proj\",\"o_proj\",\"gate_proj\",\"down_proj\",\"up_proj\",\"lm_head\"]' \
    +model.lora_config.lora_alpha=128 \
    +model.lora_config.lora_dropout=0.05 \
    +model.lora_config.r=128 \
    +model.lora_config.bias=none \
    +model.lora_config.task_type=CAUSAL_LM \
    task_name=${FULL_TASK_NAME} \
    data_split=${DATA_SPLIT} \
    data_sub_set=full \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.gradient_checkpointing=True"

if [ -n "${MAX_LENGTH:-}" ]; then
    TRAIN_CMD="${TRAIN_CMD} data.train.MUSE_train.args.max_length=${MAX_LENGTH}"
fi

if [ "${USE_8BIT_OPTIMIZER:-false}" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} trainer.args.optim=paged_adamw_32bit"
fi

# Clear GPU cache before training
echo "Clearing GPU cache before training..."
python -c "import torch; torch.cuda.empty_cache()" || true

eval $TRAIN_CMD

echo "✓ Fine-tuning on full dataset completed"
echo "Model saved to: saves/finetune/${FULL_TASK_NAME}"
echo ""

########################################################################################################################
########################################### Step 2: Fine-tune Retain Model ###########################################
########################################################################################################################
# NOTE: Retain dataset contains ONLY retain data (without forget data). This model represents the "ideal" target:
# a model that never saw forget data. It's used as a reference to evaluate unlearning quality.
# 
# Why train both Full and Retain models?
# - Full model: knows everything (forget + retain) - this is what we start with
# - Retain model: knows only retain data - this is what we want to achieve through unlearning
# - Unlearning: transforms Full model → something similar to Retain model (forgets forget, keeps retain)
#
# If you already have a retain model (e.g., from HuggingFace or previous experiments), you can skip this step
# and set RETAIN_MODEL_PATH variable below to use an existing model instead.

RETAIN_TASK_NAME="muse_${MODEL}_${DATA_SPLIT}_${RETAIN_SPLIT}"

# Uncomment and set this if you want to use an existing retain model instead of training one
# RETAIN_MODEL_PATH="open-unlearning/muse_${MODEL}_${DATA_SPLIT}_${RETAIN_SPLIT}"  # or path to your existing model
# SKIP_RETAIN_TRAINING=true

if [ -z "${SKIP_RETAIN_TRAINING:-}" ]; then
    echo "Step 2: Fine-tuning retain model on ${RETAIN_SPLIT} split..."
    echo "-------------------------------------------"
    echo "Note: Retain dataset = ONLY retain data (without forget data)"
    echo "      This model represents the ideal target: a model that never saw forget data."
    echo "      It's used as a reference for forget_quality metric evaluation."
    echo ""

    TRAIN_CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python src/train.py \
        --config-name=train \
        experiment=finetune/muse/lora \
        model=${MODEL} \
        +model.use_lora=true \
        +model.lora_config.target_modules='[\"q_proj\",\"v_proj\",\"k_proj\",\"o_proj\",\"gate_proj\",\"down_proj\",\"up_proj\",\"lm_head\"]' \
        +model.lora_config.lora_alpha=128 \
        +model.lora_config.lora_dropout=0.05 \
        +model.lora_config.r=128 \
        +model.lora_config.bias=none \
        +model.lora_config.task_type=CAUSAL_LM \
        task_name=${RETAIN_TASK_NAME} \
        data_split=${DATA_SPLIT} \
        data_sub_set=${RETAIN_SPLIT} \
        trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
        trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        trainer.args.gradient_checkpointing=True"

    if [ -n "${MAX_LENGTH:-}" ]; then
        TRAIN_CMD="${TRAIN_CMD} data.train.MUSE_train.args.max_length=${MAX_LENGTH}"
    fi

    if [ "${USE_8BIT_OPTIMIZER:-false}" = "true" ]; then
        TRAIN_CMD="${TRAIN_CMD} trainer.args.optim=paged_adamw_32bit"
    fi

    # Clear GPU cache before training
    echo "Clearing GPU cache before training..."
    python -c "import torch; torch.cuda.empty_cache()" || true

    eval $TRAIN_CMD

    echo "✓ Retain model fine-tuning completed"
    echo "Model saved to: saves/finetune/${RETAIN_TASK_NAME}"
    echo ""
else
    echo "Step 2: Using existing retain model (skipping training)..."
    echo "-------------------------------------------"
    echo "Using retain model from: ${RETAIN_MODEL_PATH:-saves/finetune/${RETAIN_TASK_NAME}}"
    echo ""
fi

########################################################################################################################
########################################### Step 3: Evaluate Retain Model ###########################################
########################################################################################################################

echo "Step 3: Evaluating retain model..."
echo "-------------------------------------------"

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    --config-name=eval \
    experiment=eval/muse/default \
    model=${MODEL} \
    task_name=${RETAIN_TASK_NAME} \
    data_split=${DATA_SPLIT} \
    model.model_args.pretrained_model_name_or_path=saves/finetune/${RETAIN_TASK_NAME}

echo "✓ Retain model evaluation completed"
echo "Evaluation results saved to: saves/eval/${RETAIN_TASK_NAME}/MUSE_EVAL.json"
echo ""

########################################################################################################################
########################################### Step 4: Unlearning #######################################################
########################################################################################################################

echo "Step 4: Unlearning using ${TRAINER}..."
echo "-------------------------------------------"

UNLEARN_TASK_NAME="muse_${MODEL}_${DATA_SPLIT}_${FORGET_SPLIT}_${TRAINER}"

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
    model.model_args.pretrained_model_name_or_path=saves/finetune/${FULL_TASK_NAME} \
    retain_logs_path=saves/eval/${RETAIN_TASK_NAME}/MUSE_EVAL.json \
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

eval $TRAIN_CMD

echo "✓ Unlearning completed"
echo "Unlearned model saved to: saves/unlearn/${UNLEARN_TASK_NAME}"
echo ""

########################################################################################################################
########################################### Step 5: Evaluate Unlearned Model #########################################
########################################################################################################################

echo "Step 5: Evaluating unlearned model..."
echo "-------------------------------------------"

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    --config-name=eval \
    experiment=eval/muse/default \
    model=${MODEL} \
    task_name=${UNLEARN_TASK_NAME} \
    data_split=${DATA_SPLIT} \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/${UNLEARN_TASK_NAME} \
    paths.output_dir=saves/unlearn/${UNLEARN_TASK_NAME}/evals \
    retain_logs_path=saves/eval/${RETAIN_TASK_NAME}/MUSE_EVAL.json

echo "✓ Unlearned model evaluation completed"
echo "Evaluation results saved to: saves/unlearn/${UNLEARN_TASK_NAME}/evals/MUSE_EVAL.json"
echo ""

########################################################################################################################
########################################### Summary ####################################################################
########################################################################################################################

echo "=========================================="
echo "Pipeline Summary"
echo "=========================================="
echo "✓ Original model evaluation: saves/eval/${ORIGINAL_TASK_NAME}/MUSE_EVAL.json"
echo "✓ Fine-tuned model (full): saves/finetune/${FULL_TASK_NAME}"
echo "✓ Retain model: saves/finetune/${RETAIN_TASK_NAME}"
echo "✓ Retain evaluation: saves/eval/${RETAIN_TASK_NAME}/MUSE_EVAL.json"
echo "✓ Unlearned model: saves/unlearn/${UNLEARN_TASK_NAME}"
echo "✓ Unlearned evaluation: saves/unlearn/${UNLEARN_TASK_NAME}/evals/MUSE_EVAL.json"
echo "=========================================="
echo ""
echo "Pipeline completed successfully! 🎉"

