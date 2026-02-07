#!/bin/bash

# Full Pipeline Script for TOFU Benchmark with LoRA - SimNPO Method
# This script demonstrates a complete workflow:
# 0. Evaluating original model (baseline)
# 1. Fine-tuning on TOFU full dataset
# 2. Fine-tuning retain model
# 3. Evaluating retain model
# 4. Unlearning using SimNPO (Simplified Negative Preference Optimization)
# 5. Evaluating unlearned model
#
# Memory Requirements: Low-Medium
# - Does NOT create a reference model (more memory efficient than NPO)
# - Uses simplified loss function without DPO
# - Memory: ~1x model + activations for 2 batches (forget + retain)

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
MODEL="Qwen2.5-3B-Instruct"  # Model name (used for config selection), will use LoRA for training
MODEL_BASE_PATH="Qwen/Qwen2.5-3B-Instruct"  # HuggingFace model path or local path to base model
FORGET_SPLIT="forget10"  # Options: forget10, forget5, forget1
RETAIN_SPLIT="retain90"  # Options: retain90, retain95, retain99
HOLDOUT_SPLIT="holdout10"  # Options: holdout10, holdout5, holdout1
TRAINER="SimNPO"  # Unlearning method: SimNPO

# SimNPO-specific parameters
DELTA=0.0  # Threshold for forget loss
BETA=4.5  # Temperature parameter for sigmoid
GAMMA=0.125  # Weight for forget loss (npo_coeff)
ALPHA=1.0  # Weight for retain loss
RETAIN_LOSS_TYPE="NLL"  # Options: NLL, KL

# Training parameters
# Memory optimization: reduce batch size and increase gradient accumulation
PER_DEVICE_TRAIN_BATCH_SIZE=1  # Reduced to 1 to save VRAM (minimum batch size)
GRADIENT_ACCUMULATION_STEPS=16  # Increased to maintain effective batch size
NUM_GPUS=1  # Number of GPUs to use
GPU_IDS=0  # GPU device IDs (use 0 for single GPU, 0,1 for multi-GPU)

# Additional memory-saving options
MAX_LENGTH=256  # Reduce sequence length (default: 512) - reduces VRAM significantly
USE_8BIT_OPTIMIZER=true  # Use 8-bit optimizer to save memory (paged_adamw_32bit)

# Set master port for distributed training
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Set PyTorch CUDA memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "TOFU Full Pipeline (LoRA) - SimNPO"
echo "=========================================="
echo "Model: $MODEL"
echo "Forget Split: $FORGET_SPLIT"
echo "Retain Split: $RETAIN_SPLIT"
echo "Holdout Split: $HOLDOUT_SPLIT"
echo "Trainer: $TRAINER"
echo "Delta (threshold): $DELTA"
echo "Beta (temperature): $BETA"
echo "Gamma (forget weight): $GAMMA"
echo "Alpha (retain weight): $ALPHA"
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
echo "- Max Length: ${MAX_LENGTH:-512} (reduced to save VRAM)"
echo "- 8-bit Optimizer: ${USE_8BIT_OPTIMIZER:-false}"
echo "- Gradient Checkpointing: Enabled"
echo ""
echo "✓ SimNPO Memory Requirements:"
echo "- Does NOT create a reference model (more memory efficient than NPO)"
echo "- Uses simplified loss function without DPO"
echo "- Memory: ~1x model + activations for 2 batches (forget + retain)"
echo "- More memory efficient than NPO while maintaining good performance"
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

echo "Checking TOFU evaluation data availability..."
echo "-------------------------------------------"

# Check if eval logs directory exists and has content
EVAL_LOGS_DIR="saves/eval"
EVAL_LOGS_CHECK=false

# Check for some common TOFU eval log files
if [ -d "$EVAL_LOGS_DIR" ] && [ "$(ls -A $EVAL_LOGS_DIR 2>/dev/null)" ]; then
    # Check if there are any TOFU eval files
    if find "$EVAL_LOGS_DIR" -name "*TOFU*" -o -name "*tofu*" 2>/dev/null | grep -q .; then
        EVAL_LOGS_CHECK=true
    fi
fi

if [ "$EVAL_LOGS_CHECK" = false ]; then
    echo "⚠️  TOFU evaluation logs not found!"
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
echo "Note: TOFU dataset will be automatically downloaded from HuggingFace when needed."
echo ""

########################################################################################################################
########################################### Step 0: Evaluate Original Model ###########################################
########################################################################################################################

echo "Step 0: Evaluating original model (baseline)..."
echo "-------------------------------------------"

ORIGINAL_TASK_NAME="tofu_${MODEL}_original"

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    --config-name=eval \
    experiment=eval/tofu/default \
    model=${MODEL} \
    task_name=${ORIGINAL_TASK_NAME} \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${MODEL_BASE_PATH}

echo "✓ Original model evaluation completed"
echo "Evaluation results saved to: saves/eval/${ORIGINAL_TASK_NAME}/TOFU_EVAL.json"
echo ""

########################################################################################################################
########################################### Step 1: Fine-tune on TOFU Full ###########################################
########################################################################################################################

echo "Step 1: Fine-tuning on TOFU full dataset..."
echo "-------------------------------------------"
echo "Note: Full dataset = forget data + retain data (all data together)"
echo "      This model will serve as the starting point for unlearning."
echo ""

FULL_TASK_NAME="tofu_${MODEL}_full"

TRAIN_CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python src/train.py \
    --config-name=train \
    experiment=finetune/tofu/lora \
    model=${MODEL} \
    +model.use_lora=true \
    +model.lora_config.target_modules='[\"q_proj\",\"v_proj\",\"k_proj\",\"o_proj\",\"gate_proj\",\"down_proj\",\"up_proj\",\"lm_head\"]' \
    +model.lora_config.lora_alpha=128 \
    +model.lora_config.lora_dropout=0.05 \
    +model.lora_config.r=128 \
    +model.lora_config.bias=none \
    +model.lora_config.task_type=CAUSAL_LM \
    task_name=${FULL_TASK_NAME} \
    data/datasets@data.train=TOFU_QA_full \
    data.train.TOFU_QA_full.args.hf_args.name=full \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.gradient_checkpointing=True"

if [ -n "${MAX_LENGTH:-}" ]; then
    TRAIN_CMD="${TRAIN_CMD} data.train.TOFU_QA_full.args.max_length=${MAX_LENGTH}"
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

RETAIN_TASK_NAME="tofu_${MODEL}_${RETAIN_SPLIT}"

# Uncomment and set this if you want to use an existing retain model instead of training one
# RETAIN_MODEL_PATH="open-unlearning/tofu_${MODEL}_${RETAIN_SPLIT}"  # or path to your existing model
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
        experiment=finetune/tofu/lora \
        model=${MODEL} \
        +model.use_lora=true \
        +model.lora_config.target_modules='[\"q_proj\",\"v_proj\",\"k_proj\",\"o_proj\",\"gate_proj\",\"down_proj\",\"up_proj\",\"lm_head\"]' \
        +model.lora_config.lora_alpha=128 \
        +model.lora_config.lora_dropout=0.05 \
        +model.lora_config.r=128 \
        +model.lora_config.bias=none \
        +model.lora_config.task_type=CAUSAL_LM \
        task_name=${RETAIN_TASK_NAME} \
        data/datasets@data.train=TOFU_QA_retain \
        data.train.TOFU_QA_retain.args.hf_args.name=${RETAIN_SPLIT} \
        trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
        trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        trainer.args.gradient_checkpointing=True"

    if [ -n "${MAX_LENGTH:-}" ]; then
        TRAIN_CMD="${TRAIN_CMD} data.train.TOFU_QA_retain.args.max_length=${MAX_LENGTH}"
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
    experiment=eval/tofu/default \
    model=${MODEL} \
    task_name=${RETAIN_TASK_NAME} \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    model.model_args.pretrained_model_name_or_path=saves/finetune/${RETAIN_TASK_NAME}

echo "✓ Retain model evaluation completed"
echo "Evaluation results saved to: saves/eval/${RETAIN_TASK_NAME}/TOFU_EVAL.json"
echo ""

########################################################################################################################
########################################### Step 4: Unlearning #######################################################
########################################################################################################################

echo "Step 4: Unlearning using ${TRAINER}..."
echo "-------------------------------------------"
echo "SimNPO (Simplified Negative Preference Optimization) uses a simplified loss function"
echo "without requiring a reference model, making it more memory efficient than NPO."
echo ""

UNLEARN_TASK_NAME="tofu_${MODEL}_${FORGET_SPLIT}_${TRAINER}"

TRAIN_CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python src/train.py \
    --config-name=unlearn \
    experiment=unlearn/tofu/lora \
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
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=saves/finetune/${FULL_TASK_NAME} \
    retain_logs_path=saves/eval/${RETAIN_TASK_NAME}/TOFU_EVAL.json \
    trainer.method_args.delta=${DELTA} \
    trainer.method_args.beta=${BETA} \
    trainer.method_args.gamma=${GAMMA} \
    trainer.method_args.alpha=${ALPHA} \
    trainer.method_args.retain_loss_type=${RETAIN_LOSS_TYPE} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.gradient_checkpointing=True"

if [ -n "${MAX_LENGTH:-}" ]; then
    TRAIN_CMD="${TRAIN_CMD} data.forget.TOFU_QA_forget.args.max_length=${MAX_LENGTH} data.retain.TOFU_QA_retain.args.max_length=${MAX_LENGTH}"
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
echo "This evaluation includes MIA (Membership Inference Attack) metrics to assess"
echo "how well the model has forgotten the forget data."
echo ""

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    --config-name=eval \
    experiment=eval/tofu/default \
    eval=tofu_with_mia \
    model=${MODEL} \
    task_name=${UNLEARN_TASK_NAME} \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/${UNLEARN_TASK_NAME} \
    paths.output_dir=saves/unlearn/${UNLEARN_TASK_NAME}/evals \
    retain_logs_path=saves/eval/${RETAIN_TASK_NAME}/TOFU_EVAL.json

echo "✓ Unlearned model evaluation completed"
echo "Evaluation results saved to: saves/unlearn/${UNLEARN_TASK_NAME}/evals/TOFU_EVAL.json"
echo "Note: MIA metrics are included to assess forgetting quality."
echo ""

########################################################################################################################
########################################### Step 6: Generate Visualizations ###########################################
########################################################################################################################

echo "Step 6: Generating visualizations..."
echo "-------------------------------------------"

# Check if matplotlib is available
if python -c "import matplotlib" 2>/dev/null; then
    echo "Matplotlib found, generating visualizations..."
    
    # Create plots directory for comparisons
    PLOTS_DIR="plots/${UNLEARN_TASK_NAME}"
    mkdir -p "${PLOTS_DIR}"
    
    # Generate comparison plot: Fine-tune vs Unlearn
    echo "Generating comparison plot..."
    cd src
    python -m plot compare \
        -e "../saves/finetune/${FULL_TASK_NAME}" \
        -e "../saves/unlearn/${UNLEARN_TASK_NAME}" \
        -n "Fine-tune (Full)" \
        -n "${TRAINER}" \
        -o "../${PLOTS_DIR}/comparison.png" \
        -t "TOFU: Fine-tune vs ${TRAINER} Unlearning" 2>/dev/null || echo "⚠️  Comparison plot generation failed (non-critical)"
    
    # Generate metrics comparison
    echo "Generating metrics comparison..."
    python -m plot metrics \
        -e "../saves/finetune/${FULL_TASK_NAME}" \
        -e "../saves/unlearn/${UNLEARN_TASK_NAME}" \
        -o "../${PLOTS_DIR}/metrics_comparison.png" \
        -t "TOFU: Metrics Comparison - ${TRAINER}" 2>/dev/null || echo "⚠️  Metrics comparison generation failed (non-critical)"
    
    # Generate dashboard if retain model exists
    if [ -d "../saves/finetune/${RETAIN_TASK_NAME}" ]; then
        echo "Generating dashboard..."
        python -m plot dashboard \
            -f "../saves/finetune/${FULL_TASK_NAME}" \
            -u "../saves/unlearn/${UNLEARN_TASK_NAME}" \
            -o "../${PLOTS_DIR}/dashboard.png" \
            -t "TOFU: Unlearning Dashboard - ${TRAINER}" 2>/dev/null || echo "⚠️  Dashboard generation failed (non-critical)"
    fi
    
    cd ..
    
    echo "✓ Visualizations saved to: ${PLOTS_DIR}/"
    echo "  - comparison.png"
    echo "  - metrics_comparison.png"
    if [ -d "saves/finetune/${RETAIN_TASK_NAME}" ]; then
        echo "  - dashboard.png"
    fi
else
    echo "⚠️  Matplotlib not found. Skipping visualization generation."
    echo "   Install with: pip install matplotlib seaborn"
fi

echo ""

########################################################################################################################
########################################### Summary ####################################################################
########################################################################################################################

echo "=========================================="
echo "Pipeline Summary"
echo "=========================================="
echo "✓ Original model evaluation: saves/eval/${ORIGINAL_TASK_NAME}/TOFU_EVAL.json"
echo "✓ Fine-tuned model (full): saves/finetune/${FULL_TASK_NAME}"
echo "✓ Retain model: saves/finetune/${RETAIN_TASK_NAME}"
echo "✓ Retain evaluation: saves/eval/${RETAIN_TASK_NAME}/TOFU_EVAL.json"
echo "✓ Unlearned model: saves/unlearn/${UNLEARN_TASK_NAME}"
echo "✓ Unlearned evaluation: saves/unlearn/${UNLEARN_TASK_NAME}/evals/TOFU_EVAL.json"
echo "=========================================="
echo ""
echo "Pipeline completed successfully! 🎉"

