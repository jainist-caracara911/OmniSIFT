#!/usr/bin/env bash
set -euo pipefail

: "${MS_SWIFT_ROOT:?Set MS_SWIFT_ROOT=/path/to/ms-swift}"
: "${MODEL_PATH:?Set MODEL_PATH=/path/to/model}"
: "${DATASET_JSONL:?Set DATASET_JSONL=/path/to/train.jsonl}"
: "${DS_CONFIG:?Set DS_CONFIG=/path/to/deepspeed_config.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNISIFT_ROOT="${OMNISIFT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
HOSTFILE="${HOSTFILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
EXP_NAME="${EXP_NAME:-omnisift_sft}"
LOG_DIR="${LOG_DIR:-$MS_SWIFT_ROOT/train_logs}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$((GLOBAL_BATCH_SIZE / NPROC_PER_NODE))}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LR_SCHEDULER_KWARGS="${LR_SCHEDULER_KWARGS:-{\"num_cycles\": 0.47}}"
DEEPSPEED_ENV_FILE="${DEEPSPEED_ENV_FILE:-}"

SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export DS_ACCELERATOR="${DS_ACCELERATOR:-cuda}"
export PYTHONPATH="$OMNISIFT_ROOT:$MS_SWIFT_ROOT:${PYTHONPATH:-}"

if [[ "$GRADIENT_ACCUMULATION_STEPS" -lt 1 ]]; then
  echo "GRADIENT_ACCUMULATION_STEPS must be at least 1. Check GLOBAL_BATCH_SIZE and NPROC_PER_NODE." >&2
  exit 1
fi

cd "$MS_SWIFT_ROOT"

# Some multi-node DeepSpeed environments need a local .deepspeed_env symlink.
if [[ -n "$DEEPSPEED_ENV_FILE" ]]; then
  ln -sf "$DEEPSPEED_ENV_FILE" ./.deepspeed_env
fi

deepspeed_args=()
if [[ -n "$HOSTFILE" ]]; then
  deepspeed_args+=(--hostfile "$HOSTFILE")
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

deepspeed "${deepspeed_args[@]}" swift/cli/sft.py \
  --model "$MODEL_PATH" \
  --dataset "$DATASET_JSONL" \
  --system "$SYSTEM_PROMPT" \
  --load_from_cache_file true \
  --split_dataset_ratio "$SPLIT_DATASET_RATIO" \
  --train_type full \
  --torch_dtype bfloat16 \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --freeze_vit true \
  --freeze_aligner true \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --logging_steps "$LOGGING_STEPS" \
  --max_length "$MAX_LENGTH" \
  --truncation_strategy delete \
  --output_dir "$OUTPUT_DIR/$EXP_NAME" \
  --warmup_ratio "$WARMUP_RATIO" \
  --attn_impl flash_attn \
  --lr_scheduler_type cosine \
  --lr_scheduler_kwargs "$LR_SCHEDULER_KWARGS" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --deepspeed "$DS_CONFIG" \
  --report_to tensorboard 2>&1 | tee "$LOG_DIR/${EXP_NAME}_${timestamp}.log"
