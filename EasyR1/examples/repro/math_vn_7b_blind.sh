#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG=${CONFIG:-configs/repro/math_vn_qwen25vl7b_gspo.yaml}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
TRAIN_FILES=${TRAIN_FILES:-${MATH_TRAIN_FILES:-your-org/visual-math-rl@train}}
VAL_FILES=${VAL_FILES:-${MATH_VAL_FILES:-mathverse=your-org/MathVerse-vision-dependent@validation,mmk12=your-org/MMK12-Test@validation}}
GPUS=${GPUS:-8}
NNODES=${NNODES:-1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-math_vn_7b_blind_gspo}

python3 -m verl.trainer.main \
  config="${CONFIG}" \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_image_mask_mode=all_blank \
  data.train_image_mask_ratio=0.0 \
  worker.actor.model.model_path="${MODEL_PATH}" \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${GPUS}" \
  trainer.experiment_name="${EXPERIMENT_NAME}"
