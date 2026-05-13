#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG=${CONFIG:-configs/repro/physrl_qwen3vl4b_gspo.yaml}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}
TRAIN_FILES=${TRAIN_FILES:-${PHYSRL_TRAIN_FILES:-Kun-Xiang/PhysRL@train}}
VAL_FILES=${VAL_FILES:-${PHYSICS_VAL_FILES:-seephyspro_l1=your-org/SeePhysPro-L1@validation,seephyspro_l2=your-org/SeePhysPro-L2@validation,seephyspro_l3=your-org/SeePhysPro-L3@validation,seephyspro_l4=your-org/SeePhysPro-L4@validation,physreason=your-org/PhysReason-val@validation,phyx=your-org/PhyX-val@validation}}
GPUS=${GPUS:-4}
NNODES=${NNODES:-1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-physrl_4b_blind_gspo}

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
