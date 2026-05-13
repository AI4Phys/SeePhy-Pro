# SeePhys Pro Reproduction Launchers

These scripts keep the key RLVR runs used in the paper's training-time diagnostic:

- PhysRL physics training, normal vs. blind RL.
- Visual-math control training, normal vs. blind RL.
- Qwen3-VL-4B-Instruct and Qwen2.5-VL-7B-Instruct.

Blind RL is implemented only by replacing training images with black images:
`data.train_image_mask_mode=all_blank`. Validation images remain unmasked in the
data loader.

Replace the placeholder dataset ids in `configs/repro/*.yaml` or override them
from the shell:

```bash
PHYSRL_TRAIN_FILES=Kun-Xiang/PhysRL@train \
PHYSICS_VAL_FILES='seephyspro_l1=your-org/SeePhysPro-L1@validation,seephyspro_l2=your-org/SeePhysPro-L2@validation' \
bash examples/repro/physrl_4b_normal.sh
```

Use `MODEL_PATH`, `TRAIN_FILES`, `VAL_FILES`, `GPUS`, `NNODES`, and
`EXPERIMENT_NAME` to override defaults without editing the scripts.
