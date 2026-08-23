#!/bin/bash

#SBATCH --job-name=eval_how2sign_ours_kernel_smooth
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/eval_how2sign_ours_kernel_smooth_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=20:00:00

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
NUM_SHARDS=4
CHECKPOINT=/projects/u6ga/sk3925_misc/checkpoints/stage2_50_AAB_UNet100_v3/step_00025999.pt
RADIUS=4
# 4 combos (radius fixed): sigma x temperature. Run sequentially, sharing the
# same 4 GPUs across combos - see evaluation/methods/ours_method.py's
# OursKernelSmoothMethod cache (SViT only recomputed on the first combo) and
# evaluation/gt_cache.py (crop/GT-landmark caches, shared with every method)
# for why combos 2-4 are much cheaper than combo 1.
SIGMAS=(0.5 0.5 1 1)
TEMPERATURES=(0.1 0.05 0.1 0.05)

cd "$PROJECT_DIR"

for combo_idx in "${!SIGMAS[@]}"; do
  SIGMA=${SIGMAS[$combo_idx]}
  TEMPERATURE=${TEMPERATURES[$combo_idx]}
  OUTPUT_DIR=evaluation/output_dataset_ours_kernel_smooth/how2sign/r${RADIUS}_sigma${SIGMA}_T${TEMPERATURE}

  for ((i = 0; i < NUM_SHARDS; i++)); do
    CUDA_VISIBLE_DEVICES=$i .venv/bin/python evaluation/run_evaluation_dataset.py \
      --input_dir /projects/u6ga/sk3925_datasets/sign_datasets/how2sign/test_rgb_front_clips/ \
      --output_dir "$OUTPUT_DIR" \
      --method ours_kernel_smooth \
      --checkpoint "$CHECKPOINT" \
      --dataset_name how2sign \
      --kernel_radius "$RADIUS" --kernel_sigma "$SIGMA" --kernel_temperature "$TEMPERATURE" \
      --num_shards "$NUM_SHARDS" --shard_index $i &
  done
  wait

  .venv/bin/python evaluation/merge_dataset_shards.py --output_dir "$OUTPUT_DIR" --num_shards "$NUM_SHARDS" --method ours_kernel_smooth
done

