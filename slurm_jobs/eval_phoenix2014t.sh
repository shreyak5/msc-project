#!/bin/bash

#SBATCH --job-name=eval_phoenix2014t
#SBATCH --output=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project/slurm_jobs/output/eval_phoenix2014t_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=02:00:00

PROJECT_DIR=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project
NUM_SHARDS=4

### SMIRK
# OUTPUT_DIR=evaluation/output_dataset/phoenix2014t

### Pretrain
OUTPUT_DIR=evaluation/output_dataset_pretrain/phoenix2014t

cd "$PROJECT_DIR"

for ((i = 0; i < NUM_SHARDS; i++)); do
  ### SMIRK
  # CUDA_VISIBLE_DEVICES=$i .venv/bin/python evaluation/run_evaluation_dataset.py \
  #   --input_dir /projects/u6kf/sk3925_datasets/sign_datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test \
  #   --image_seq --fps 25 \
  #   --output_dir "$OUTPUT_DIR" \
  #   --num_shards "$NUM_SHARDS" --shard_index $i &

  ### Pretrain
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python evaluation/run_evaluation_dataset.py \
    --input_dir /projects/u6kf/sk3925_datasets/sign_datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test \
    --image_seq --fps 25 \
    --output_dir "$OUTPUT_DIR" \
    --method ours_no_temporal \
    --checkpoint /projects/u6kf/sk3925_misc/checkpoints/pretrain/step_00059999.pt \
    --num_shards "$NUM_SHARDS" --shard_index $i &
done
wait

.venv/bin/python evaluation/merge_dataset_shards.py --output_dir "$OUTPUT_DIR" --num_shards "$NUM_SHARDS" --method ours_no_temporal
