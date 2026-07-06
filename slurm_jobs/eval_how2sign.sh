#!/bin/bash

#SBATCH --job-name=eval_how2sign
#SBATCH --output=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project/slurm_jobs/output/eval_how2sign_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=10:00:00

PROJECT_DIR=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project
OUTPUT_DIR=evaluation/output_dataset/how2sign
NUM_SHARDS=4

cd "$PROJECT_DIR"

for ((i = 0; i < NUM_SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python evaluation/run_evaluation_dataset.py \
    --input_dir /projects/u6kf/sk3925_datasets/sign_datasets/how2sign/test_rgb_front_clips/ \
    --output_dir "$OUTPUT_DIR" \
    --num_shards "$NUM_SHARDS" --shard_index $i &
done
wait

.venv/bin/python evaluation/merge_dataset_shards.py --output_dir "$OUTPUT_DIR" --num_shards "$NUM_SHARDS"
