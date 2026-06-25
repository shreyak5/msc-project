#!/bin/bash

#SBATCH --job-name=eval_phoenix2014t
#SBATCH --output=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project/slurm_jobs/output/eval_phoenix2014t_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=10:00:00

PROJECT_DIR=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project
OUTPUT_DIR=evaluation/output_dataset/phoenix2014t
NUM_SHARDS=4

cd "$PROJECT_DIR"

for ((i = 0; i < NUM_SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python evaluation/run_evaluation_dataset.py \
    --input_dir /projects/u6kf/sk3925_datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test \
    --image_seq --fps 25 \
    --output_dir "$OUTPUT_DIR" \
    --num_shards "$NUM_SHARDS" --shard_index $i &
done
wait

.venv/bin/python evaluation/merge_dataset_shards.py --output_dir "$OUTPUT_DIR" --num_shards "$NUM_SHARDS"
