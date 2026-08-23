#!/bin/bash

#SBATCH --job-name=eval_phoenix2014t_smirk
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/eval_phoenix2014t_smirk_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
NUM_SHARDS=4
OUTPUT_DIR=evaluation/output_dataset/phoenix2014t

cd "$PROJECT_DIR"

for ((i = 0; i < NUM_SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python evaluation/run_evaluation_dataset.py \
    --input_dir /projects/u6ga/sk3925_datasets/sign_datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test \
    --image_seq --fps 25 \
    --method smirk \
    --output_dir "$OUTPUT_DIR" \
    --dataset_name phoenix2014t \
    --num_shards "$NUM_SHARDS" --shard_index $i &
done
wait

.venv/bin/python evaluation/merge_dataset_shards.py --output_dir "$OUTPUT_DIR" --num_shards "$NUM_SHARDS" --method smirk
