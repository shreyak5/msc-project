#!/bin/bash

#SBATCH --job-name=eval_how2sign_emica
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/eval_how2sign_emica_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=24:00:00

# EMICA is a single feed-forward pass per clip (~48.9s/clip average measured during Phase
# B validation, including its own per-clip model-reload overhead), ~10x faster than
# Pixel3DMM's ~5000-iteration per-clip optimization - fits comfortably in the same
# single-node 4-GPU pattern eval_how2sign.sh already uses for smirk/ours_no_temporal, no
# multi-node complexity needed (unlike slurm_jobs/method_evals/eval_how2sign_pixel3dmm.sh).
#
# run_evaluation_dataset.py resumes automatically if this script is resubmitted after a
# timeout: it detects each shard's already-written results.csv and skips those clips
# rather than redoing them.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
NUM_SHARDS=4
OUTPUT_DIR=evaluation/output_dataset_emica/how2sign

cd "$PROJECT_DIR"

for ((i = 0; i < NUM_SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python evaluation/run_evaluation_dataset.py \
    --input_dir /projects/u6ga/sk3925_datasets/sign_datasets/how2sign/test_rgb_front_clips/ \
    --method emica --crop_size 224 \
    --output_dir "$OUTPUT_DIR" \
    --dataset_name how2sign \
    --num_shards "$NUM_SHARDS" --shard_index $i &
done
wait

.venv/bin/python evaluation/merge_dataset_shards.py --output_dir "$OUTPUT_DIR" --num_shards "$NUM_SHARDS" --method emica

