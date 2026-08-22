#!/bin/bash
set -e

START=$SECONDS

.venv/bin/python evaluation/run_evaluation_dataset.py \
  --input_dir /projects/u6ga/sk3925_datasets/sign_datasets/csl-daily/test \
  --output_dir evaluation/output_dataset_pretrain/csl-daily \
  --image_seq --fps 30 \
  --method ours_no_temporal \
  --checkpoint /projects/u6ga/sk3925_misc/checkpoints/pretrain/step_00059999.pt

ELAPSED=$((SECONDS - START))
printf 'Total time taken: %02d:%02d:%02d\n' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
