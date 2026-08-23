#!/bin/bash

#SBATCH --job-name=render_orig_demo
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/render_orig_demo_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00

# One-off demo_videos.py sweep across 4 checkpoints x 7 sample clips, all with
# --render_orig. Submitted as a real sbatch job (not srun --overlap on an interactive
# session) specifically because the equivalent srun --overlap runs kept getting killed
# when the user's SSH session logged out - sbatch jobs run independently of any login
# session, immune to that.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
cd "$PROJECT_DIR"

run_seven() {
  local CKPT="$1"
  local OUT="$2"
  .venv/bin/python inference/demo_videos.py --input_path /projects/u6ga/sk3925_datasets/sign_datasets/csl-daily/test/S000040_P0008_T00 --image_seq --fps 30 --checkpoint "$CKPT" --render_2d_recon --render_orig --out_path "$OUT"
  .venv/bin/python inference/demo_videos.py --input_path /projects/u6ga/sk3925_datasets/sign_datasets/csl-daily/test/S000185_P0004_T00 --image_seq --fps 30 --checkpoint "$CKPT" --render_2d_recon --render_orig --out_path "$OUT"
  .venv/bin/python inference/demo_videos.py --input_path /projects/u6ga/sk3925_datasets/sign_datasets/csl-daily/test/S000201_P0000_T00 --image_seq --fps 30 --checkpoint "$CKPT" --render_2d_recon --render_orig --out_path "$OUT"
  .venv/bin/python inference/demo_videos.py --input_path /projects/u6ga/sk3925_datasets/sign_datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test/01April_2011_Friday_tagesschau-3374 --image_seq --fps 25 --checkpoint "$CKPT" --render_2d_recon --render_orig --out_path "$OUT"
  .venv/bin/python inference/demo_videos.py --input_path /projects/u6ga/sk3925_datasets/sign_datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test/01April_2011_Friday_tagesschau-3377 --image_seq --fps 25 --checkpoint "$CKPT" --render_2d_recon --render_orig --out_path "$OUT"
  .venv/bin/python inference/demo_videos.py --input_path /projects/u6ga/sk3925_datasets/sign_datasets/how2sign/test_rgb_front_clips/_FzvMVnR_aU_2-10-rgb_front.mp4 --checkpoint "$CKPT" --render_2d_recon --render_orig --out_path "$OUT"
  .venv/bin/python inference/demo_videos.py --input_path /projects/u6ga/sk3925_datasets/sign_datasets/how2sign/test_rgb_front_clips/_FzvMVnR_aU_3-10-rgb_front.mp4 --checkpoint "$CKPT" --render_2d_recon --render_orig --out_path "$OUT"
}

run_seven /projects/u6ga/sk3925_misc/checkpoints/stage2_50_unet_warmup/step_00007999.pt inference/output/unet_warmup_50
run_seven /projects/u6ga/sk3925_misc/checkpoints/pretrain/step_00059999.pt inference/output/pretrain
run_seven /projects/u6ga/sk3925_misc/checkpoints/pretrain_50/step_00059999.pt inference/output/pretrain_50
run_seven /projects/u6ga/sk3925_misc/checkpoints/stage2_unet_warmup/step_00007999.pt inference/output/unet_warmup

echo "ALL_DONE"

