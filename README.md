# 3D Face Reconstruction for Sign Language Data
Submitted in partial fulfillment of the requirements for the MSc Computing (Artificial Intelligence and Machine Learning) of Imperial College London. 

**Author: Shreya Kumar (sk3925)** \
**Supervisor: Dr. Ronglai Zuo**

## Overview
In this project, a new method is proposed for monocular 3D face reconstruction on sign language video. It uses a tokenized vision transformer encoder alongside a temporal transformer with a novel visibility-score based attention mechanism, allowing it to handle hand-over-face occlusions and capture accurate facial expressions.

## Acknowledgements
This work builds on [SMIRK](https://github.com/georgeretsi/smirk) and [TokenFace](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Accurate_3D_Face_Reconstruction_with_Facial_Component_Tokens_ICCV_2023_paper.pdf), with SMIRK's codebase used as a reference and reused in large part.

The work is also evaluated against the following baselines - [EMICA/inferno](https://github.com/radekd91/inferno/tree/master/inferno_apps/FaceReconstruction), [Pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm).

<!-- ## Navigating the codebase
- `assets/` — FLAME model files, landmark embeddings, and other third-party assets required at runtime (gitignored, downloaded/extracted locally).
- `dataset_processing/` — Dataset config, dataloaders, indexers, and manifest schema used to load and prepare training/eval data.
- `datasets/` — Notes on the datasets used (see `datasets/README.md`).
- `evaluation/` — Evaluation pipeline: metrics, landmark extraction, and scripts to run evaluation over datasets/pretrained checkpoints.
- `inference/` — Demo and inference scripts (images, videos, cycle augmentation, occlusion recovery) for running the trained model.
- `model/` — Model architecture: encoder, FLAME decoder, heads, temporal module, and emotion/MICA sub-modules.
- `preprocessing/` — Face cropping and I/O utilities used before frames are fed to the model.
- `samples/` — Sample video/image data used for demos and manual inspection (e.g. CSL-Daily, How2Sign, Phoenix-2014T).
- `scripts/` — One-off/utility scripts for cache prewarming, analysis, and dataset migration.
- `tests/` — Unit and smoke tests for the model, data pipeline, and utilities.
- `training/` — Training loops and config for pretraining and Stage 2 fine-tuning.
- `utils/` — Shared helpers (caching, landmark utilities, kernel smoothing, inference utilities). -->

## Requirements setup

### 1. Python environment & packages
```
cd msc-project
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Post-install fixups
```
python scripts/fix_ibug_packaging.py
uv pip install -e "./uniface[gpu]"
```

### 3. Assets
Follow `assets/readme.md` for how to obtain the required
model/data assets (FLAME, FaRL, MICA, etc.)


## Demo script
<!-- Run demo on a video (how2sign) / series of images -->

## Training - each phase
<!-- Point to how to setup the data -->