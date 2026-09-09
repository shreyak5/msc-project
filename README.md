# 3D Face Reconstruction for Sign Language Data
Submitted in partial fulfillment of the requirements for the MSc Computing (Artificial Intelligence and Machine Learning) of Imperial College London. 

**Author: Shreya Kumar (sk3925)** \
**Supervisor: Dr. Ronglai Zuo**

## Overview
In this project, a new method is proposed for monocular 3D face reconstruction on sign language video. It uses a tokenized vision transformer encoder alongside a temporal transformer with a novel visibility-score based attention mechanism, allowing it to handle hand-over-face occlusions and capture accurate facial expressions.

## Acknowledgements
This work builds on [SMIRK](https://github.com/georgeretsi/smirk) and [TokenFace](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Accurate_3D_Face_Reconstruction_with_Facial_Component_Tokens_ICCV_2023_paper.pdf), with SMIRK's codebase used as a reference and reused in large part.

The work is also evaluated against the following baselines - [EMICA/inferno](https://github.com/radekd91/inferno/tree/master/inferno_apps/FaceReconstruction), [Pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm).

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
- Download the final checkpoints from [here](https://imperiallondon-my.sharepoint.com/:f:/g/personal/sk3925_ic_ac_uk/IgCspEpLkxl_QoVCSS30AhdaARGzKwiIwf6xiRLVMPd6X0s)
- Output for the demos will be written under `inference/output/demo/`.

### Demo on the Full Model
```
python inference/demo_videos.py --input_path samples/how2sign.mp4 --checkpoint final_checkpoints/full_model.pt
```

### Demo on the Spatial-Only Model (no temporal transformer)
```
python inference/demo_videos.py --input_path samples/how2sign.mp4 --checkpoint final_checkpoints/spatial_only_model.pt --no_tt
```

### Running on a directory of frames
Datasets like CSL-Daily and PHOENIX-2014T, have their videos stored as sequence of individual image frames. The demo for such inputs can be run as: 
```
python inference/demo_videos.py --input_path samples/csl-daily-sample --image_seq --fps 30 --checkpoint final_checkpoints/full_model.pt
```


## Training

### 1. Prepare the datasets
Download the datasets listed in `datasets/README.md` and run the data preparation / preprocessing steps as explained.

### 2. Pretrain 

    python -m training.pretrain --config training/config/pretrain.yaml

### 3. Main Training

    python -m training.stage2 --config training/config/main_training.yaml
