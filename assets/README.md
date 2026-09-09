# Assets setup

- **FLAME2020** (`FLAME2020/{generic,male,female}_model.pkl`) \
Register at
   [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/), and then:
   ```
   wget --post-data "username=<email>&password=<pw>" \
     'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' \
     -O assets/FLAME2020.zip --no-check-certificate --continue
   unzip assets/FLAME2020.zip -d assets/FLAME2020/
   ```

- **FLAME_masks** (`FLAME_masks/FLAME_masks.pkl`)
   ```
   wget https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip -O assets/FLAME_masks.zip
   unzip assets/FLAME_masks.zip -d assets/FLAME_masks/
   ```

- **Assets from SMIRK:**
   ```
   git clone https://github.com/georgeretsi/smirk.git /tmp/smirk
   cp /tmp/smirk/assets/landmark_embedding.npy assets/
   cp -r /tmp/smirk/assets/mediapipe_landmark_embedding assets/
   cp /tmp/smirk/assets/l_eyelid.npy /tmp/smirk/assets/r_eyelid.npy assets/
   rm -rf /tmp/smirk
   ```

- **Pretrained FaRL weights**
   (`pretrained_weights/farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth`)
   ```
   mkdir -p pretrained_weights/farl
   wget https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth \
     -O pretrained_weights/farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth
   ```

## Required for Training / Evaluation

6. **expression_templates_famos** (SMIRK's FaMoS-derived training data):
   ```
   gdown --id 1wEL7KPHw2kl5DxP0UAB3h9QcQLXk7BM_ -O assets/expression_templates_famos.zip
   unzip -q assets/expression_templates_famos.zip -d assets/
   ```

7. **ResNet50 (EMOCA emotion checkpoint)**
   (`ResNet50/checkpoints/deca-epoch=01-val_loss_total/dataloader_idx_0=1.27607644.ckpt`) \
   Register
   at [emoca.is.tue.mpg.de](https://emoca.is.tue.mpg.de/), then:
   ```
   wget https://download.is.tue.mpg.de/emoca/assets/EmotionRecognition/image_based_networks/ResNet50.zip \
     -O assets/ResNet50.zip
   unzip assets/ResNet50.zip -d assets/
   ```

8. **mica.tar**
    ```
    wget -O assets/mica.tar "https://keeper.mpdl.mpg.de/f/db172dc4bd4f4c0f96de/?dl=1"
    ```



10. **face_landmarker.task** 
    ```
    wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
      -O assets/face_landmarker.task
    ```

