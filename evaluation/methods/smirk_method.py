import inspect
import os
import sys

import cv2
import numpy as np
import torch

from methods.base import ReconstructionMethod

# chumpy (used to unpickle the FLAME model) was written for Python <3.11, where
# inspect.getargspec still existed. Shim it rather than pin an old interpreter.
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

SMIRK_ROOT = 'baselines/smirk_experiments'
SMIRK_CHECKPOINT = 'baselines/smirk_experiments/pretrained_models/SMIRK_em1.pt'


class SmirkMethod(ReconstructionMethod):
    def setup(self, device, crop_size=224):
        self.device = device
        self.crop_size = crop_size

        smirk_root_abs = os.path.abspath(SMIRK_ROOT)
        checkpoint_abs = os.path.abspath(SMIRK_CHECKPOINT)

        sys.path.insert(0, smirk_root_abs)
        original_cwd = os.getcwd()
        os.chdir(smirk_root_abs)
        try:
            from src.smirk_encoder import SmirkEncoder
            from src.FLAME.FLAME import FLAME
            from src.renderer.util import batch_orth_proj

            self._batch_orth_proj = batch_orth_proj

            self.smirk_encoder = SmirkEncoder().to(device)
            checkpoint_state = torch.load(checkpoint_abs, map_location=device)
            encoder_state = {
                k.replace('smirk_encoder.', ''): v
                for k, v in checkpoint_state.items() if 'smirk_encoder' in k
            }
            self.smirk_encoder.load_state_dict(encoder_state)
            self.smirk_encoder.eval()

            self.flame = FLAME().to(device)

            mp_embedding = np.load('assets/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz')
            self.landmark_indices = mp_embedding['landmark_indices']
        finally:
            os.chdir(original_cwd)

    def _project_to_pixels(self, landmarks, cam):
        projected = self._batch_orth_proj(landmarks, cam)
        projected = projected.clone()
        projected[:, :, 1:] = -projected[:, :, 1:]
        projected = projected[..., :2]
        return (projected + 1) * (self.crop_size / 2)

    @torch.no_grad()
    def predict(self, cropped_bgr_image):
        image = cv2.cvtColor(cropped_bgr_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        outputs = self.smirk_encoder(image_tensor)
        flame_output = self.flame.forward(outputs)

        fan_pixels = self._project_to_pixels(flame_output['landmarks_fan'], outputs['cam'])
        mp_pixels = self._project_to_pixels(flame_output['landmarks_mp'], outputs['cam'])

        return {
            'fan': fan_pixels[0].cpu().numpy(),
            'mediapipe': mp_pixels[0].cpu().numpy(),
            'vertices': flame_output['vertices'][0].cpu().numpy(),
        }

    def mediapipe_gt_indices(self):
        return self.landmark_indices
