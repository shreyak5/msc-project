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
    def setup(self, device, crop_size=224, crop_scale=1.4, checkpoint_path=None):
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
        finally:
            os.chdir(original_cwd)

    def _project_to_pixels(self, landmarks, cam):
        projected = self._batch_orth_proj(landmarks, cam)
        projected = projected.clone()
        projected[:, :, 1:] = -projected[:, :, 1:]
        projected = projected[..., :2]
        return (projected + 1) * (self.crop_size / 2)

    @torch.no_grad()
    def predict_frames(self, cropped_bgr_frames):
        images = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (224, 224)) for f in cropped_bgr_frames]
        image_tensor = torch.tensor(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        outputs = self.smirk_encoder(image_tensor)
        flame_output = self.flame.forward(outputs)

        fan_pixels = self._project_to_pixels(flame_output['landmarks_fan'], outputs['cam'])
        mp_pixels = self._project_to_pixels(flame_output['landmarks_mp'], outputs['cam'])

        fan_pixels = fan_pixels.cpu().numpy()
        mp_pixels = mp_pixels.cpu().numpy()
        vertices = flame_output['vertices'].cpu().numpy()
        cam = outputs['cam'].cpu().numpy()
        return [
            {'fan': fan_pixels[i], 'mediapipe': mp_pixels[i], 'vertices': vertices[i], 'cam': cam[i]}
            for i in range(len(cropped_bgr_frames))
        ]
