import os
import shlex
import shutil
import subprocess

import cv2
import numpy as np

from methods.base import ReconstructionMethod

_MODULE_ENV_PREFIX = (
    'module load cudatoolkit/24.11_12.6 && '
    'export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 && '
)

_CODE_BASE = os.path.abspath('baselines/inferno_experiments')
_VENV_PYTHON = f'{_CODE_BASE}/.venv/bin/python'
_SCRATCH_ROOT = os.path.abspath('evaluation/output/.emica_scratch')


class EmicaMethod(ReconstructionMethod):
    def setup(self, device, crop_size=224, crop_scale=1.4, checkpoint_path=None):
        self.crop_size = crop_size

    def _run(self, args):
        inner = shlex.join(str(a) for a in args)
        cmd = f'{_MODULE_ENV_PREFIX}cd {shlex.quote(_CODE_BASE)} && exec {inner}'
        subprocess.run(['bash', '-lc', cmd], check=True)

    def predict_frames(self, cropped_bgr_frames):
        return self.predict_video(cropped_bgr_frames, cropped_bgr_frames, 'single_frame')

    def predict_video(self, cropped_bgr_frames, raw_bgr_frames, clip_id):
        num_frames = len(cropped_bgr_frames)
        valid_indices = [i for i, f in enumerate(cropped_bgr_frames) if f is not None]
        if not valid_indices:
            return [None] * num_frames

        clip_dir = f'{_SCRATCH_ROOT}/{clip_id}'
        cropped_dir = f'{clip_dir}/cropped'
        out_dir = f'{clip_dir}/out'
        shutil.rmtree(clip_dir, ignore_errors=True)
        os.makedirs(cropped_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        for dense_idx, orig_i in enumerate(valid_indices):
            cv2.imwrite(f'{cropped_dir}/{dense_idx:05d}.jpg', cropped_bgr_frames[orig_i])

        self._run([_VENV_PYTHON, 'scripts/eval_infer.py',
                    '--cropped_dir', cropped_dir, '--out_dir', out_dir,
                    '--crop_size', str(self.crop_size)])

        results = [None] * num_frames
        for dense_idx, orig_i in enumerate(valid_indices):
            fan_path = f'{out_dir}/{dense_idx:05d}_fan.npy'
            mp_path = f'{out_dir}/{dense_idx:05d}_mediapipe.npy'
            vert_path = f'{out_dir}/{dense_idx:05d}_vertices.npy'
            cam_path = f'{out_dir}/{dense_idx:05d}_cam.npy'
            if os.path.exists(fan_path) and os.path.exists(vert_path):
                results[orig_i] = {
                    'fan': np.load(fan_path),
                    'mediapipe': np.load(mp_path) if os.path.exists(mp_path) else None,
                    'vertices': np.load(vert_path),
                    'cam': np.load(cam_path) if os.path.exists(cam_path) else None,
                }

        shutil.rmtree(clip_dir, ignore_errors=True)
        return results

