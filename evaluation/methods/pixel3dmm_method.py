"""Pixel3DMM evaluation method.

Unlike SmirkMethod/OursFullMethod (both in-process, single forward pass per clip in this
project's own venv), Pixel3DMM needs its own incompatible python/CUDA stack (see
baselines/pixel3dmm_experiments/ - a separate uv-managed venv, python 3.9, torch+cu126,
pytorch3d/nvdiffrast built from source) and is architecturally an offline, multi-stage,
per-clip optimization (~5000 iterations) that writes to disk rather than returning a
value from a function call. predict_video() here orchestrates that whole pipeline via
subprocess calls into the separate venv, using the *same* RetinaFace crop eval_core.py
already produced for every other method (see baselines/pixel3dmm_experiments/scripts/
run_pipnet_landmarks_only.py's docstring for why Pixel3DMM's own internal cropper can be
bypassed safely), then reads back plain .npy arrays written by the separate venv's own
scripts/export_eval_outputs.py - this process never imports anything from the pixel3dmm
package itself, only numpy/cv2/subprocess.
"""

import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

import cv2
import numpy as np

from methods.base import ReconstructionMethod
from model import constants

# Matches configs/tracking.yaml's defaults (num_views=1, no_pho=True, all others False/0,
# uv_map_super=2000.0, normal_super=1000.0) via Tracker.__init__'s actor_name suffix
# construction (tracker.py:137,160-180) - predict_video() below never overrides these when
# invoking track.py, so this is deterministic. Would need updating if that ever changes.
_ACTOR_NAME_SUFFIX = '_nV1_noPho_uv2000.0_n1000.0'

# Same asset base.py's own mediapipe_gt_indices() default loads (see that docstring) -
# pure barycentric coords into stock FLAME2020 topology, not tied to any project's
# trained network, so it applies to Pixel3DMM's FLAME output unmodified (verified:
# Pixel3DMM's FLAME.py loads the same FLAME2020/generic_model.pkl, 5023 verts/9976
# faces, comfortably covering this embedding's face indices).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MEDIAPIPE_EMBEDDING_PATH = _REPO_ROOT / constants.FLAME_MEDIAPIPE_LMK_EMBEDDING_PATH

_MODULE_ENV_PREFIX = (
    'module load cudatoolkit/24.11_12.6 && '
    'export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 && '
)


def _read_pixel3dmm_env():
    """Minimal parser for ~/.config/pixel3dmm/.env (the same file the separate Pixel3DMM
    venv's own `environs`-based env_paths.py reads) - avoids needing that package, or any
    of pixel3dmm's own dependencies, importable in this project's own venv."""
    path = os.path.expanduser('~/.config/pixel3dmm/.env')
    values = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            values[key.strip()] = value.strip()
    return values


class Pixel3dmmMethod(ReconstructionMethod):
    # Pixel3DMM's own tracking pipeline needs its native 512px crop (see setup()'s
    # crop_size), but that makes its raw-pixel landmark error ~2.3x larger than every
    # other method's (all evaluated at 224px) purely from resolution, not accuracy - see
    # base.py's gt_crop_size docstring. Always score at 224px regardless of crop_size.
    gt_crop_size = 224

    def setup(self, device, crop_size=224, crop_scale=1.4, checkpoint_path=None,
              preprocessed_data=None, tracking_output=None):
        env = _read_pixel3dmm_env()
        self.code_base = env['PIXEL3DMM_CODE_BASE'].rstrip('/')
        self.preprocessed_data = (preprocessed_data or env['PIXEL3DMM_PREPROCESSED_DATA']).rstrip('/')
        self.tracking_output = (tracking_output or env['PIXEL3DMM_TRACKING_OUTPUT']).rstrip('/')
        self.venv_python = f'{self.code_base}/.venv/bin/python'
        self.crop_size = crop_size

    def _run(self, workdir, args):
        inner = shlex.join(str(a) for a in args)
        cmd = f'{_MODULE_ENV_PREFIX}cd {shlex.quote(workdir)} && exec {inner}'
        subprocess.run(['bash', '-lc', cmd], check=True)

    def _run_pipeline(self, clip_id):
        scripts_dir = f'{self.code_base}/scripts'
        mica_dir = f'{self.code_base}/src/pixel3dmm/preprocessing/MICA'
        py = self.venv_python

        # --video_name/-video_name passed as a single '=' joined token, not two separate
        # argv entries: how2sign clip_ids are YouTube-id-derived and can start with '-'
        # (e.g. '-fZc293MpJk_2-1-rgb_front'), which a two-token '--video_name', clip_id
        # would make argparse/Typer misparse as an unrelated flag rather than this
        # option's value - '--video_name=-fZc...'/'​-video_name=-fZc...' parses correctly
        # regardless of what clip_id starts with.
        self._run(scripts_dir, [py, 'run_pipnet_landmarks_only.py', f'--video_name={clip_id}'])
        self._run(mica_dir, [py, 'demo.py', f'-video_name={clip_id}',
                              '-a', f'{self.preprocessed_data}/{clip_id}/arcface/'])
        self._run(scripts_dir, [py, 'run_facer_segmentation.py', f'--video_name={clip_id}'])
        self._run(self.code_base, [py, 'scripts/network_inference.py',
                                    'model.prediction_type=normals', f'video_name={clip_id}'])
        self._run(self.code_base, [py, 'scripts/network_inference.py',
                                    'model.prediction_type=uv_map', f'video_name={clip_id}'])
        self._run(self.code_base, [py, 'scripts/track.py', f'video_name={clip_id}'])
        self._run(self.code_base, [py, 'scripts/export_eval_outputs.py',
                                    f'--video_name={clip_id}', '--size', str(self.crop_size),
                                    '--mediapipe_embedding_path', str(_MEDIAPIPE_EMBEDDING_PATH)])

    def predict_frames(self, cropped_bgr_frames):
        # Pixel3DMM has no meaningful independent-frame mode (its optimization is
        # inherently per-clip), so this always goes through predict_video with each
        # frame treated as its own length-1 clip - slow, but only exercised by callers
        # that explicitly want a single-frame prediction rather than a whole clip.
        #
        # clip_id must be unique per call: predict_video()'s already_exported check only
        # tests file existence at tracking_output/{clip_id}.../eval_export/, not whether
        # those files match this call's actual input - a constant clip_id here would let
        # a later call silently read back an earlier, unrelated call's cached prediction
        # instead of ever re-running the optimization (see local_runs/landmark_debug's
        # single_frame_overlay.py incident: every call used the literal 'single_frame'
        # and kept re-reading one stale Aug-18 export for every subsequent frame).
        clip_id = f'single_frame_{uuid.uuid4().hex}'
        return self.predict_video(cropped_bgr_frames, cropped_bgr_frames, clip_id)

    def predict_video(self, cropped_bgr_frames, raw_bgr_frames, clip_id):
        num_frames = len(cropped_bgr_frames)
        valid_indices = [i for i, f in enumerate(cropped_bgr_frames) if f is not None]
        if not valid_indices:
            return [None] * num_frames

        export_dir = f'{self.tracking_output}/{clip_id}{_ACTOR_NAME_SUFFIX}/eval_export'
        already_exported = all(
            os.path.exists(f'{export_dir}/{dense_idx:05d}_lmk68_2d.npy')
            and os.path.exists(f'{export_dir}/{dense_idx:05d}_vertices.npy')
            for dense_idx in range(len(valid_indices))
        )

        if not already_exported:
            clip_dir = f'{self.preprocessed_data}/{clip_id}'
            cropped_dir = f'{clip_dir}/cropped'
            shutil.rmtree(cropped_dir, ignore_errors=True)
            os.makedirs(cropped_dir, exist_ok=True)
            for dense_idx, orig_i in enumerate(valid_indices):
                cv2.imwrite(f'{cropped_dir}/{dense_idx:05d}.jpg', cropped_bgr_frames[orig_i])

            self._run_pipeline(clip_id)

        results = [None] * num_frames
        for dense_idx, orig_i in enumerate(valid_indices):
            lmk_path = f'{export_dir}/{dense_idx:05d}_lmk68_2d.npy'
            vert_path = f'{export_dir}/{dense_idx:05d}_vertices.npy'
            mp_path = f'{export_dir}/{dense_idx:05d}_mp105_2d.npy'
            vert_screen_path = f'{export_dir}/{dense_idx:05d}_vertices_screen.npy'
            if os.path.exists(lmk_path) and os.path.exists(vert_path):
                results[orig_i] = {
                    'fan': np.load(lmk_path),
                    'mediapipe': np.load(mp_path) if os.path.exists(mp_path) else None,
                    'vertices': np.load(vert_path),
                    'vertices_screen': np.load(vert_screen_path) if os.path.exists(vert_screen_path) else None,
                }
        return results

