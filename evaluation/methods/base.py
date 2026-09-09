from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from model import constants

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MEDIAPIPE_EMBEDDING_PATH = _REPO_ROOT / constants.FLAME_MEDIAPIPE_LMK_EMBEDDING_PATH


class ReconstructionMethod(ABC):
    """Interface a 3D reconstruction method implements to be evaluated by run_evaluation.py.

    Cropping is not part of this interface: run_evaluation.py crops each frame once and
    hands the same crop to the reference (FAN/MediaPipe) detectors and to the method.
    """

    # None = score landmarks at whatever crop_size the method was set up with (default).
    # A method overrides this to a fixed pixel size when its own crop_size is dictated by
    # its own pipeline's requirements rather than being a fair scoring resolution (see
    # Pixel3dmmMethod, whose 512px tracking crop would otherwise inflate its raw-pixel
    # landmark error relative to every other method's 224px crop) - eval_core.py's
    # evaluate_clip() downscales the crop and rescales predicted landmarks to this size
    # before computing error against it.
    gt_crop_size = None

    @abstractmethod
    def setup(self, device, **method_params):
        """Load models/checkpoints. Called once before any predict_* calls."""

    @abstractmethod
    def predict_frames(self, cropped_bgr_frames):
        """Run the method on a batch of independent cropped BGR frames, one forward pass.

        cropped_bgr_frames: list[np.ndarray], no Nones - N independent crops with no
        temporal coupling between them. Returns list[dict], same length/order, each
        e.g. {'fan': (68, 2) or None, 'mediapipe': (105, 2) or None, 'vertices': (V, 3)
        or None}, with 2D landmarks in pixel space of the crop and 3D mesh vertices in
        the method's own mesh space. 'vertices' is optional: if a method's output omits
        it, the temporal_smoothness metric just reports no data (NaN) for that method
        rather than erroring. Each sample's keys must come from a single forward pass -
        do not re-run inference per metric.

        Backs both predict_frame (a batch of 1) and the default predict_video (below)
        for methods without real temporal processing.
        """

    def predict_frame(self, cropped_bgr_image):
        """Convenience wrapper: run predict_frames on a single cropped BGR frame."""
        return self.predict_frames([cropped_bgr_image])[0]

    def predict_video(self, cropped_bgr_frames, raw_bgr_frames, clip_id):
        valid_indices = [i for i, f in enumerate(cropped_bgr_frames) if f is not None]
        if not valid_indices:
            return [None] * len(cropped_bgr_frames)

        batch_preds = self.predict_frames([cropped_bgr_frames[i] for i in valid_indices])

        results = [None] * len(cropped_bgr_frames)
        for i, pred in zip(valid_indices, batch_preds):
            results[i] = pred
        return results

    def mediapipe_gt_indices(self):
        return np.load(_MEDIAPIPE_EMBEDDING_PATH)["landmark_indices"]
