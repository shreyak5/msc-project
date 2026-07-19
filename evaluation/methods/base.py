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
        """Run the method on a whole clip's frames.

        Default (no temporal context): batches every successfully-cropped frame into
        one predict_frames() call - not a Python loop of N separate forward passes -
        and re-inserts None at indices whose crop failed. raw_bgr_frames/clip_id are
        unused here; only needed by methods with real temporal/visibility processing,
        which override this (e.g. TT-based ones - they need raw_bgr_frames because
        visibility/mask scoring requires its own independent face detection on the
        un-cropped frame, which this default per-frame path never computes at all).

        cropped_bgr_frames: list[np.ndarray|None], one per clip frame (None = crop
        failed for that frame). raw_bgr_frames: list[np.ndarray], the same clip's
        frames before cropping, same length/order. clip_id: a stable string naming
        this clip. Returns list[dict|None], same length/order as cropped_bgr_frames.
        """
        valid_indices = [i for i, f in enumerate(cropped_bgr_frames) if f is not None]
        if not valid_indices:
            return [None] * len(cropped_bgr_frames)

        batch_preds = self.predict_frames([cropped_bgr_frames[i] for i in valid_indices])

        results = [None] * len(cropped_bgr_frames)
        for i, pred in zip(valid_indices, batch_preds):
            results[i] = pred
        return results

    def mediapipe_gt_indices(self):
        """Indices into the full 478-point MediaPipe GT array matching predict_frames()'s
        'mediapipe' output order. Every method currently in this framework (SMIRK and
        this project's own model) decodes MediaPipe landmarks via the same curated
        105-point FLAME/DECA/EMOCA-lineage embedding - SMIRK's own copy under
        baselines/smirk_experiments/assets/ is byte-identical (verified) to this
        project's assets/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz
        - so this loads the latter once here as the shared default. Override only if a
        method's predict_frames() output is ordered differently, or already returns the
        full 478 points (in which case override to return None instead)."""
        return np.load(_MEDIAPIPE_EMBEDDING_PATH)["landmark_indices"]
