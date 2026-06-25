from abc import ABC, abstractmethod


class ReconstructionMethod(ABC):
    """Interface a 3D reconstruction method implements to be evaluated by run_evaluation.py.

    Cropping is not part of this interface: run_evaluation.py crops each frame once and
    hands the same crop to the reference (FAN/MediaPipe) detectors and to predict().
    """

    @abstractmethod
    def setup(self, device, **method_params):
        """Load models/checkpoints. Called once before any predict() calls."""

    @abstractmethod
    def predict(self, cropped_bgr_image):
        """Run the method on a single cropped BGR frame.

        Returns a dict, e.g. {'fan': (68, 2) or None, 'mediapipe': (105, 2) or None,
        'vertices': (V, 3) or None}, with 2D landmarks in pixel space of the crop and
        3D mesh vertices in the method's own mesh space. 'vertices' is optional: if a
        method's predict() omits it, the temporal_smoothness metric just reports no
        data (NaN) for that method rather than erroring. All keys must come from a
        single forward pass - do not re-run inference per metric.
        """

    def mediapipe_gt_indices(self):
        """Indices into the full 478-point MediaPipe GT array matching predict()'s
        'mediapipe' output order, or None if predict() already returns the full 478."""
        return None
