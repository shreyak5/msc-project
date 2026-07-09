from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class FrameSource(Protocol):
    def num_frames(self) -> int: ...
    def read_frame(self, index: int) -> np.ndarray: ...
    def close(self) -> None: ...


class ImageSequenceFrameSource:
    def __init__(self, image_paths: list[str]):
        self.image_paths = image_paths

    def num_frames(self) -> int:
        return len(self.image_paths)

    def read_frame(self, index: int) -> np.ndarray:
        path = self.image_paths[index]
        image = cv2.imread(path)
        if image is None:
            raise IOError(f"failed to read frame: {path}")
        return image

    def close(self) -> None:
        pass


class VideoFileFrameSource:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self._cap: cv2.VideoCapture | None = None
        self._num_frames: int | None = None

    def _ensure_open(self) -> None:
        if self._cap is None:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise IOError(f"failed to open video: {self.video_path}")
            self._cap = cap

    def num_frames(self) -> int:
        if self._num_frames is None:
            self._ensure_open()
            count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                count = self._count_by_decoding()
            self._num_frames = count
        return self._num_frames

    def _count_by_decoding(self) -> int:
        self._ensure_open()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        count = 0
        while True:
            ret, _ = self._cap.read()
            if not ret:
                break
            count += 1
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return count

    def read_frame(self, index: int) -> np.ndarray:
        self._ensure_open()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()
        actual_pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if not ret or actual_pos != index:
            # Seek landed off-target (codec/keyframe quirk) - fall back to a sequential read
            # from the start, which is always accurate at the cost of being slower.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = None
            for _ in range(index + 1):
                ret, candidate = self._cap.read()
                if not ret:
                    raise IOError(f"failed to read frame {index} of {self.video_path}")
                frame = candidate
        if frame is None:
            raise IOError(f"failed to read frame {index} of {self.video_path}")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self) -> None:
        self.close()


def make_frame_source(image_paths: list[str]) -> FrameSource:
    if len(image_paths) == 1:
        return VideoFileFrameSource(image_paths[0])
    return ImageSequenceFrameSource(image_paths)


def segment_starts(num_frames_total: int, max_frames: int) -> list[int]:
    if num_frames_total <= max_frames:
        return [0]
    return list(range(0, num_frames_total, max_frames))


def frame_indices_for_segment(start: int, max_frames: int, num_frames_total: int) -> list[int]:
    end = min(start + max_frames, num_frames_total)
    indices = list(range(start, end))
    last = indices[-1] if indices else max(num_frames_total - 1, 0)
    while len(indices) < max_frames:
        indices.append(last)
    return indices
