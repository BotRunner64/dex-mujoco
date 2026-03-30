"""Unified input source adapters for retargeting CLI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hand_detector import HandDetection, HandDetector


@dataclass
class InputFrame:
    detection: HandDetection | None
    preview_frame: np.ndarray | None = None


class MediaPipeInputSource:
    def __init__(
        self,
        source: int | str,
        *,
        target_hand: str | None,
        swap_handedness: bool,
        source_desc: str,
    ):
        self.source_desc = source_desc
        self._frames = HandDetector.create_source(source)
        self._detector = HandDetector(
            target_hand=target_hand,
            swap_handedness=swap_handedness,
        )
        self._available = True

    @property
    def fps(self) -> int:
        return 30

    def is_available(self) -> bool:
        return self._available

    def get_frame(self) -> InputFrame:
        if not self._available:
            raise StopIteration
        try:
            frame = next(self._frames)
        except StopIteration as exc:
            self._available = False
            raise StopIteration from exc
        return InputFrame(
            detection=self._detector.detect(frame),
            preview_frame=frame,
        )

    def annotate_preview(self, frame: np.ndarray, detection: HandDetection) -> np.ndarray:
        return self._detector.draw_landmarks(frame, detection)

    def reset(self) -> bool:
        return False

    def close(self) -> None:
        self._detector.close()

    def stats_snapshot(self) -> dict[str, object]:
        return {}


class HCMocapInputSource:
    def __init__(self, provider: object, *, source_desc: str):
        self.source_desc = source_desc
        self._provider = provider

    @property
    def fps(self) -> int:
        return int(getattr(self._provider, "fps", 30))

    def is_available(self) -> bool:
        return bool(self._provider.is_available())

    def get_frame(self) -> InputFrame:
        return InputFrame(detection=self._provider.get_detection())

    def reset(self) -> bool:
        reset_fn = getattr(getattr(self._provider, "_provider", None), "reset", None)
        if not callable(reset_fn):
            return False
        reset_fn()
        return True

    def close(self) -> None:
        self._provider.close()

    def stats_snapshot(self) -> dict[str, object]:
        stats_fn = getattr(self._provider, "stats_snapshot", None)
        if callable(stats_fn):
            return dict(stats_fn())
        return {}
