"""Fixed-rate sampling wrappers for live tracking sources."""

from __future__ import annotations

import time
from threading import Lock

from somehand.core import BiHandFrame, BiHandSourceFrame, HandFrame, SourceFrame

from .source_transforms import copy_bihand_frame, copy_hand_frame


class FixedRateHandTrackingSource:
    """Samples a hand source at a stable output cadence without filtering."""

    def __init__(self, wrapped_source: object, *, sample_fps: int | None = None):
        resolved_fps = int(getattr(wrapped_source, "fps", 30)) if sample_fps is None else int(sample_fps)
        if resolved_fps <= 0:
            raise ValueError("sample_fps must be > 0")

        self._wrapped_source = wrapped_source
        self._fps = resolved_fps
        self._sample_period = 1.0 / float(resolved_fps)
        self._next_sample_at: float | None = None
        self._last_raw_frame_index = 0
        self._last_output: HandFrame | None = None
        self._output_frame_index = 0
        self._snapshot_lock = Lock()

    def __getattr__(self, name: str):
        return getattr(self._wrapped_source, name)

    @property
    def fps(self) -> int:
        return self._fps

    def is_available(self) -> bool:
        return bool(self._wrapped_source.is_available())

    def get_frame(self) -> SourceFrame:
        self._wait_for_next_sample()
        detection = self._sample_detection()
        with self._snapshot_lock:
            self._output_frame_index += 1
            self._last_output = None if detection is None else copy_hand_frame(detection)
        return SourceFrame(detection=None if detection is None else copy_hand_frame(detection))

    def latest_hand_frame_snapshot(self) -> tuple[int, HandFrame] | None:
        with self._snapshot_lock:
            if self._last_output is None or self._output_frame_index <= 0:
                return None
            return self._output_frame_index, copy_hand_frame(self._last_output)

    def reset(self) -> bool:
        self._next_sample_at = None
        self._last_raw_frame_index = 0
        with self._snapshot_lock:
            self._last_output = None
            self._output_frame_index = 0
        return bool(self._wrapped_source.reset())

    def close(self) -> None:
        self._wrapped_source.close()

    def stats_snapshot(self) -> dict[str, object]:
        stats_fn = getattr(self._wrapped_source, "stats_snapshot", None)
        base_stats = {} if not callable(stats_fn) else dict(stats_fn())
        base_stats["sample_fps"] = self._fps
        return base_stats

    def _wait_for_next_sample(self) -> None:
        now = time.monotonic()
        if self._next_sample_at is None:
            self._next_sample_at = now
            return

        target_time = self._next_sample_at + self._sample_period
        if target_time > now:
            time.sleep(target_time - now)
        else:
            target_time = now
        self._next_sample_at = target_time

    def _sample_detection(self) -> HandFrame | None:
        snapshot_fn = getattr(self._wrapped_source, "latest_hand_frame_snapshot", None)
        if callable(snapshot_fn):
            snapshot = snapshot_fn()
            if snapshot is not None:
                frame_index, frame = snapshot
                if frame_index > self._last_raw_frame_index:
                    self._last_raw_frame_index = frame_index
                    return copy_hand_frame(frame)
                if self._last_output is not None:
                    return copy_hand_frame(self._last_output)

        frame = self._wrapped_source.get_frame()
        if frame.detection is None:
            return None
        return copy_hand_frame(frame.detection)


class FixedRateBiHandTrackingSource:
    """Samples a bi-hand source at a stable output cadence without filtering."""

    def __init__(self, wrapped_source: object, *, sample_fps: int | None = None):
        resolved_fps = int(getattr(wrapped_source, "fps", 30)) if sample_fps is None else int(sample_fps)
        if resolved_fps <= 0:
            raise ValueError("sample_fps must be > 0")

        self._wrapped_source = wrapped_source
        self._fps = resolved_fps
        self._sample_period = 1.0 / float(resolved_fps)
        self._next_sample_at: float | None = None
        self._last_raw_frame_index = 0
        self._last_output: BiHandFrame | None = None
        self._output_frame_index = 0
        self._snapshot_lock = Lock()

    def __getattr__(self, name: str):
        return getattr(self._wrapped_source, name)

    @property
    def fps(self) -> int:
        return self._fps

    def is_available(self) -> bool:
        return bool(self._wrapped_source.is_available())

    def get_frame(self) -> BiHandSourceFrame:
        self._wait_for_next_sample()
        detection = self._sample_detection()
        with self._snapshot_lock:
            self._output_frame_index += 1
            self._last_output = None if detection is None else copy_bihand_frame(detection)
        return BiHandSourceFrame(detection=None if detection is None else copy_bihand_frame(detection))

    def latest_bihand_frame_snapshot(self) -> tuple[int, BiHandFrame] | None:
        with self._snapshot_lock:
            if self._last_output is None or self._output_frame_index <= 0:
                return None
            return self._output_frame_index, copy_bihand_frame(self._last_output)

    def reset(self) -> bool:
        self._next_sample_at = None
        self._last_raw_frame_index = 0
        with self._snapshot_lock:
            self._last_output = None
            self._output_frame_index = 0
        return bool(self._wrapped_source.reset())

    def close(self) -> None:
        self._wrapped_source.close()

    def stats_snapshot(self) -> dict[str, object]:
        stats_fn = getattr(self._wrapped_source, "stats_snapshot", None)
        base_stats = {} if not callable(stats_fn) else dict(stats_fn())
        base_stats["sample_fps"] = self._fps
        return base_stats

    def _wait_for_next_sample(self) -> None:
        now = time.monotonic()
        if self._next_sample_at is None:
            self._next_sample_at = now
            return

        target_time = self._next_sample_at + self._sample_period
        if target_time > now:
            time.sleep(target_time - now)
        else:
            target_time = now
        self._next_sample_at = target_time

    def _sample_detection(self) -> BiHandFrame | None:
        snapshot_fn = getattr(self._wrapped_source, "latest_bihand_frame_snapshot", None)
        if callable(snapshot_fn):
            snapshot = snapshot_fn()
            if snapshot is not None:
                frame_index, frame = snapshot
                if frame_index > self._last_raw_frame_index:
                    self._last_raw_frame_index = frame_index
                    return copy_bihand_frame(frame)
                if self._last_output is not None:
                    return copy_bihand_frame(self._last_output)

        frame = self._wrapped_source.get_frame()
        if frame.detection is None:
            return None
        return copy_bihand_frame(frame.detection)

