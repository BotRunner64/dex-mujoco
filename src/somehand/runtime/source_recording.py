"""Recording wrappers and replay data sources."""

from __future__ import annotations

from threading import Lock

from somehand.core import BiHandFrame, BiHandSourceFrame, HandFrame, SourceFrame
from somehand.infrastructure.artifacts import load_bihand_recording_artifact, load_hand_recording_artifact

from .source_transforms import copy_bihand_frame, copy_hand_frame


class RecordingHandTrackingSource:
    def __init__(self, wrapped_source: object, *, recording_enabled: bool = True):
        self._wrapped_source = wrapped_source
        self.recorded_frames: list[HandFrame] = []
        self._recording_lock = Lock()
        self._recording_enabled = recording_enabled

    def __getattr__(self, name: str):
        return getattr(self._wrapped_source, name)

    @property
    def source_desc(self) -> str:
        return str(getattr(self._wrapped_source, "source_desc"))

    @property
    def fps(self) -> int:
        return int(getattr(self._wrapped_source, "fps"))

    def is_available(self) -> bool:
        return bool(self._wrapped_source.is_available())

    def get_frame(self) -> SourceFrame:
        frame = self._wrapped_source.get_frame()
        if frame.detection is not None and self.is_recording:
            self.recorded_frames.append(copy_hand_frame(frame.detection))
        return frame

    def latest_hand_frame_snapshot(self) -> tuple[int, HandFrame] | None:
        snapshot_fn = getattr(self._wrapped_source, "latest_hand_frame_snapshot", None)
        if not callable(snapshot_fn):
            return None
        return snapshot_fn()

    def reset(self) -> bool:
        return bool(self._wrapped_source.reset())

    def close(self) -> None:
        self._wrapped_source.close()

    def stats_snapshot(self) -> dict[str, object]:
        return dict(self._wrapped_source.stats_snapshot())

    @property
    def is_recording(self) -> bool:
        with self._recording_lock:
            return self._recording_enabled

    def start_recording(self) -> None:
        with self._recording_lock:
            self._recording_enabled = True

    def stop_recording(self) -> None:
        with self._recording_lock:
            self._recording_enabled = False


class RecordingBiHandTrackingSource:
    def __init__(self, wrapped_source: object, *, recording_enabled: bool = True):
        self._wrapped_source = wrapped_source
        self.recorded_frames: list[BiHandFrame] = []
        self._recording_lock = Lock()
        self._recording_enabled = recording_enabled

    def __getattr__(self, name: str):
        return getattr(self._wrapped_source, name)

    @property
    def source_desc(self) -> str:
        return str(getattr(self._wrapped_source, "source_desc"))

    @property
    def fps(self) -> int:
        return int(getattr(self._wrapped_source, "fps"))

    def is_available(self) -> bool:
        return bool(self._wrapped_source.is_available())

    def get_frame(self) -> BiHandSourceFrame:
        frame = self._wrapped_source.get_frame()
        if frame.detection is not None and frame.detection.has_detection and self.is_recording:
            self.recorded_frames.append(copy_bihand_frame(frame.detection))
        return frame

    def latest_bihand_frame_snapshot(self) -> tuple[int, BiHandFrame] | None:
        snapshot_fn = getattr(self._wrapped_source, "latest_bihand_frame_snapshot", None)
        if not callable(snapshot_fn):
            return None
        return snapshot_fn()

    def reset(self) -> bool:
        return bool(self._wrapped_source.reset())

    def close(self) -> None:
        self._wrapped_source.close()

    def stats_snapshot(self) -> dict[str, object]:
        return dict(self._wrapped_source.stats_snapshot())

    @property
    def is_recording(self) -> bool:
        with self._recording_lock:
            return self._recording_enabled

    def start_recording(self) -> None:
        with self._recording_lock:
            self._recording_enabled = True

    def stop_recording(self) -> None:
        with self._recording_lock:
            self._recording_enabled = False


class RecordedHandDataSource:
    def __init__(self, recording_path: str):
        recording = load_hand_recording_artifact(recording_path)
        self._frames: list[HandFrame] = list(recording["frames"])
        self._fps = int(recording["fps"])
        self._index = 0
        self.recording_path = recording_path
        self.source_desc = recording_path
        self.recording_metadata = {
            "input_source": recording["input_source"],
            "input_type": recording["input_type"],
            "hand_side": recording.get("hand_side"),
            "num_frames": recording["num_frames"],
            "num_detected": recording["num_detected"],
        }

    @property
    def fps(self) -> int:
        return self._fps

    def is_available(self) -> bool:
        return self._index < len(self._frames)

    def get_frame(self) -> SourceFrame:
        if not self.is_available():
            raise StopIteration
        frame = self._frames[self._index]
        self._index += 1
        return SourceFrame(detection=copy_hand_frame(frame))

    def reset(self) -> bool:
        if not self._frames:
            return False
        self._index = 0
        return True

    def close(self) -> None:
        return None

    def stats_snapshot(self) -> dict[str, object]:
        return {}


class RecordedBiHandDataSource:
    def __init__(self, recording_path: str):
        recording = load_bihand_recording_artifact(recording_path)
        self._frames: list[BiHandFrame] = list(recording["frames"])
        self._fps = int(recording["fps"])
        self._index = 0
        self.recording_path = recording_path
        self.source_desc = recording_path
        self.recording_metadata = {
            "input_source": recording["input_source"],
            "input_type": recording["input_type"],
            "num_frames": recording["num_frames"],
            "num_detected": recording["num_detected"],
        }

    @property
    def fps(self) -> int:
        return self._fps

    def is_available(self) -> bool:
        return self._index < len(self._frames)

    def get_frame(self) -> BiHandSourceFrame:
        if not self.is_available():
            raise StopIteration
        frame = self._frames[self._index]
        self._index += 1
        return BiHandSourceFrame(detection=copy_bihand_frame(frame))

    def reset(self) -> bool:
        if not self._frames:
            return False
        self._index = 0
        return True

    def close(self) -> None:
        return None

    def stats_snapshot(self) -> dict[str, object]:
        return {}


def create_recording_source(*, recording_path: str) -> RecordedHandDataSource:
    return RecordedHandDataSource(recording_path)


def create_bihand_recording_source(*, recording_path: str) -> RecordedBiHandDataSource:
    return RecordedBiHandDataSource(recording_path)
