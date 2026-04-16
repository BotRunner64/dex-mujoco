"""Compatibility re-exports for runtime output sinks."""

from somehand.runtime.sink_outputs import (
    AsyncBiHandLandmarkOutputSink,
    AsyncLandmarkOutputSink,
    BiHandOutputWindowSink,
    BiHandVideoOutputSink,
    RobotHandOutputSink,
    RobotHandTargetOutputSink,
    RobotHandVideoOutputSink,
    TrajectoryRecorder,
)
from somehand.runtime.sink_rendering import fit_video_size as _fit_video_size

__all__ = [
    "AsyncBiHandLandmarkOutputSink",
    "AsyncLandmarkOutputSink",
    "BiHandOutputWindowSink",
    "BiHandVideoOutputSink",
    "RobotHandOutputSink",
    "RobotHandTargetOutputSink",
    "RobotHandVideoOutputSink",
    "TrajectoryRecorder",
    "_fit_video_size",
]
