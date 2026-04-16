"""Infrastructure adapters for external systems."""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULES = {
    "AsyncBiHandLandmarkOutputSink": ".sinks",
    "AsyncLandmarkOutputSink": ".sinks",
    "BiHCMocapInputSource": ".sources",
    "BiHandMediaPipeInputSource": ".sources",
    "BiHandOutputWindowSink": ".sinks",
    "BiHandPicoInputSource": ".sources",
    "BiHandVideoOutputSink": ".sinks",
    "HCMocapInputSource": ".sources",
    "HandModel": ".hand_model",
    "LinkerHandModelAdapter": ".controllers",
    "LinkerHandSdkController": ".controllers",
    "MediaPipeInputSource": ".sources",
    "ModelNameResolver": ".model_name_resolver",
    "MujocoSimController": ".controllers",
    "OpenCvPreviewWindow": ".preview",
    "RecordedBiHandDataSource": ".sources",
    "RecordedHandDataSource": ".sources",
    "RecordingBiHandTrackingSource": ".sources",
    "RecordingHandTrackingSource": ".sources",
    "RobotHandOutputSink": ".sinks",
    "RobotHandTargetOutputSink": ".sinks",
    "RobotHandVideoOutputSink": ".sinks",
    "TerminalRecordingController": ".terminal_controls",
    "TrajectoryRecorder": ".sinks",
    "VectorRetargeter": ".vector_solver",
    "create_bihand_hc_mocap_udp_source": ".sources",
    "create_bihand_pico_source": ".sources",
    "create_bihand_recording_source": ".sources",
    "create_hc_mocap_udp_source": ".sources",
    "create_pico_source": ".sources",
    "create_recording_source": ".sources",
    "infer_linkerhand_model_family": ".controllers",
    "load_bihand_config": ".config_loader",
    "load_bihand_recording_artifact": ".artifacts",
    "load_hand_recording_artifact": ".artifacts",
    "load_retargeting_config": ".config_loader",
    "save_bihand_recording_artifact": ".artifacts",
    "save_hand_recording_artifact": ".artifacts",
    "save_trajectory_artifact": ".artifacts",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
