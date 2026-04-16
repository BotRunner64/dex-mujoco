"""Compatibility re-exports for runtime source adapters."""

from somehand.runtime.source_adapters import (
    BiHCMocapInputSource,
    BiHandMediaPipeInputSource,
    BiHandPicoInputSource,
    HCMocapInputSource,
    MediaPipeInputSource,
    create_bihand_hc_mocap_udp_source,
    create_bihand_pico_source,
    create_hc_mocap_udp_source,
    create_pico_source,
)
from somehand.runtime.source_recording import (
    RecordedBiHandDataSource,
    RecordedHandDataSource,
    RecordingBiHandTrackingSource,
    RecordingHandTrackingSource,
    create_bihand_recording_source,
    create_recording_source,
)

__all__ = [
    "BiHCMocapInputSource",
    "BiHandMediaPipeInputSource",
    "BiHandPicoInputSource",
    "HCMocapInputSource",
    "MediaPipeInputSource",
    "RecordedBiHandDataSource",
    "RecordedHandDataSource",
    "RecordingBiHandTrackingSource",
    "RecordingHandTrackingSource",
    "create_bihand_hc_mocap_udp_source",
    "create_bihand_pico_source",
    "create_bihand_recording_source",
    "create_hc_mocap_udp_source",
    "create_pico_source",
    "create_recording_source",
]
