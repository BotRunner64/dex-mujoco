#!/usr/bin/env python3
"""Render recorded hand landmarks with the same MuJoCo view as the online viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from dex_mujoco.domain import preprocess_landmarks
from dex_mujoco.infrastructure.artifacts import load_bihand_recording_artifact, load_hand_recording_artifact
from dex_mujoco.infrastructure.sinks import _create_offscreen_renderer, _fit_video_size, _transform_points
from dex_mujoco.visualization import (
    _DEFAULT_BIHAND_LANDMARK_CAMERA,
    _DEFAULT_LANDMARK_CAMERA,
    _LANDMARK_VIEWER_XML,
    _append_bihand_landmark_geoms,
    _append_single_landmark_geoms,
    _try_frame_camera_to_points,
    configure_free_camera,
)

_LEFT_POS = (0.22, 0.04, 0.02)
_RIGHT_POS = (-0.22, 0.04, 0.02)
_LEFT_QUAT = (0.69288325, 0.01522078, -0.05862347, 0.71850151)
_RIGHT_QUAT = (0.71846417, 0.05829359, -0.01490552, 0.69295665)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a recorded landmarks-only MP4 with online-viewer framing")
    parser.add_argument("--recording", required=True, help="Input recording artifact (.pkl)")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--mode", choices=["auto", "single", "bihand"], default="auto", help="Recording mode")
    parser.add_argument("--width", type=int, default=960, help="Requested output width")
    parser.add_argument("--height", type=int, default=720, help="Requested output height")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for quick previews")
    return parser.parse_args()


def _load_recording(recording_path: str, mode: str) -> tuple[str, list[object], int, str]:
    if mode in {"auto", "single"}:
        try:
            recording = load_hand_recording_artifact(recording_path)
            return "single", list(recording["frames"]), int(recording["fps"]), str(recording["input_source"])
        except ValueError:
            if mode == "single":
                raise
    recording = load_bihand_recording_artifact(recording_path)
    return "bihand", list(recording["frames"]), int(recording["fps"]), str(recording["input_source"])


def _prepare_single_frame(frame) -> np.ndarray:
    return preprocess_landmarks(frame.landmarks_3d, hand_side=frame.hand_side)


def _prepare_bihand_frame(frame) -> np.ndarray:
    left = np.full((21, 3), np.nan, dtype=np.float64)
    right = np.full((21, 3), np.nan, dtype=np.float64)
    if frame.left is not None:
        left = preprocess_landmarks(frame.left.landmarks_3d, hand_side=frame.left.hand_side)
        left = _transform_points(left, pos=_LEFT_POS, quat=_LEFT_QUAT)
    if frame.right is not None:
        right = preprocess_landmarks(frame.right.landmarks_3d, hand_side=frame.right.hand_side)
        right = _transform_points(right, pos=_RIGHT_POS, quat=_RIGHT_QUAT)
    return np.stack([left, right], axis=0)


def _render_frames(mode: str, raw_frames: list[object]) -> list[np.ndarray]:
    if mode == "single":
        return [_prepare_single_frame(frame) for frame in raw_frames]
    return [_prepare_bihand_frame(frame) for frame in raw_frames]


def _camera_defaults(mode: str) -> dict[str, object]:
    return _DEFAULT_LANDMARK_CAMERA if mode == "single" else _DEFAULT_BIHAND_LANDMARK_CAMERA


def _initialize_camera(camera, *, mode: str) -> None:
    configure_free_camera(camera, **_camera_defaults(mode))


def _try_frame_camera(camera, *, model, landmarks: np.ndarray, mode: str, aspect_ratio: float) -> bool:
    defaults = _camera_defaults(mode)
    if mode == "single":
        return _try_frame_camera_to_points(
            camera,
            model=model,
            points=landmarks,
            azimuth=float(defaults["azimuth"]),
            elevation=float(defaults["elevation"]),
            aspect_ratio=aspect_ratio,
        )

    finite_points = landmarks[np.isfinite(landmarks).all(axis=2)]
    if finite_points.size == 0:
        return False
    return _try_frame_camera_to_points(
        camera,
        model=model,
        points=finite_points.reshape(-1, 3),
        azimuth=float(defaults["azimuth"]),
        elevation=float(defaults["elevation"]),
        aspect_ratio=aspect_ratio,
    )


def _append_overlay_geoms(scene, *, mode: str, landmarks: np.ndarray) -> None:
    if mode == "single":
        _append_single_landmark_geoms(scene, landmarks)
        return
    _append_bihand_landmark_geoms(scene, landmarks)


def main() -> None:
    args = _parse_args()
    mode, raw_frames, fps, input_source = _load_recording(args.recording, args.mode)
    if args.max_frames > 0:
        raw_frames = raw_frames[: args.max_frames]

    prepared_frames = _render_frames(mode, raw_frames)
    if not prepared_frames:
        raise RuntimeError("Recording contains no frames to render")

    model = mujoco.MjModel.from_xml_string(_LANDMARK_VIEWER_XML)
    data = mujoco.MjData(model)
    width, height = _fit_video_size(
        requested_width=args.width,
        requested_height=args.height,
        max_width=max(int(model.vis.global_.offwidth), 1),
        max_height=max(int(model.vis.global_.offheight), 1),
    )
    renderer = _create_offscreen_renderer(model, width=width, height=height)
    frame_aspect_ratio = width / max(height, 1)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    _initialize_camera(camera, mode=mode)
    camera_initialized = False

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*args.codec),
        float(max(fps, 1)),
        (width, height),
    )
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"Cannot open video writer for: {output_path}")

    try:
        for landmarks in prepared_frames:
            mujoco.mj_forward(model, data)
            if not camera_initialized and _try_frame_camera(
                camera,
                model=model,
                landmarks=landmarks,
                mode=mode,
                aspect_ratio=frame_aspect_ratio,
            ):
                camera_initialized = True
            renderer.update_scene(data, camera=camera)
            _append_overlay_geoms(renderer.scene, mode=mode, landmarks=landmarks)
            frame_rgb = renderer.render()
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        renderer.close()

    print(
        f"Saved landmark video ({len(prepared_frames)} frames, {width}x{height}, source={input_source}) "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
