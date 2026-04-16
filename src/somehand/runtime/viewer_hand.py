"""Hand and bi-hand MuJoCo viewer implementations."""

from __future__ import annotations

import mujoco
import numpy as np

from somehand.infrastructure.hand_model import HandModel

from .viewer_camera import DEFAULT_BIHAND_CAMERA, DEFAULT_HAND_CAMERA, configure_free_camera, try_frame_hand_camera
from .viewer_passive import ManagedPassiveViewer, compile_model_with_name, mujoco_key_callback, set_viewer_overlay_label, set_viewer_window_title


class HandVisualizer:
    """Real-time MuJoCo visualization of the retargeted robot hand."""

    def __init__(
        self,
        hand_model: HandModel,
        *,
        key_callback=None,
        overlay_label: str | None = None,
        window_title: str | None = None,
    ):
        self.hand_model = hand_model
        if window_title:
            self.model, self.data = compile_model_with_name(hand_model.mjcf_path, window_title)
        else:
            self.model = hand_model.model
            self.data = hand_model.data
        self._overlay_label = overlay_label
        self.viewer = ManagedPassiveViewer(
            model=self.model,
            data=self.data,
            key_callback=mujoco_key_callback(key_callback),
            show_left_ui=False,
            show_right_ui=False,
            window_title=window_title,
        )
        set_viewer_window_title(self.viewer, window_title)
        set_viewer_overlay_label(self.viewer, self._overlay_label)
        self._configure_camera(**DEFAULT_HAND_CAMERA)
        self._camera_initialized = False

    def _configure_camera(
        self,
        *,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: tuple[float, float, float],
    ) -> None:
        with self.viewer.lock():
            configure_free_camera(
                self.viewer.cam,
                distance=distance,
                azimuth=azimuth,
                elevation=elevation,
                lookat=lookat,
            )
        self.viewer.sync(state_only=True)

    def update(self, qpos: np.ndarray):
        with self.viewer.lock():
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            if not self._camera_initialized and try_frame_hand_camera(self.viewer.cam, model=self.model, data=self.data):
                self._camera_initialized = True
        set_viewer_overlay_label(self.viewer, self._overlay_label)
        self.viewer.sync()

    @property
    def is_running(self) -> bool:
        return self.viewer.is_running()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()


class BiHandScene:
    """Combined MuJoCo scene containing left and right hand models."""

    def __init__(
        self,
        left_hand_model: HandModel,
        right_hand_model: HandModel,
        *,
        left_pos: tuple[float, float, float] = (0.22, 0.04, 0.02),
        right_pos: tuple[float, float, float] = (-0.22, 0.04, 0.02),
        left_quat: tuple[float, float, float, float] = (0.69288325, 0.01522078, -0.05862347, 0.71850151),
        right_quat: tuple[float, float, float, float] = (0.71846417, 0.05829359, -0.01490552, 0.69295665),
    ):
        self.left_hand_model = left_hand_model
        self.right_hand_model = right_hand_model
        self.left_pos = tuple(float(value) for value in left_pos)
        self.right_pos = tuple(float(value) for value in right_pos)
        self.left_quat = tuple(float(value) for value in left_quat)
        self.right_quat = tuple(float(value) for value in right_quat)
        self.model, self.data = self._build_model()
        self.left_qpos_indices = self._resolve_qpos_indices(left_hand_model, prefix="left_")
        self.right_qpos_indices = self._resolve_qpos_indices(right_hand_model, prefix="right_")

    def _build_model(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        spec = mujoco.MjSpec()
        spec.modelname = "somehand_bihand"
        spec.visual.global_.offwidth = max(
            int(self.left_hand_model.model.vis.global_.offwidth),
            int(self.right_hand_model.model.vis.global_.offwidth),
        )
        spec.visual.global_.offheight = max(
            int(self.left_hand_model.model.vis.global_.offheight),
            int(self.right_hand_model.model.vis.global_.offheight),
        )

        left_frame = spec.worldbody.add_frame()
        left_frame.pos = list(self.left_pos)
        left_frame.quat = list(self.left_quat)
        right_frame = spec.worldbody.add_frame()
        right_frame.pos = list(self.right_pos)
        right_frame.quat = list(self.right_quat)

        spec.attach(
            mujoco.MjSpec.from_file(self.left_hand_model.mjcf_path),
            frame=left_frame,
            prefix="left_",
        )
        spec.attach(
            mujoco.MjSpec.from_file(self.right_hand_model.mjcf_path),
            frame=right_frame,
            prefix="right_",
        )

        model = spec.compile()
        data = mujoco.MjData(model)
        return model, data

    def _resolve_qpos_indices(self, hand_model: HandModel, *, prefix: str) -> np.ndarray:
        qpos_indices: list[int] = []
        for joint_name in hand_model.get_joint_names():
            source_joint_id = mujoco.mj_name2id(hand_model.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            joint_type = int(hand_model.model.jnt_type[source_joint_id])
            width = 7 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else 4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            combined_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{joint_name}")
            combined_qpos_adr = int(self.model.jnt_qposadr[combined_joint_id])
            qpos_indices.extend(range(combined_qpos_adr, combined_qpos_adr + width))
        return np.array(qpos_indices, dtype=np.int32)

    def update(self, left_qpos: np.ndarray, right_qpos: np.ndarray) -> None:
        self.data.qpos[self.left_qpos_indices] = left_qpos
        self.data.qpos[self.right_qpos_indices] = right_qpos
        mujoco.mj_forward(self.model, self.data)


class BiHandVisualizer:
    """Real-time MuJoCo visualization of both retargeted robot hands."""

    def __init__(
        self,
        left_hand_model: HandModel,
        right_hand_model: HandModel,
        *,
        key_callback=None,
        left_pos: tuple[float, float, float] = (0.22, 0.04, 0.02),
        right_pos: tuple[float, float, float] = (-0.22, 0.04, 0.02),
        camera_lookat: tuple[float, float, float] = (0.0, 0.04, 0.02),
        left_quat: tuple[float, float, float, float] = (0.69288325, 0.01522078, -0.05862347, 0.71850151),
        right_quat: tuple[float, float, float, float] = (0.71846417, 0.05829359, -0.01490552, 0.69295665),
    ):
        self.scene = BiHandScene(
            left_hand_model,
            right_hand_model,
            left_pos=left_pos,
            right_pos=right_pos,
            left_quat=left_quat,
            right_quat=right_quat,
        )
        self.model = self.scene.model
        self.data = self.scene.data
        self._camera_lookat = tuple(float(value) for value in camera_lookat)
        self.viewer = ManagedPassiveViewer(
            model=self.model,
            data=self.data,
            key_callback=mujoco_key_callback(key_callback),
            show_left_ui=False,
            show_right_ui=False,
        )
        self._configure_camera(
            distance=DEFAULT_BIHAND_CAMERA["distance"],
            azimuth=DEFAULT_BIHAND_CAMERA["azimuth"],
            elevation=DEFAULT_BIHAND_CAMERA["elevation"],
            lookat=self._camera_lookat,
        )
        self._camera_initialized = False

    def _configure_camera(
        self,
        *,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: tuple[float, float, float],
    ) -> None:
        with self.viewer.lock():
            configure_free_camera(
                self.viewer.cam,
                distance=distance,
                azimuth=azimuth,
                elevation=elevation,
                lookat=lookat,
            )
        self.viewer.sync(state_only=True)

    def update(self, left_qpos: np.ndarray, right_qpos: np.ndarray) -> None:
        with self.viewer.lock():
            self.scene.update(left_qpos, right_qpos)
            if not self._camera_initialized and try_frame_hand_camera(
                self.viewer.cam,
                model=self.model,
                data=self.data,
                azimuth=DEFAULT_BIHAND_CAMERA["azimuth"],
                elevation=DEFAULT_BIHAND_CAMERA["elevation"],
            ):
                self._camera_initialized = True
        self.viewer.sync()

    @property
    def is_running(self) -> bool:
        return self.viewer.is_running()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()


__all__ = ["HandVisualizer", "BiHandScene", "BiHandVisualizer"]
