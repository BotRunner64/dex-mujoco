"""Core vector retargeting algorithm.

Uses weighted cosine similarity loss (frame-invariant direction matching),
analytical MuJoCo Jacobians, and scipy SLSQP optimizer.
"""

import numpy as np
import mujoco
from scipy.optimize import minimize

from .hand_model import HandModel
from .retargeting_config import RetargetingConfig

# MediaPipe landmark indices for pinch detection
_THUMB_TIP_IDX = 4
_FINGER_TIP_INDICES = [8, 12, 16, 20]  # index, middle, ring, pinky
# Thumb-related landmark indices (used to identify thumb vectors)
_THUMB_LANDMARKS = {0, 1, 2, 3, 4}

_OPERATOR2ROBOT_RIGHT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _mediapipe_to_mujoco(landmarks_3d: np.ndarray) -> np.ndarray:
    """Transform MediaPipe landmarks to robot-aligned frame."""
    out = np.empty_like(landmarks_3d)
    out[:, 0] = -landmarks_3d[:, 2]
    out[:, 1] = landmarks_3d[:, 0]
    out[:, 2] = -landmarks_3d[:, 1]
    return out


def _mirror_left_to_right(landmarks_3d: np.ndarray) -> np.ndarray:
    mirrored = landmarks_3d.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    return mirrored


def _estimate_wrist_frame(landmarks_3d: np.ndarray) -> np.ndarray:
    palm = landmarks_3d[[0, 5, 9, 13, 17], :]
    x_vector = landmarks_3d[0] - landmarks_3d[9]
    palm_centered = palm - palm.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(palm_centered, full_matrices=False)
    normal = vh[-1]
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        raise ValueError("Cannot estimate palm normal from degenerate landmarks")
    normal = normal / normal_norm

    x_axis = x_vector - np.dot(x_vector, normal) * normal
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("Cannot estimate palm x-axis from degenerate landmarks")
    x_axis = x_axis / x_norm

    z_axis = np.cross(x_axis, normal)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        raise ValueError("Cannot estimate palm z-axis from degenerate landmarks")
    z_axis = z_axis / z_norm

    if np.dot(z_axis, landmarks_3d[17] - landmarks_3d[5]) < 0.0:
        normal *= -1.0
        z_axis *= -1.0

    return np.stack([x_axis, normal, z_axis], axis=1)


def preprocess_landmarks(
    landmarks_3d: np.ndarray,
    handedness: str = "Right",
    frame: str = "wrist_local",
) -> np.ndarray:
    if handedness == "Left":
        landmarks_3d = _mirror_left_to_right(landmarks_3d)

    centered = landmarks_3d - landmarks_3d[0:1, :]
    if frame == "camera_aligned":
        return _mediapipe_to_mujoco(centered)
    if frame != "wrist_local":
        raise ValueError(f"Unsupported preprocess frame: {frame}")

    try:
        wrist_frame = _estimate_wrist_frame(centered)
        return centered @ wrist_frame @ _OPERATOR2ROBOT_RIGHT
    except ValueError:
        return _mediapipe_to_mujoco(centered)


def compute_target_directions(
    landmarks_3d: np.ndarray,
    human_vector_pairs: list[tuple[int, int]],
    handedness: str = "Right",
    frame: str = "wrist_local",
) -> np.ndarray:
    landmarks = preprocess_landmarks(landmarks_3d, handedness=handedness, frame=frame)
    directions = np.empty((len(human_vector_pairs), 3), dtype=np.float64)
    for i, (origin_idx, target_idx) in enumerate(human_vector_pairs):
        vector = landmarks[target_idx] - landmarks[origin_idx]
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            directions[i] = 0.0
        else:
            directions[i] = vector / norm
    return directions


class TemporalFilter:
    """Exponential moving average filter for smooth landmark tracking."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._prev: np.ndarray | None = None

    def filter(self, value: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = value.copy()
            return value
        self._prev = self.alpha * value + (1 - self.alpha) * self._prev
        return self._prev.copy()

    def reset(self):
        self._prev = None


class VectorRetargeter:
    """Optimizes robot joint angles to match human finger vector directions."""

    def __init__(self, hand_model: HandModel, config: RetargetingConfig):
        self.hand_model = hand_model
        self.config = config
        self.model = hand_model.model
        self.data = hand_model.data

        self.human_vector_pairs = [
            (pair[0], pair[1]) for pair in config.human_vector_pairs
        ]
        self.origin_link_names = config.origin_link_names
        self.task_link_names = config.task_link_names

        self.origin_ids = []
        self.origin_is_site = []
        for i, name in enumerate(self.origin_link_names):
            is_site = config.origin_link_types[i] == "site"
            self.origin_is_site.append(is_site)
            obj_type = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
            link_id = mujoco.mj_name2id(self.model, obj_type, name)
            assert link_id >= 0, f"Origin link '{name}' not found in model"
            self.origin_ids.append(link_id)

        self.task_ids = []
        self.task_is_site = []
        for i, name in enumerate(self.task_link_names):
            is_site = config.task_link_types[i] == "site"
            self.task_is_site.append(is_site)
            obj_type = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
            link_id = mujoco.mj_name2id(self.model, obj_type, name)
            assert link_id >= 0, f"Task link '{name}' not found in model"
            self.task_ids.append(link_id)

        self.landmark_filter = TemporalFilter(
            alpha=config.preprocess.temporal_filter_alpha
        )

        self._norm_delta = config.solver.norm_delta
        self._max_iterations = config.solver.max_iterations
        self._output_alpha = config.solver.output_alpha
        self._weights = np.array(config.vector_weights, dtype=np.float64)
        self._preprocess_frame = config.preprocess.frame

        self._target_directions: np.ndarray | None = None
        self._target_angles: np.ndarray | None = None
        self._last_qpos: np.ndarray | None = None

        # Pinch-aware dynamic weighting
        self._pinch_enabled = config.pinch.enabled
        self._pinch_alphas = np.zeros(4)      # per-finger pinch alpha
        self._pinch_alpha_thumb = 0.0         # max across fingers
        self._pinch_d1 = config.pinch.d1
        self._pinch_d2 = config.pinch.d2
        self._pinch_weight = config.pinch.weight
        self._thumb_weight_boost = config.pinch.thumb_weight_boost
        self._thumb_vector_indices: set[int] = set()
        self._thumb_site_id = -1
        self._finger_site_ids: list[int] = []

        if self._pinch_enabled:
            # Identify which vector indices involve thumb landmarks
            for i, (o, t) in enumerate(self.human_vector_pairs):
                if o in _THUMB_LANDMARKS or t in _THUMB_LANDMARKS:
                    self._thumb_vector_indices.add(i)

            # Look up fingertip site IDs
            sites = config.pinch.fingertip_sites
            if len(sites) >= 5:
                self._thumb_site_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, sites[0]
                )
                for name in sites[1:5]:
                    sid = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_SITE, name
                    )
                    self._finger_site_ids.append(sid)
                assert self._thumb_site_id >= 0, (
                    f"Thumb site '{sites[0]}' not found in model"
                )
                for i, sid in enumerate(self._finger_site_ids):
                    assert sid >= 0, (
                        f"Finger site '{sites[i+1]}' not found in model"
                    )

        # Position constraints: wrist-relative position matching for thumb joints
        self._pos_enabled = config.position.enabled
        self._pos_weight = config.position.weight
        self._pos_landmark_indices: list[int] = []
        self._pos_body_ids: list[int] = []
        self._pos_body_is_site: list[bool] = []
        self._pos_per_weights: list[float] = []
        self._position_targets: np.ndarray | None = None
        self._robot_palm_size: float = 0.0
        self._scale_landmark_idx: int = 0  # index into landmarks for scale ref
        self._wrist_body_id: int = 0

        if self._pos_enabled:
            pos_cfg = config.position
            # Scale reference: compute robot palm size from two bodies
            self._scale_landmark_idx = pos_cfg.scale_landmarks[1]
            scale_ids = []
            for i, name in enumerate(pos_cfg.scale_bodies):
                is_site = pos_cfg.scale_body_types[i] == "site"
                otype = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
                bid = mujoco.mj_name2id(self.model, otype, name)
                assert bid >= 0, f"Scale body '{name}' not found"
                scale_ids.append((bid, is_site))

            # Wrist body ID (first scale body, typically "world")
            self._wrist_body_id = scale_ids[0][0]
            self._wrist_is_site = scale_ids[0][1]

            # Precompute robot palm size at init (palm bones don't move)
            self._forward()
            p0 = self._get_pos(scale_ids[0][0], scale_ids[0][1])
            p1 = self._get_pos(scale_ids[1][0], scale_ids[1][1])
            self._robot_palm_size = float(np.linalg.norm(p1 - p0))

            # Set up per-constraint data
            for pc in pos_cfg.constraints:
                self._pos_landmark_indices.append(pc.landmark)
                is_site = pc.body_type == "site"
                self._pos_body_is_site.append(is_site)
                otype = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
                bid = mujoco.mj_name2id(self.model, otype, pc.body)
                assert bid >= 0, f"Position constraint body '{pc.body}' not found"
                self._pos_body_ids.append(bid)
                self._pos_per_weights.append(pc.weight)

        # Angle constraints: map human joint flexion to robot joint angle
        self._angle_landmarks = []  # list of (a, b, c) landmark triples
        self._angle_qpos_ids = []  # qpos index for each constrained joint
        self._angle_dof_ids = []   # velocity DOF index for gradient
        self._angle_joint_ranges = []  # (lo, hi) for each constrained joint
        self._angle_weights = []
        for ac in config.angle_constraints:
            self._angle_landmarks.append(tuple(ac.landmarks))
            jname = ac.joint
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            assert jid >= 0, f"Angle constraint joint '{jname}' not found in model"
            qadr = self.model.jnt_qposadr[jid]
            dadr = self.model.jnt_dofadr[jid]
            self._angle_qpos_ids.append(int(qadr))
            self._angle_dof_ids.append(int(dadr))
            lo, hi = self.model.jnt_range[jid]
            self._angle_joint_ranges.append((float(lo), float(hi)))
            self._angle_weights.append(ac.weight)

        nq = self.model.nq
        self._bounds = []
        for j in range(nq):
            lo, hi = self.model.jnt_range[j]
            if lo < hi:
                self._bounds.append((float(lo), float(hi)))
            else:
                self._bounds.append((None, None))

        # Initialize only the first thumb joint (cmc_yaw/cmc_roll) to mid-range
        # so the thumb starts pointing inward instead of sideways at qpos=0.
        # Only the base rotation joint needs this; leave upper joints at 0
        # so the solver is free to drive them from the direction constraints.
        for j in range(nq):
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jname and "thumb" in jname and "cmc" in jname:
                lo, hi = self.model.jnt_range[j]
                if lo < hi:
                    self.data.qpos[j] = (lo + hi) / 2.0
                break  # only the first cmc joint
        self._forward()

    def _forward(self, qpos: np.ndarray | None = None):
        if qpos is not None:
            self.data.qpos[:] = qpos
        mujoco.mj_fwdPosition(self.model, self.data)

    def _get_pos(self, idx: int, is_site: bool) -> np.ndarray:
        if is_site:
            return self.data.site_xpos[idx].copy()
        return self.data.xpos[idx].copy()

    def _get_robot_vectors(self) -> np.ndarray:
        vectors = np.empty((len(self.origin_ids), 3))
        for i in range(len(self.origin_ids)):
            p_origin = self._get_pos(self.origin_ids[i], self.origin_is_site[i])
            p_task = self._get_pos(self.task_ids[i], self.task_is_site[i])
            vectors[i] = p_task - p_origin
        return vectors

    def _get_effective_weight(self, i: int) -> float:
        """Get vector weight, boosted for thumb vectors during pinch."""
        w = self._weights[i]
        if (self._pinch_enabled and self._pinch_alpha_thumb > 0
                and i in self._thumb_vector_indices):
            w *= (1.0 + self._pinch_alpha_thumb * self._thumb_weight_boost)
        return w

    def _compute_pinch_loss(self) -> float:
        """Compute fingertip attraction loss for pinch."""
        if not self._pinch_enabled or self._pinch_alpha_thumb < 1e-6:
            return 0.0
        thumb_pos = self.data.site_xpos[self._thumb_site_id]
        loss = 0.0
        for i in range(4):
            alpha = self._pinch_alphas[i]
            if alpha < 1e-6:
                continue
            finger_pos = self.data.site_xpos[self._finger_site_ids[i]]
            diff = thumb_pos - finger_pos
            loss += alpha * self._pinch_weight * np.dot(diff, diff)
        return loss

    def _compute_position_loss(self) -> float:
        """Compute wrist-relative position matching loss for thumb joints."""
        if not self._pos_enabled or self._position_targets is None:
            return 0.0
        wrist_pos = self._get_pos(self._wrist_body_id, self._wrist_is_site)
        loss = 0.0
        for k in range(len(self._pos_body_ids)):
            body_pos = self._get_pos(
                self._pos_body_ids[k], self._pos_body_is_site[k]
            )
            robot_rel = body_pos - wrist_pos
            diff = robot_rel - self._position_targets[k]
            w = self._pos_weight * self._pos_per_weights[k]
            loss += w * np.dot(diff, diff)
        return loss

    def _compute_loss(self, qpos: np.ndarray) -> float:
        self._forward(qpos)
        robot_vecs = self._get_robot_vectors()
        loss = 0.0
        for i in range(len(robot_vecs)):
            r_norm = np.linalg.norm(robot_vecs[i])
            w = self._get_effective_weight(i)
            if r_norm < 1e-8:
                loss += w
                continue
            cos_sim = np.dot(robot_vecs[i] / r_norm, self._target_directions[i])
            loss += w * (1.0 - cos_sim)
        if self._last_qpos is not None:
            loss += self._norm_delta * np.sum((qpos - self._last_qpos) ** 2)
        if self._target_angles is not None:
            for k in range(len(self._angle_qpos_ids)):
                qadr = self._angle_qpos_ids[k]
                diff = qpos[qadr] - self._target_angles[k]
                loss += self._angle_weights[k] * diff * diff
        loss += self._compute_pinch_loss()
        loss += self._compute_position_loss()
        return loss

    def _compute_loss_and_grad(self, qpos: np.ndarray) -> tuple[float, np.ndarray]:
        self._forward(qpos)
        robot_vecs = self._get_robot_vectors()
        nv = self.model.nv
        grad = np.zeros(nv)
        loss = 0.0

        for i in range(len(self.origin_ids)):
            r_vec = robot_vecs[i]
            r_norm = np.linalg.norm(r_vec)
            w = self._get_effective_weight(i)
            if r_norm < 1e-8:
                loss += w
                continue

            r_dir = r_vec / r_norm
            t_dir = self._target_directions[i]
            cos_sim = np.dot(r_dir, t_dir)
            loss += w * (1.0 - cos_sim)

            grad_vec = -(t_dir - cos_sim * r_dir) / r_norm
            jac_task = np.zeros((3, nv))
            jac_origin = np.zeros((3, nv))

            if self.task_is_site[i]:
                mujoco.mj_jacSite(self.model, self.data, jac_task, None, self.task_ids[i])
            else:
                mujoco.mj_jacBody(self.model, self.data, jac_task, None, self.task_ids[i])

            if self.origin_is_site[i]:
                mujoco.mj_jacSite(self.model, self.data, jac_origin, None, self.origin_ids[i])
            else:
                mujoco.mj_jacBody(self.model, self.data, jac_origin, None, self.origin_ids[i])

            grad += w * (grad_vec @ (jac_task - jac_origin))

        if self._last_qpos is not None:
            delta_q = qpos - self._last_qpos
            loss += self._norm_delta * np.sum(delta_q ** 2)
            grad += 2.0 * self._norm_delta * delta_q

        # Angle constraints: direct joint angle matching
        if self._target_angles is not None:
            for k in range(len(self._angle_qpos_ids)):
                qadr = self._angle_qpos_ids[k]
                dadr = self._angle_dof_ids[k]
                target = self._target_angles[k]
                w = self._angle_weights[k]
                diff = qpos[qadr] - target
                loss += w * diff * diff
                grad[dadr] += 2.0 * w * diff

        # Pinch fingertip attraction loss
        if self._pinch_enabled and self._pinch_alpha_thumb > 1e-6:
            thumb_pos = self.data.site_xpos[self._thumb_site_id]
            jac_thumb = np.zeros((3, nv))
            mujoco.mj_jacSite(
                self.model, self.data, jac_thumb, None, self._thumb_site_id
            )
            for i in range(4):
                alpha = self._pinch_alphas[i]
                if alpha < 1e-6:
                    continue
                finger_pos = self.data.site_xpos[self._finger_site_ids[i]]
                diff = thumb_pos - finger_pos
                coeff = alpha * self._pinch_weight
                loss += coeff * np.dot(diff, diff)
                jac_finger = np.zeros((3, nv))
                mujoco.mj_jacSite(
                    self.model, self.data, jac_finger, None,
                    self._finger_site_ids[i],
                )
                grad += 2.0 * coeff * (diff @ (jac_thumb - jac_finger))

        # Position constraints: wrist-relative position matching
        if self._pos_enabled and self._position_targets is not None:
            wrist_pos = self._get_pos(self._wrist_body_id, self._wrist_is_site)
            jac_wrist = np.zeros((3, nv))
            if self._wrist_is_site:
                mujoco.mj_jacSite(
                    self.model, self.data, jac_wrist, None, self._wrist_body_id
                )
            else:
                mujoco.mj_jacBody(
                    self.model, self.data, jac_wrist, None, self._wrist_body_id
                )
            for k in range(len(self._pos_body_ids)):
                body_pos = self._get_pos(
                    self._pos_body_ids[k], self._pos_body_is_site[k]
                )
                robot_rel = body_pos - wrist_pos
                diff = robot_rel - self._position_targets[k]
                w = self._pos_weight * self._pos_per_weights[k]
                loss += w * np.dot(diff, diff)
                jac_body = np.zeros((3, nv))
                if self._pos_body_is_site[k]:
                    mujoco.mj_jacSite(
                        self.model, self.data, jac_body, None,
                        self._pos_body_ids[k],
                    )
                else:
                    mujoco.mj_jacBody(
                        self.model, self.data, jac_body, None,
                        self._pos_body_ids[k],
                    )
                grad += 2.0 * w * (diff @ (jac_body - jac_wrist))

        return loss, grad

    def update_targets(self, landmarks_3d: np.ndarray, handedness: str = "Right"):
        landmarks = preprocess_landmarks(
            landmarks_3d,
            handedness=handedness,
            frame=self._preprocess_frame,
        )
        landmarks = self.landmark_filter.filter(landmarks)

        directions = np.empty((len(self.human_vector_pairs), 3), dtype=np.float64)
        for i, (origin_idx, target_idx) in enumerate(self.human_vector_pairs):
            v = landmarks[target_idx] - landmarks[origin_idx]
            norm = np.linalg.norm(v)
            if norm < 1e-8:
                directions[i] = 0.0
            else:
                directions[i] = v / norm
        self._target_directions = directions

        # Compute position targets for thumb joints
        if self._pos_enabled:
            human_palm_size = np.linalg.norm(landmarks[self._scale_landmark_idx])
            scale = self._robot_palm_size / max(human_palm_size, 1e-6)
            n_pc = len(self._pos_landmark_indices)
            targets = np.empty((n_pc, 3), dtype=np.float64)
            for k in range(n_pc):
                targets[k] = scale * landmarks[self._pos_landmark_indices[k]]
            self._position_targets = targets

        # Compute pinch alphas from human landmarks
        if self._pinch_enabled:
            thumb_pos = landmarks[_THUMB_TIP_IDX]
            for i, tip_idx in enumerate(_FINGER_TIP_INDICES):
                dist = np.linalg.norm(landmarks[tip_idx] - thumb_pos)
                d1, d2 = self._pinch_d1, self._pinch_d2
                self._pinch_alphas[i] = np.clip(
                    (d2 - dist) / (d2 - d1 + 1e-8), 0.0, 1.0
                )
            self._pinch_alpha_thumb = float(np.max(self._pinch_alphas))

        # Compute target angles from human landmarks
        if self._angle_landmarks:
            target_angles = np.zeros(len(self._angle_landmarks))
            for k, (a, b, c) in enumerate(self._angle_landmarks):
                v_ba = landmarks[a] - landmarks[b]
                v_bc = landmarks[c] - landmarks[b]
                n_ba = np.linalg.norm(v_ba)
                n_bc = np.linalg.norm(v_bc)
                if n_ba < 1e-8 or n_bc < 1e-8:
                    flexion = 0.0
                else:
                    cos_angle = np.clip(
                        np.dot(v_ba, v_bc) / (n_ba * n_bc), -1.0, 1.0
                    )
                    flexion = np.pi - np.arccos(cos_angle)
                # Map human flexion [0, pi] to robot joint range [lo, hi]
                lo, hi = self._angle_joint_ranges[k]
                target_angles[k] = lo + (flexion / np.pi) * (hi - lo)
            self._target_angles = target_angles

    def solve(self) -> np.ndarray:
        if self._target_directions is None:
            return self.data.qpos.copy()

        x0 = self.data.qpos.copy()
        previous_qpos = None if self._last_qpos is None else self._last_qpos.copy()

        result = minimize(
            fun=self._compute_loss_and_grad,
            x0=x0,
            method="SLSQP",
            jac=True,
            bounds=self._bounds,
            options={
                "maxiter": self._max_iterations,
                "ftol": 1e-6,
            },
        )

        qpos = result.x.copy()
        if previous_qpos is not None and self._output_alpha < 1.0:
            qpos = previous_qpos + self._output_alpha * (qpos - previous_qpos)

        self._last_qpos = qpos.copy()
        self._forward(qpos)
        return qpos

    def compute_error(self) -> float:
        self._forward()
        if self._target_directions is None:
            return 0.0
        return self._compute_loss(self.data.qpos.copy())

    def get_target_directions(self) -> np.ndarray | None:
        if self._target_directions is None:
            return None
        return self._target_directions.copy()
