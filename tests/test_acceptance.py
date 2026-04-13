import sys
from pathlib import Path

import numpy as np
import pytest
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.acceptance import mirror_pose_to_left, rotation_matrix, synthetic_hand_pose
from somehand.domain.preprocessing import compute_target_directions, preprocess_landmarks
from somehand.infrastructure.artifacts import load_hand_recording_artifact
from somehand.infrastructure.config_loader import load_retargeting_config
from somehand.infrastructure.hand_model import HandModel
from somehand.infrastructure.vector_solver import VectorRetargeter
from somehand.acceptance import current_alignment_metrics

_LEFT_RIGHT_ROBOT_MIRROR = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


def _asymmetric_pose() -> np.ndarray:
    pose = synthetic_hand_pose("pinch")
    pose[[2, 3, 4], 2] += [0.010, 0.020, 0.030]
    pose[[14, 15, 16], 2] += [0.005, 0.015, 0.025]
    pose[[17, 18, 19, 20], 0] -= [0.000, 0.002, 0.004, 0.006]
    return pose


def test_config_resolves_absolute_mjcf_path():
    config = load_retargeting_config("configs/retargeting/right/linkerhand_l20_right.yaml")
    assert Path(config.hand.mjcf_path).is_absolute()
    assert Path(config.hand.mjcf_path).exists()


def test_wrist_local_preprocess_is_rotation_invariant():
    config = load_retargeting_config("configs/retargeting/right/linkerhand_l20_right.yaml")
    vector_pairs = [(a, b) for a, b in config.human_vector_pairs]
    base_pose = synthetic_hand_pose("open")
    base_dirs = compute_target_directions(
        base_pose,
        vector_pairs,
        hand_side="right",
    )
    rotated_pose = base_pose @ rotation_matrix("z", 70.0).T
    rotated_dirs = compute_target_directions(
        rotated_pose,
        vector_pairs,
        hand_side="right",
    )
    cosine = float(np.mean(np.sum(base_dirs * rotated_dirs, axis=1)))
    assert cosine > 0.98


def test_left_and_right_inputs_match_after_mirroring():
    config = load_retargeting_config("configs/retargeting/right/linkerhand_l20_right.yaml")
    vector_pairs = [(a, b) for a, b in config.human_vector_pairs]
    right_pose = synthetic_hand_pose("pinch")
    left_pose = mirror_pose_to_left(right_pose)
    right_dirs = compute_target_directions(
        right_pose,
        vector_pairs,
        hand_side="right",
    )
    left_dirs = compute_target_directions(
        left_pose,
        vector_pairs,
        hand_side="left",
    )
    cosine = float(np.mean(np.sum(right_dirs * (left_dirs @ _LEFT_RIGHT_ROBOT_MIRROR), axis=1)))
    assert cosine > 0.98


def test_left_and_right_preprocess_match_for_asymmetric_3d_pose():
    right_pose = _asymmetric_pose()
    left_pose = mirror_pose_to_left(right_pose)

    right_processed = preprocess_landmarks(right_pose, hand_side="right")
    left_processed = preprocess_landmarks(left_pose, hand_side="left")

    assert np.allclose(left_processed, right_processed @ _LEFT_RIGHT_ROBOT_MIRROR)


def test_wrist_local_preprocess_matches_reference_operator_frame():
    pose = synthetic_hand_pose("pinch")
    centered = pose - pose[0:1, :]
    points = centered[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    points = points - np.mean(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(points, full_matrices=False)
    normal = vh[-1] / np.linalg.norm(vh[-1])
    x_axis = x_vector - np.dot(x_vector, normal) * normal
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, normal)
    z_axis = z_axis / np.linalg.norm(z_axis)
    if np.dot(z_axis, points[1] - points[2]) < 0.0:
        normal *= -1.0
        z_axis *= -1.0

    operator2robot = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    expected = centered @ np.stack([x_axis, normal, z_axis], axis=1) @ operator2robot
    actual = preprocess_landmarks(pose, hand_side="right")
    assert np.allclose(actual, expected)


def test_thumb_frame_is_local_to_cmc_body():
    config = load_retargeting_config("configs/retargeting/right/linkerhand_l21_right.yaml")
    hand_model = HandModel(config.hand.mjcf_path)
    retargeter = VectorRetargeter(hand_model, config)

    base_primary, base_secondary = retargeter._get_robot_frame_axes()
    thumb_tip_site_id = mujoco.mj_name2id(
        hand_model.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "thumb_distal_tip",
    )
    thumb_tip_before = hand_model.data.site_xpos[thumb_tip_site_id].copy()

    qpos = hand_model.get_qpos().copy()
    name_to_idx = hand_model.get_joint_name_to_qpos_index()
    qpos[name_to_idx["thumb_mcp"]] = 1.0
    qpos[name_to_idx["thumb_ip"]] = 1.0
    hand_model.set_qpos(qpos)
    retargeter._forward(hand_model.get_qpos())

    primary_after, secondary_after = retargeter._get_robot_frame_axes()
    thumb_tip_after = hand_model.data.site_xpos[thumb_tip_site_id].copy()

    assert np.linalg.norm(thumb_tip_after - thumb_tip_before) > 1e-3
    assert np.allclose(primary_after, base_primary)
    assert np.allclose(secondary_after, base_secondary)


def test_thumb_frame_responds_to_cmc_joints():
    """Frame axes must change when CMC roll or pitch change, proving coverage."""
    config = load_retargeting_config("configs/retargeting/right/linkerhand_l20_right.yaml")
    hand_model = HandModel(config.hand.mjcf_path)
    retargeter = VectorRetargeter(hand_model, config)

    base_primary, base_secondary = retargeter._get_robot_frame_axes()
    name_to_idx = hand_model.get_joint_name_to_qpos_index()

    for joint_name in ("thumb_cmc_roll", "thumb_cmc_pitch"):
        qpos = hand_model.get_qpos().copy()
        joint_range = hand_model.model.jnt_range[
            mujoco.mj_name2id(hand_model.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        ]
        qpos[name_to_idx[joint_name]] = joint_range[1]
        hand_model.set_qpos(qpos)
        retargeter._forward(hand_model.get_qpos())
        primary_after, secondary_after = retargeter._get_robot_frame_axes()

        axes_changed = (
            not np.allclose(primary_after, base_primary, atol=1e-3)
            or not np.allclose(secondary_after, base_secondary, atol=1e-3)
        )
        assert axes_changed, f"Frame axes did not change when {joint_name} was set to max"

        hand_model.reset()
        retargeter._forward(hand_model.get_qpos())


def test_thumb_cmc_joints_engage_during_pinch():
    """CMC joints must produce meaningful thumb movement during pinch — not just dip maxing out."""
    config = load_retargeting_config("configs/retargeting/right/linkerhand_l20_right.yaml")
    hand_model = HandModel(config.hand.mjcf_path)
    retargeter = VectorRetargeter(hand_model, config)

    retargeter.update_targets(synthetic_hand_pose("pinch"), hand_side="right")
    qpos = retargeter.solve()

    name_to_idx = hand_model.get_joint_name_to_qpos_index()
    model = hand_model.model

    def utilization(joint_name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        low, high = model.jnt_range[jid]
        return (qpos[name_to_idx[joint_name]] - low) / (high - low) if high > low else 0.0

    assert utilization("thumb_cmc_yaw") > 0.30, "thumb_cmc_yaw should be significantly engaged during pinch"
    assert utilization("thumb_dip") < 0.95, "thumb_dip should not max out during pinch"


def test_pinch_thumb_tip_reaches_index():
    """After solving pinch, thumb tip should be close to index tip in robot space."""
    config = load_retargeting_config("configs/retargeting/right/linkerhand_l20_right.yaml")
    hand_model = HandModel(config.hand.mjcf_path)
    retargeter = VectorRetargeter(hand_model, config)

    retargeter.update_targets(synthetic_hand_pose("pinch"), hand_side="right")
    retargeter.solve()

    thumb_tip_id = mujoco.mj_name2id(model := hand_model.model, mujoco.mjtObj.mjOBJ_SITE, "thumb_distal_tip")
    index_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "index_distal_tip")
    thumb_pos = hand_model.data.site_xpos[thumb_tip_id]
    index_pos = hand_model.data.site_xpos[index_tip_id]
    distance = np.linalg.norm(thumb_pos - index_pos)

    assert distance < 0.080, f"Thumb-index tip distance {distance:.4f}m too large during pinch (max 80mm)"


@pytest.mark.skipif(
    not Path("recordings/pico_left.pkl").exists(),
    reason="left-hand recording fixture not available",
)
def test_left_recording_replay_quality_regression():
    config = load_retargeting_config("configs/retargeting/left/linkerhand_l20_left.yaml")
    recording = load_hand_recording_artifact("recordings/pico_left.pkl")
    hand_model = HandModel(config.hand.mjcf_path)
    retargeter = VectorRetargeter(hand_model, config)

    weighted_cosines = []
    for frame in recording["frames"][::120]:
        retargeter.update_targets(frame.landmarks_3d, hand_side=frame.hand_side)
        retargeter.solve()
        weighted_cosines.append(current_alignment_metrics(retargeter)["weighted_cosine"])

    assert float(np.mean(weighted_cosines)) > 0.88
