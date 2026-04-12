import sys
from pathlib import Path

import numpy as np
import pytest

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
