import numpy as np
import pytest
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from dex_mujoco.hc_mocap_input import (
    _frame_from_bvh_values,
    _parse_bvh_reference,
    hc_mocap_frame_to_landmarks,
    hc_mocap_frame_to_local_landmarks,
)


def _joint(position):
    return (np.array(position, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]))


def test_hc_mocap_frame_to_landmarks_maps_left_hand_joints():
    frame = {
        "hc_Hand_L": _joint([0.0, 0.0, 0.0]),
        "hc_Thumb1_L": _joint([1.0, 0.0, 0.0]),
        "hc_Thumb2_L": _joint([2.0, 0.0, 0.0]),
        "hc_Thumb3_L": _joint([3.0, 0.0, 0.0]),
        "hc_Index1_L": _joint([0.0, 1.0, 0.0]),
        "hc_Index2_L": _joint([0.0, 2.0, 0.0]),
        "hc_Index3_L": _joint([0.0, 3.0, 0.0]),
        "hc_Middle1_L": _joint([0.0, 1.0, 1.0]),
        "hc_Middle2_L": _joint([0.0, 2.0, 1.0]),
        "hc_Middle3_L": _joint([0.0, 3.0, 1.0]),
        "hc_Ring1_L": _joint([0.0, 1.0, 2.0]),
        "hc_Ring2_L": _joint([0.0, 2.0, 2.0]),
        "hc_Ring3_L": _joint([0.0, 3.0, 2.0]),
        "hc_Pinky1_L": _joint([0.0, 1.0, 3.0]),
        "hc_Pinky2_L": _joint([0.0, 2.0, 3.0]),
        "hc_Pinky3_L": _joint([0.0, 3.0, 3.0]),
    }

    landmarks = hc_mocap_frame_to_landmarks(frame, "Left")

    assert landmarks.shape == (21, 3)
    np.testing.assert_allclose(landmarks[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(landmarks[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(landmarks[3], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(landmarks[4], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(landmarks[5], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(landmarks[7], [0.0, 3.0, 0.0])
    np.testing.assert_allclose(landmarks[8], [0.0, 3.0, 0.0])


def test_hc_mocap_frame_to_local_landmarks_uses_wrist_rotation():
    wrist_position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    wrist_rotation = R.from_euler("z", 90.0, degrees=True)
    wrist_quat = wrist_rotation.as_quat()
    wrist_quat_wxyz = np.array(
        [wrist_quat[3], wrist_quat[0], wrist_quat[1], wrist_quat[2]],
        dtype=np.float64,
    )

    def joint_from_local(local_position):
        world_position = wrist_position + wrist_rotation.apply(np.array(local_position, dtype=np.float64))
        return (world_position, wrist_quat_wxyz)

    frame = {
        "hc_Hand_R": (wrist_position, wrist_quat_wxyz),
        "hc_Thumb1_R": joint_from_local([1.0, 0.0, 0.0]),
        "hc_Thumb2_R": joint_from_local([2.0, 0.0, 0.0]),
        "hc_Thumb3_R": joint_from_local([3.0, 0.0, 0.0]),
        "hc_Index1_R": joint_from_local([0.0, 1.0, 0.0]),
        "hc_Index2_R": joint_from_local([0.0, 2.0, 0.0]),
        "hc_Index3_R": joint_from_local([0.0, 3.0, 0.0]),
        "hc_Middle1_R": joint_from_local([0.0, 1.0, 1.0]),
        "hc_Middle2_R": joint_from_local([0.0, 2.0, 1.0]),
        "hc_Middle3_R": joint_from_local([0.0, 3.0, 1.0]),
        "hc_Ring1_R": joint_from_local([0.0, 1.0, 2.0]),
        "hc_Ring2_R": joint_from_local([0.0, 2.0, 2.0]),
        "hc_Ring3_R": joint_from_local([0.0, 3.0, 2.0]),
        "hc_Pinky1_R": joint_from_local([0.0, 1.0, 3.0]),
        "hc_Pinky2_R": joint_from_local([0.0, 2.0, 3.0]),
        "hc_Pinky3_R": joint_from_local([0.0, 3.0, 3.0]),
    }

    landmarks = hc_mocap_frame_to_local_landmarks(frame, "Right")

    np.testing.assert_allclose(landmarks[0], [0.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(landmarks[1], [1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(landmarks[3], [3.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(landmarks[5], [0.0, 1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(landmarks[19], [0.0, 3.0, 3.0], atol=1e-7)


@pytest.mark.skipif(
    not Path("/home/wubingqian/project/teleop_projects/Teleopit/data/hc_mocap_bvh/motion-20260211203358.bvh").exists(),
    reason="local hc_mocap sample BVH not available",
)
def test_local_bvh_reference_parser_matches_motion_line():
    bvh_path = Path(
        "/home/wubingqian/project/teleop_projects/Teleopit/data/hc_mocap_bvh/motion-20260211203358.bvh"
    )
    skeleton = _parse_bvh_reference(str(bvh_path))

    motion_started = False
    sample_line = None
    for line in bvh_path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "MOTION":
            motion_started = True
            continue
        if not motion_started or stripped.startswith("Frames:") or stripped.startswith("Frame Time:") or not stripped:
            continue
        sample_line = stripped
        break

    assert sample_line is not None
    values = np.fromstring(sample_line, sep=" ", dtype=np.float64)
    assert values.size == skeleton.expected_floats

    frame = _frame_from_bvh_values(skeleton, values)
    assert "hc_Hand_R" in frame
    assert "hc_Index3_R" in frame
    assert frame["hc_Hand_R"][0].shape == (3,)
