import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco.cli import build_parser
from dex_mujoco.paths import DEFAULT_CONFIG_PATH, DEFAULT_HC_MOCAP_REFERENCE_BVH


def test_hc_mocap_udp_uses_repo_defaults():
    parser = build_parser()
    args = parser.parse_args(["hc-mocap", "udp"])

    assert args.command == "hc-mocap"
    assert args.hc_command == "udp"
    assert args.config == str(DEFAULT_CONFIG_PATH)
    assert args.reference_bvh == str(DEFAULT_HC_MOCAP_REFERENCE_BVH)
    assert args.udp_port == 1118
    assert args.hand == "Right"


def test_video_command_requires_video_path():
    parser = build_parser()
    args = parser.parse_args(["video", "--video", "input.mp4", "--hand", "Left"])

    assert args.command == "video"
    assert args.video == "input.mp4"
    assert args.hand == "Left"


def test_webcam_command_uses_default_camera():
    parser = build_parser()
    args = parser.parse_args(["webcam", "--visualize"])

    assert args.command == "webcam"
    assert args.camera == 0
    assert args.visualize is True
