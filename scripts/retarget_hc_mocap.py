"""Retarget dex hand from Teleopit hc_mocap hand data."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco.hand_model import HandModel
from dex_mujoco.hc_mocap_input import (
    create_hc_mocap_bvh_provider,
    create_hc_mocap_udp_provider,
)
from dex_mujoco.landmark_visualization import MediaPipe3DVisualizer
from dex_mujoco.retargeting_config import RetargetingConfig
from dex_mujoco.vector_retargeting import VectorRetargeter, preprocess_landmarks
from dex_mujoco.visualization import HandVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Retarget from Teleopit hc_mocap hand input")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bvh", help="Offline hc_mocap BVH file")
    source.add_argument(
        "--reference-bvh",
        help="Reference hc_mocap BVH file for UDP realtime input",
    )
    parser.add_argument(
        "--config",
        default="configs/retargeting/linkerhand_l20.yaml",
        help="Path to retargeting config YAML",
    )
    parser.add_argument(
        "--hand",
        choices=["Left", "Right"],
        default="Right",
        help="Which hand to extract from hc_mocap",
    )
    parser.add_argument(
        "--teleopit-root",
        default=None,
        help="Optional Teleopit repo root for offline BVH loading if the package is not installed",
    )
    parser.add_argument("--udp-host", default="", help="UDP bind host for hc_mocap input")
    parser.add_argument("--udp-port", type=int, default=1118, help="UDP port for hc_mocap input")
    parser.add_argument("--udp-timeout", type=float, default=30.0, help="UDP startup timeout in seconds")
    parser.add_argument(
        "--udp-stats-every",
        type=int,
        default=120,
        help="Print UDP receive statistics every N processed frames (0 disables)",
    )
    parser.add_argument("--output", default=None, help="Output pickle file for joint trajectory")
    parser.add_argument("--visualize", action="store_true", help="Show MuJoCo viewer")
    parser.add_argument("--viser", action="store_true", help="Show 3D landmarks in a browser via viser")
    parser.add_argument("--viser-host", default="127.0.0.1", help="Host address for the viser server")
    parser.add_argument("--viser-port", type=int, default=8080, help="Port for the viser server")
    parser.add_argument(
        "--viser-space",
        choices=["local", "raw"],
        default="local",
        help="Which landmark coordinates to render in viser",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep between offline BVH frames using the source fps",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop offline hc_mocap BVH input indefinitely",
    )
    args = parser.parse_args()

    config = RetargetingConfig.load(args.config)
    if config.preprocess.frame == "wrist_local":
        print(
            "hc_mocap input detected: overriding preprocess.frame "
            "from wrist_local to camera_aligned and using wrist-pose local landmarks."
        )
        config.preprocess.frame = "camera_aligned"

    hand_model = HandModel(config.hand.mjcf_path)
    retargeter = VectorRetargeter(hand_model, config)

    if args.bvh:
        provider = create_hc_mocap_bvh_provider(
            bvh_path=args.bvh,
            handedness=args.hand,
            teleopit_root=args.teleopit_root,
        )
        source_desc = args.bvh
    else:
        provider = create_hc_mocap_udp_provider(
            reference_bvh=args.reference_bvh,
            handedness=args.hand,
            host=args.udp_host,
            port=args.udp_port,
            timeout=args.udp_timeout,
        )
        source_desc = f"udp://{args.udp_host or '0.0.0.0'}:{args.udp_port}"

    visualizer = HandVisualizer(hand_model) if args.visualize else None
    mp_visualizer = None
    if args.viser:
        mp_visualizer = MediaPipe3DVisualizer(
            host=args.viser_host,
            port=args.viser_port,
            space=args.viser_space,
        )

    print(f"Model: {config.hand.name} ({hand_model.nq} DOF)")
    print(f"Retargeting: {len(config.human_vector_pairs)} vector pairs")
    print(f"HC mocap source: {source_desc}")
    print(f"Tracking hand: {args.hand} | Source fps: {provider.fps}")
    if args.reference_bvh:
        stats = provider.stats_snapshot()
        if stats:
            print(
                "UDP packet format:"
                f" expected_floats={stats.get('expected_float_count', 0)}"
                f" bind={args.udp_host or '0.0.0.0'}:{args.udp_port}"
            )
    if mp_visualizer is not None:
        print(f"HC mocap 3D viewer ({args.viser_space}): {mp_visualizer.url}")

    trajectory: list[np.ndarray] = []
    frame_count = 0
    frame_period = 1.0 / max(provider.fps, 1)

    try:
        while True:
            if not provider.is_available():
                if args.bvh and args.loop:
                    reset_fn = getattr(provider._provider, "reset", None)
                    if callable(reset_fn):
                        reset_fn()
                    else:
                        break
                else:
                    break

            tic = time.monotonic()
            detection = provider.get_detection()
            frame_count += 1

            retarget_landmarks = detection.landmarks_3d_local
            if retarget_landmarks is None:
                retarget_landmarks = detection.landmarks_3d

            retargeter.update_targets(retarget_landmarks, detection.handedness)
            qpos = retargeter.solve()
            trajectory.append(qpos.copy())

            if visualizer is not None:
                visualizer.update(qpos)
                if not visualizer.is_running:
                    break

            if mp_visualizer is not None:
                if args.viser_space == "local":
                    vis_landmarks = retarget_landmarks
                    landmarks_for_vis = preprocess_landmarks(
                        vis_landmarks,
                        handedness=detection.handedness,
                        frame=config.preprocess.frame,
                    )
                else:
                    landmarks_for_vis = detection.landmarks_3d
                mp_visualizer.update(landmarks_for_vis)

            if args.reference_bvh and args.udp_stats_every > 0 and frame_count % args.udp_stats_every == 0:
                stats = provider.stats_snapshot()
                if stats:
                    print(
                        "UDP stats:"
                        f" recv={stats.get('packets_received', 0)}"
                        f" valid={stats.get('packets_valid', 0)}"
                        f" bad_size={stats.get('packets_bad_size', 0)}"
                        f" bad_decode={stats.get('packets_bad_decode', 0)}"
                        f" floats={stats.get('last_float_count', 0)}/{stats.get('expected_float_count', 0)}"
                        f" bytes={stats.get('last_packet_bytes', 0)}"
                        f" sender={stats.get('last_sender')}"
                    )

            if args.bvh and args.realtime:
                elapsed = time.monotonic() - tic
                sleep_s = frame_period - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        provider.close()
        if mp_visualizer is not None:
            mp_visualizer.close()
        if visualizer is not None:
            visualizer.close()

    print(f"Processed {frame_count} hc_mocap hand frames")

    if args.output and trajectory:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "trajectory": np.array(trajectory),
            "joint_names": hand_model.get_joint_names(),
            "config_path": args.config,
            "num_frames": frame_count,
            "input_source": source_desc,
            "input_type": "hc_mocap",
            "handedness": args.hand,
        }
        with output_path.open("wb") as f:
            pickle.dump(data, f)
        print(f"Saved trajectory ({len(trajectory)} frames) to {args.output}")


if __name__ == "__main__":
    main()
