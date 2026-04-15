"""Acceptance script for somehand retargeting quality."""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.acceptance import (
    AcceptanceResult,
    bilateral_preprocess_consistency_score,
    current_alignment_metrics,
    rotation_invariance_score,
    solver_quality_score,
    static_jitter_score,
    synthetic_hand_pose,
    throughput_score,
)
from somehand.domain import normalize_hand_side
from somehand.infrastructure.config_loader import load_retargeting_config
from somehand.hand_detector import HandDetector
from somehand.infrastructure.hand_model import HandModel
from somehand.infrastructure.vector_solver import VectorRetargeter


def _result_to_dict(result: AcceptanceResult) -> dict:
    return {
        "name": result.name,
        "passed": result.passed,
        "metrics": result.metrics,
        "detail": result.detail,
    }


def run_video_check(
    retargeter: VectorRetargeter,
    video_path: str,
    hand: str,
    swap_hands: bool,
    min_detection_rate: float,
    min_video_cos: float,
) -> AcceptanceResult:
    detector = HandDetector(target_hand=hand, swap_handedness=swap_hands)
    total_frames = 0
    detected_frames = 0
    weighted_cosines = []
    mean_cosines = []
    min_cosines = []
    losses = []
    frame_primary_cosines = []
    frame_secondary_cosines = []

    try:
        for frame in HandDetector.create_source(video_path):
            total_frames += 1
            detection = detector.detect(frame)
            if detection is None:
                continue

            detected_frames += 1
            retargeter.update_targets(detection.landmarks_3d, hand_side=detection.hand_side)
            retargeter.solve()
            metrics = current_alignment_metrics(retargeter)
            weighted_cosines.append(metrics["weighted_cosine"])
            mean_cosines.append(metrics["mean_cosine"])
            min_cosines.append(metrics["min_cosine"])
            if "thumb_frame_primary_cosine" in metrics:
                frame_primary_cosines.append(metrics["thumb_frame_primary_cosine"])
            if "thumb_frame_secondary_cosine" in metrics:
                frame_secondary_cosines.append(metrics["thumb_frame_secondary_cosine"])
            losses.append(retargeter.compute_error())
    finally:
        detector.close()

    detection_rate = detected_frames / max(total_frames, 1)
    mean_weighted_cosine = float(sum(weighted_cosines) / len(weighted_cosines)) if weighted_cosines else float("-inf")
    mean_cosine = float(sum(mean_cosines) / len(mean_cosines)) if mean_cosines else float("-inf")
    min_cosine = float(sum(min_cosines) / len(min_cosines)) if min_cosines else float("-inf")
    mean_loss = float(sum(losses) / len(losses)) if losses else float("inf")
    passed = detection_rate >= min_detection_rate and mean_weighted_cosine >= min_video_cos
    metrics = {
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detection_rate": detection_rate,
        "mean_weighted_cosine": mean_weighted_cosine,
        "mean_cosine": mean_cosine,
        "mean_min_cosine": min_cosine,
        "mean_loss": mean_loss,
    }
    if frame_primary_cosines:
        metrics["mean_thumb_frame_primary_cosine"] = float(sum(frame_primary_cosines) / len(frame_primary_cosines))
    if frame_secondary_cosines:
        metrics["mean_thumb_frame_secondary_cosine"] = float(
            sum(frame_secondary_cosines) / len(frame_secondary_cosines)
        )
    return AcceptanceResult(
        name="video_regression",
        passed=passed,
        metrics=metrics,
        detail="Offline video acceptance on recorded webcam footage.",
    )


def main():
    parser = argparse.ArgumentParser(description="Run somehand acceptance checks")
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Path to retargeting config YAML (repeatable)",
    )
    parser.add_argument(
        "--config-glob",
        action="append",
        default=[],
        help="Glob for config YAMLs, e.g. 'configs/retargeting/right/*_right.yaml'",
    )
    parser.add_argument("--video", default=None, help="Optional input video for offline acceptance")
    parser.add_argument("--hand", type=normalize_hand_side, choices=["left", "right"], default="right")
    parser.add_argument("--swap-hands", action="store_true")
    parser.add_argument("--json", default=None, help="Optional JSON report path")
    parser.add_argument("--min-rotation-cos", type=float, default=0.98)
    parser.add_argument("--min-mirror-cos", type=float, default=0.98)
    parser.add_argument("--max-static-jitter", type=float, default=0.02)
    parser.add_argument("--min-weighted-cos", type=float, default=0.72)
    parser.add_argument("--min-pose-weighted-cos", type=float, default=0.35)
    parser.add_argument("--min-thumb-frame-cos", type=float, default=0.80)
    parser.add_argument("--min-fps", type=float, default=20.0)
    parser.add_argument("--min-detection-rate", type=float, default=0.60)
    parser.add_argument("--min-video-cos", type=float, default=0.68)
    parser.add_argument("--max-pinch-gap-scaled", type=float, default=0.80)
    parser.add_argument("--max-fist-closure-ratio", type=float, default=1.02)
    args = parser.parse_args()

    config_paths = list(args.configs or [])
    for pattern in args.config_glob:
        config_paths.extend(sorted(glob.glob(pattern)))
    if not config_paths:
        config_paths = ["configs/retargeting/right/linkerhand_l20_right.yaml"]

    reports = []
    all_passed = True
    for config_path in config_paths:
        config = load_retargeting_config(config_path)
        hand_model = HandModel(config.hand.mjcf_path)
        retargeter = VectorRetargeter(hand_model, config)
        vector_pairs = [(a, b) for a, b in config.human_vector_pairs]

        results = []

        rotation_cos = rotation_invariance_score(config, vector_pairs)
        results.append(
            AcceptanceResult(
                name="rotation_invariance",
                passed=rotation_cos >= args.min_rotation_cos,
                metrics={"min_mean_direction_cosine": rotation_cos},
                detail="Rigid palm rotations should not change articulation targets.",
            )
        )

        mirror_cos = bilateral_preprocess_consistency_score(config, vector_pairs)
        results.append(
            AcceptanceResult(
                name="bilateral_preprocess_consistency",
                passed=mirror_cos >= args.min_mirror_cos,
                metrics={"mean_direction_cosine": mirror_cos},
                detail="Left/right operator inputs should remain self-consistent after side-specific preprocessing.",
            )
        )

        jitter = static_jitter_score(retargeter, pose=synthetic_hand_pose("open"))
        results.append(
            AcceptanceResult(
                name="static_jitter",
                passed=jitter <= args.max_static_jitter,
                metrics={"max_qpos_delta_norm": jitter},
                detail="Repeated identical input should converge to a stable joint configuration.",
            )
        )

        quality = solver_quality_score(retargeter)
        thumb_frame_ok = True
        if "mean_thumb_frame_primary_cosine" in quality:
            thumb_frame_ok = thumb_frame_ok and quality["mean_thumb_frame_primary_cosine"] >= args.min_thumb_frame_cos
        if "mean_thumb_frame_secondary_cosine" in quality:
            thumb_frame_ok = thumb_frame_ok and quality["mean_thumb_frame_secondary_cosine"] >= args.min_thumb_frame_cos
        closure_ok = True
        if "pinch_thumb_index_gap_scaled" in quality:
            closure_ok = closure_ok and quality["pinch_thumb_index_gap_scaled"] <= args.max_pinch_gap_scaled
        if "fist_closure_ratio" in quality:
            closure_ok = closure_ok and quality["fist_closure_ratio"] <= args.max_fist_closure_ratio
        results.append(
            AcceptanceResult(
                name="solver_quality",
                passed=(
                    quality["mean_weighted_cosine"] >= args.min_weighted_cos
                    and quality["min_weighted_cosine"] >= args.min_pose_weighted_cos
                    and thumb_frame_ok
                    and closure_ok
                ),
                metrics=quality,
                detail="Synthetic representative poses should maintain stable direction alignment and closure after solving.",
            )
        )

        fps = throughput_score(retargeter)
        results.append(
            AcceptanceResult(
                name="throughput",
                passed=fps >= args.min_fps,
                metrics={"fps": fps},
                detail="Synthetic sequence throughput on the current machine.",
            )
        )

        if args.video:
            results.append(
                run_video_check(
                    retargeter=retargeter,
                    video_path=args.video,
                    hand=args.hand,
                    swap_hands=args.swap_hands,
                    min_detection_rate=args.min_detection_rate,
                    min_video_cos=args.min_video_cos,
                )
            )

        passed = all(result.passed for result in results)
        all_passed = all_passed and passed
        reports.append(
            {
                "config": str(Path(config_path).resolve()),
                "hand": config.hand.name,
                "hand_side": config.hand.side,
                "all_passed": passed,
                "results": [_result_to_dict(result) for result in results],
            }
        )

        print(f"=== {config.hand.name} ({config_path}) ===")
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {result.name}: {json.dumps(result.metrics, ensure_ascii=False)}")

    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = reports[0] if len(reports) == 1 else {"all_passed": all_passed, "reports": reports}
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"Saved report to {output_path}")

    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
