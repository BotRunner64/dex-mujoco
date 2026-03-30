"""YAML configuration loader for hand model and retargeting mapping."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SolverConfig:
    max_iterations: int = 30
    norm_delta: float = 0.01
    output_alpha: float = 0.70


@dataclass
class PreprocessConfig:
    frame: str = "wrist_local"
    temporal_filter_alpha: float = 0.35


@dataclass
class HandConfig:
    name: str = ""
    mjcf_path: str = ""
    urdf_source: str = ""


@dataclass
class PinchConfig:
    """Pinch-aware dynamic weighting for thumb retargeting.

    When the human thumb tip approaches a fingertip (distance < d2),
    thumb vector weights are boosted and a fingertip attraction loss
    pulls the robot thumb toward the corresponding finger.
    """
    enabled: bool = False
    d1: float = 0.03        # full pinch threshold (meters)
    d2: float = 0.06        # no pinch threshold (meters)
    weight: float = 5.0     # fingertip attraction loss weight
    thumb_weight_boost: float = 1.5  # multiplier for thumb vector weights when pinching
    fingertip_sites: list[str] = field(default_factory=list)  # [thumb, index, middle, ring, pinky]


@dataclass
class PositionConstraint:
    """Map a MediaPipe landmark to a robot body/site position."""
    landmark: int = 0
    body: str = ""
    body_type: str = "body"  # "body" or "site"
    weight: float = 1.0


@dataclass
class PositionConfig:
    """Position-based constraints for thumb retargeting.

    Adds wrist-relative position matching loss for intermediate joints,
    forcing actual bending instead of just direction matching.
    Auto-computes human-to-robot scale from palm geometry.
    """
    enabled: bool = False
    weight: float = 8.0              # global position loss weight
    scale_landmarks: list[int] = field(default_factory=lambda: [0, 9])
    scale_bodies: list[str] = field(default_factory=lambda: ["world", "middle_proximal"])
    scale_body_types: list[str] = field(default_factory=lambda: ["body", "body"])
    constraints: list[PositionConstraint] = field(default_factory=list)


@dataclass
class AngleConstraint:
    """Map human landmark angle to robot joint angle.

    ``landmarks`` is a triple [a, b, c] of MediaPipe indices; the flexion
    angle is measured at *b* between vectors (a→b) and (b→c).
    ``joint`` is the robot joint name to constrain.
    """
    landmarks: list[int] = field(default_factory=list)
    joint: str = ""
    weight: float = 1.0


@dataclass
class RetargetingConfig:
    hand: HandConfig = field(default_factory=HandConfig)
    human_vector_pairs: list[list[int]] = field(default_factory=list)
    origin_link_names: list[str] = field(default_factory=list)
    task_link_names: list[str] = field(default_factory=list)
    origin_link_types: list[str] = field(default_factory=list)
    task_link_types: list[str] = field(default_factory=list)
    vector_weights: list[float] = field(default_factory=list)
    angle_constraints: list[AngleConstraint] = field(default_factory=list)
    pinch: PinchConfig = field(default_factory=PinchConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    @classmethod
    def load(cls, config_path: str) -> "RetargetingConfig":
        """Load config from a YAML file."""
        config_path = Path(config_path)
        with open(config_path) as f:
            data = yaml.safe_load(f)

        config = cls()

        hand_data = data.get("hand", {})
        if isinstance(hand_data, str):
            hand_path = config_path.parent / hand_data
            with open(hand_path) as f:
                hand_data = yaml.safe_load(f)

        mjcf_path = Path(hand_data.get("mjcf_path", ""))
        if not mjcf_path.is_absolute():
            mjcf_path = (config_path.parent / mjcf_path).resolve()

        config.hand = HandConfig(
            name=hand_data.get("name", ""),
            mjcf_path=str(mjcf_path),
            urdf_source=hand_data.get("urdf_source", ""),
        )

        rt = data.get("retargeting", {})
        config.human_vector_pairs = rt.get("human_vector_pairs", [])
        config.origin_link_names = rt.get("origin_link_names", [])
        config.task_link_names = rt.get("task_link_names", [])
        config.origin_link_types = rt.get(
            "origin_link_types", ["body"] * len(config.origin_link_names)
        )
        config.task_link_types = rt.get(
            "task_link_types", ["body"] * len(config.task_link_names)
        )
        config.vector_weights = rt.get(
            "vector_weights", [1.0] * len(config.human_vector_pairs)
        )

        for ac_data in rt.get("angle_constraints", []):
            config.angle_constraints.append(AngleConstraint(
                landmarks=ac_data["landmarks"],
                joint=ac_data["joint"],
                weight=ac_data.get("weight", 1.0),
            ))

        pos_data = rt.get("position_constraints", {})
        if pos_data and pos_data.get("enabled", False):
            constraints = []
            for pc in pos_data.get("constraints", []):
                constraints.append(PositionConstraint(
                    landmark=pc["landmark"],
                    body=pc["body"],
                    body_type=pc.get("body_type", "body"),
                    weight=pc.get("weight", 1.0),
                ))
            config.position = PositionConfig(
                enabled=True,
                weight=pos_data.get("weight", 8.0),
                scale_landmarks=pos_data.get("scale_landmarks", [0, 9]),
                scale_bodies=pos_data.get("scale_bodies", ["world", "middle_proximal"]),
                scale_body_types=pos_data.get("scale_body_types", ["body", "body"]),
                constraints=constraints,
            )

        pinch_data = rt.get("pinch", {})
        if pinch_data:
            config.pinch = PinchConfig(
                enabled=pinch_data.get("enabled", False),
                d1=pinch_data.get("d1", 0.03),
                d2=pinch_data.get("d2", 0.06),
                weight=pinch_data.get("weight", 5.0),
                thumb_weight_boost=pinch_data.get("thumb_weight_boost", 1.5),
                fingertip_sites=pinch_data.get("fingertip_sites", []),
            )

        preprocess_data = rt.get("preprocess", {})
        config.preprocess = PreprocessConfig(**{
            k: v for k, v in preprocess_data.items()
            if k in PreprocessConfig.__dataclass_fields__
        })

        solver_data = rt.get("solver", {})
        config.solver = SolverConfig(**{
            k: v for k, v in solver_data.items()
            if k in SolverConfig.__dataclass_fields__
        })

        config.validate()
        return config

    def validate(self):
        n = len(self.human_vector_pairs)
        assert len(self.origin_link_names) == n, (
            f"origin_link_names length ({len(self.origin_link_names)}) "
            f"must match human_vector_pairs length ({n})"
        )
        assert len(self.task_link_names) == n, (
            f"task_link_names length ({len(self.task_link_names)}) "
            f"must match human_vector_pairs length ({n})"
        )
        assert len(self.origin_link_types) == n, (
            f"origin_link_types length ({len(self.origin_link_types)}) "
            f"must match human_vector_pairs length ({n})"
        )
        assert len(self.task_link_types) == n, (
            f"task_link_types length ({len(self.task_link_types)}) "
            f"must match human_vector_pairs length ({n})"
        )
        assert len(self.vector_weights) == n, (
            f"vector_weights length ({len(self.vector_weights)}) "
            f"must match human_vector_pairs length ({n})"
        )
        assert self.preprocess.frame in {"camera_aligned", "wrist_local"}, (
            f"Unsupported preprocess.frame: {self.preprocess.frame}"
        )
        assert 0.0 < self.preprocess.temporal_filter_alpha <= 1.0, (
            "temporal_filter_alpha must be in (0, 1]"
        )
        assert 0.0 < self.solver.output_alpha <= 1.0, (
            "solver.output_alpha must be in (0, 1]"
        )
        assert all(link_type in {"body", "site"} for link_type in self.origin_link_types), (
            "origin_link_types must only contain 'body' or 'site'"
        )
        assert all(link_type in {"body", "site"} for link_type in self.task_link_types), (
            "task_link_types must only contain 'body' or 'site'"
        )
        assert Path(self.hand.mjcf_path).exists(), (
            f"MJCF file not found: {self.hand.mjcf_path}"
        )
