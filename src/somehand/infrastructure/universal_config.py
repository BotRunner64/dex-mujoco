"""Universal retargeting preset materialization."""

from __future__ import annotations

from somehand.constants import (
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    LITTLE_DIP,
    LITTLE_MCP,
    LITTLE_PIP,
    LITTLE_TIP,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    RING_DIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
)
from somehand.domain.config import AngleConstraint, DistanceConstraint, FrameConstraint, RetargetingConfig, VectorConstraint


_FINGERS = (
    ("index", INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, 2000.0),
    ("middle", MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, 1500.0),
    ("ring", RING_MCP, RING_PIP, RING_DIP, RING_TIP, 1000.0),
    ("pinky", LITTLE_MCP, LITTLE_PIP, LITTLE_DIP, LITTLE_TIP, 800.0),
)


def apply_universal_preset(config: RetargetingConfig) -> None:
    vector_constraints: list[VectorConstraint] = [
        VectorConstraint(
            human=[WRIST, THUMB_TIP],
            robot=["world", "thumb_tip"],
            robot_types=["body", "site"],
            weight=1.5,
        ),
        VectorConstraint(
            human=[THUMB_CMC, THUMB_MCP],
            robot=["thumb_base", "thumb_mid"],
            weight=1.0,
            optional=True,
        ),
        VectorConstraint(
            human=[THUMB_MCP, THUMB_TIP],
            robot=["thumb_mid", "thumb_tip"],
            robot_types=["body", "site"],
            weight=1.2,
            loss_type="residual",
            loss_scale=1.0,
            optional=True,
        ),
        VectorConstraint(
            human=[THUMB_IP, THUMB_TIP],
            robot=["thumb_distal", "thumb_tip"],
            robot_types=["body", "site"],
            weight=0.9,
            loss_type="residual",
            loss_scale=1.0,
            optional=True,
        ),
    ]
    angle_constraints: list[AngleConstraint] = [
        AngleConstraint(
            landmarks=[THUMB_CMC, THUMB_MCP, THUMB_IP],
            joint="thumb_proximal_flex",
            weight=0.5,
            optional=True,
        ),
        AngleConstraint(
            landmarks=[THUMB_MCP, THUMB_IP, THUMB_TIP],
            joint="thumb_distal_flex",
            weight=0.5,
            optional=True,
        ),
    ]
    distance_constraints: list[DistanceConstraint] = [
        DistanceConstraint(
            human=[THUMB_TIP, INDEX_TIP],
            robot=["thumb_tip", "index_tip"],
            robot_types=["site", "site"],
            weight=2000.0,
            scale=1.0,
            threshold=0.04,
            activation_type="linear",
            scale_mode="hand_scaled",
        ),
        DistanceConstraint(
            human=[THUMB_TIP, MIDDLE_TIP],
            robot=["thumb_tip", "middle_tip"],
            robot_types=["site", "site"],
            weight=1500.0,
            scale=1.0,
            threshold=0.04,
            activation_type="linear",
            scale_mode="hand_scaled",
        ),
        DistanceConstraint(
            human=[THUMB_TIP, RING_TIP],
            robot=["thumb_tip", "ring_tip"],
            robot_types=["site", "site"],
            weight=1000.0,
            scale=1.0,
            threshold=0.04,
            activation_type="linear",
            scale_mode="hand_scaled",
        ),
        DistanceConstraint(
            human=[THUMB_TIP, LITTLE_TIP],
            robot=["thumb_tip", "pinky_tip"],
            robot_types=["site", "site"],
            weight=800.0,
            scale=1.0,
            threshold=0.04,
            activation_type="linear",
            scale_mode="hand_scaled",
        ),
    ]
    frame_constraints = [
        FrameConstraint(
            name="thumb_cmc_frame",
            human_origin=THUMB_CMC,
            human_primary=THUMB_MCP,
            human_secondary=INDEX_MCP,
            robot_origin="thumb_base",
            robot_primary="thumb_mid",
            robot_secondary="index_base",
            primary_weight=2.0,
            secondary_weight=1.8,
            optional=True,
        )
    ]

    for finger_name, mcp, pip, dip, tip, pinch_weight in _FINGERS:
        vector_constraints.extend(
            [
                VectorConstraint(
                    human=[WRIST, mcp],
                    robot=["world", f"{finger_name}_base"],
                    robot_types=["body", "body"],
                    weight=0.8,
                    optional=True,
                ),
                VectorConstraint(
                    human=[mcp, pip],
                    robot=[f"{finger_name}_base", f"{finger_name}_mid"],
                    weight=1.0,
                    optional=True,
                ),
                VectorConstraint(
                    human=[pip, tip],
                    robot=[f"{finger_name}_mid", f"{finger_name}_tip"],
                    robot_types=["body", "site"],
                    weight=1.0,
                    loss_type="residual",
                    loss_scale=1.0,
                    optional=True,
                ),
                VectorConstraint(
                    human=[dip, tip],
                    robot=[f"{finger_name}_distal", f"{finger_name}_tip"],
                    robot_types=["body", "site"],
                    weight=0.8,
                    loss_type="residual",
                    loss_scale=1.0,
                    optional=True,
                ),
            ]
        )
        angle_constraints.extend(
            [
                AngleConstraint(
                    landmarks=[WRIST, mcp, pip],
                    joint=f"{finger_name}_base_flex",
                    weight=0.7,
                    scale=1.0,
                    optional=True,
                ),
                AngleConstraint(
                    landmarks=[mcp, pip, dip],
                    joint=f"{finger_name}_proximal_flex",
                    weight=1.2,
                    scale=1.4,
                    optional=True,
                ),
                AngleConstraint(
                    landmarks=[pip, dip, tip],
                    joint=f"{finger_name}_distal_flex",
                    weight=2.0,
                    scale=2.4,
                    optional=True,
                ),
            ]
        )
        distance_constraints.append(
            DistanceConstraint(
                human=[tip, mcp],
                robot=[f"{finger_name}_tip", f"{finger_name}_base"],
                robot_types=["site", "body"],
                weight=pinch_weight,
                scale=1.0,
                threshold=0.06,
                activation_type="linear",
                scale_mode="hand_scaled",
                optional=True,
            )
        )

    config.vector_constraints = vector_constraints
    config.frame_constraints = frame_constraints
    config.distance_constraints = distance_constraints
    config.angle_constraints = angle_constraints
