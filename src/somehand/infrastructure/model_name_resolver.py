"""Resolve semantic config names to concrete MuJoCo object names."""

from __future__ import annotations

import mujoco
import re


_SIDE_PREFIXES = ("lh", "rh", "left", "right", "l", "r")
_PREFERRED_SIDE_PREFIXES: dict[str, tuple[str, ...]] = {
    "left": ("lh", "left", "l"),
    "right": ("rh", "right", "r"),
}
_SEMANTIC_ALIASES: dict[str, tuple[str, ...]] = {
    "thumb_dip": ("thumb_ip",),
    "thumb_ip": ("thumb_dip",),
    "middle_dip": ("middle_distal",),
    "middle_distal": ("middle_dip",),
    "middle_base": ("middle_metacarpals", "middle_proximal", "middle_proximal_link", "f_link3_1", "finger3_link1"),
}
_FINGER_NUMBERS = {
    "thumb": 1,
    "index": 2,
    "middle": 3,
    "ring": 4,
    "pinky": 5,
    "little": 5,
}
_GENERIC_SEMANTIC_PATTERN = re.compile(r"^(thumb|index|middle|ring|pinky|little)_(.+)$")


def _finger_role_candidates(finger: str, role: str) -> tuple[str, ...]:
    finger = "pinky" if finger == "little" else finger
    number = _FINGER_NUMBERS[finger]
    if finger == "thumb":
        body_roles = {
            "base": ("thumb_metacarpals", "thumb_metacarpal_link", "thumb_metacarpals_base2", "f_link1_2", "finger1_link2"),
            "mid": ("thumb_proximal", "thumb_proximal_link", "thumb_distal", "f_link1_3", "finger1_link3"),
            "distal": ("thumb_distal", "thumb_distal_link", "f_link1_4", "finger1_link4"),
            "tip": ("thumb_distal_tip", "thumb_distal_link_tip", "f_link1_4_tip", "finger1_link4_tip"),
        }
        joint_roles = {
            "proximal_flex": ("thumb_mcp", "thumb_proximal_joint", "f_joint1_3", "finger1_joint3"),
            "distal_flex": ("thumb_dip", "thumb_ip", "thumb_distal_joint", "f_joint1_4", "finger1_joint4"),
        }
        return body_roles.get(role, joint_roles.get(role, ()))
    body_roles = {
        "base": (
            f"{finger}_metacarpals",
            f"{finger}_proximal",
            f"{finger}_proximal_link",
            f"f_link{number}_1",
            f"finger{number}_link1",
        ),
        "mid": (
            f"{finger}_middle",
            f"{finger}_distal",
            f"{finger}_distal_link",
            f"f_link{number}_2",
            f"finger{number}_link2",
        ),
        "distal": (
            f"{finger}_distal",
            f"{finger}_distal_link",
            f"f_link{number}_3",
            f"finger{number}_link3",
        ),
        "tip": (
            f"{finger}_distal_tip",
            f"{finger}_middle_tip",
            f"{finger}_distal_link_tip",
            f"f_link{number}_4_tip",
            f"finger{number}_link4_tip",
        ),
    }
    joint_roles = {
        "base_flex": (
            f"{finger}_mcp_pitch",
            f"{finger}_base_pitch",
            f"{finger}_proximal_joint",
            f"f_joint{number}_1",
            f"finger{number}_joint1",
        ),
        "proximal_flex": (
            f"{finger}_pip",
            f"{finger}_middle_joint",
            f"f_joint{number}_2",
            f"finger{number}_joint2",
        ),
        "distal_flex": (
            f"{finger}_dip",
            f"{finger}_distal_joint",
            f"f_joint{number}_3",
            f"finger{number}_joint3",
        ),
    }
    return body_roles.get(role, joint_roles.get(role, ()))


def _strip_side_prefix(name: str) -> str:
    for prefix in _SIDE_PREFIXES:
        token = f"{prefix}_"
        if name.startswith(token):
            return name[len(token):]
    return name


class ModelNameResolver:
    """Maps semantic body/site/joint names onto a specific MuJoCo model."""

    def __init__(self, model: mujoco.MjModel, *, hand_side: str):
        self.model = model
        self.hand_side = hand_side
        self._preferred_prefixes = _PREFERRED_SIDE_PREFIXES[hand_side]
        self._body_names = self._collect_names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
        self._site_names = self._collect_names(mujoco.mjtObj.mjOBJ_SITE, model.nsite)
        self._joint_names = self._collect_names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt)

    def _collect_names(self, obj_type, count: int) -> set[str]:
        names: set[str] = set()
        for index in range(count):
            name = mujoco.mj_id2name(self.model, obj_type, index)
            if name:
                names.add(name)
        return names

    def _candidate_names(self, semantic_name: str) -> list[str]:
        semantic = _strip_side_prefix(semantic_name)
        semantic_variants = [semantic_name, semantic]
        semantic_variants.extend(_SEMANTIC_ALIASES.get(semantic, ()))
        generic_match = _GENERIC_SEMANTIC_PATTERN.match(semantic)
        if generic_match:
            semantic_variants.extend(_finger_role_candidates(generic_match.group(1), generic_match.group(2)))

        candidates: list[str] = []
        for variant in semantic_variants:
            stripped_variant = _strip_side_prefix(variant)
            prefixed_candidates = tuple(
                f"{prefix}_{stripped_variant}" for prefix in self._preferred_prefixes
            )
            for candidate in (variant, *prefixed_candidates, stripped_variant):
                if candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    def resolve(self, semantic_name: str, *, obj_type, role: str) -> str:
        names = {
            mujoco.mjtObj.mjOBJ_BODY: self._body_names,
            mujoco.mjtObj.mjOBJ_SITE: self._site_names,
            mujoco.mjtObj.mjOBJ_JOINT: self._joint_names,
        }[obj_type]
        for candidate in self._candidate_names(semantic_name):
            if candidate in names:
                return candidate
        raise ValueError(
            f"{role} semantic name '{semantic_name}' could not be resolved for {self.hand_side} hand model"
        )

    def resolve_optional(self, semantic_name: str, *, obj_type, role: str) -> str | None:
        try:
            return self.resolve(semantic_name, obj_type=obj_type, role=role)
        except ValueError:
            return None
