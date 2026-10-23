"""Robot profile schema and validation for the SCARA platform menu."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    label: str
    description: str
    mode: str
    command: list[str] | None = None
    target: str | None = None
    working_directory: str | None = None


@dataclass(frozen=True)
class RobotProfile:
    profile_id: str
    display_name: str
    dof: int
    joint_names: list[str]
    joint_types: list[str]
    state_dim: int
    action_dim: int
    end_effector: str
    controller: str
    notes: str
    actions: list[ActionSpec]
    raw: dict[str, Any]


def load_profile(path: Path) -> RobotProfile:
    data = json.loads(path.read_text(encoding="utf-8"))
    actions = [
        ActionSpec(
            action_id=item["id"],
            label=item["label"],
            description=item.get("description", ""),
            mode=item.get("mode", "new_console"),
            command=item.get("command"),
            target=item.get("target"),
            working_directory=item.get("working_directory"),
        )
        for item in data.get("actions", [])
    ]
    return RobotProfile(
        profile_id=data["profile_id"],
        display_name=data["display_name"],
        dof=int(data["dof"]),
        joint_names=list(data.get("joint_names", [])),
        joint_types=list(data.get("joint_types", [])),
        state_dim=int(data.get("state_dim", 0)),
        action_dim=int(data.get("action_dim", 0)),
        end_effector=data.get("end_effector", ""),
        controller=data.get("controller", ""),
        notes=data.get("notes", ""),
        actions=actions,
        raw=data,
    )


def load_profiles(directory: Path) -> dict[str, RobotProfile]:
    profiles = {}
    for path in sorted(directory.glob("*.json")):
        profile = load_profile(path)
        profiles[profile.profile_id] = profile
    return profiles


def expand_path(value: str, root: Path, python_exe: str) -> str:
    return value.format(root=str(root), python=python_exe)


def validate_profile(
    profile: RobotProfile,
    root: Path,
    python_exe: str,
) -> list[str]:
    errors: list[str] = []
    if profile.dof <= 0:
        errors.append("dof must be greater than zero")
    if len(profile.joint_names) != profile.dof:
        errors.append(
            f"joint_names has {len(profile.joint_names)} entries, "
            f"expected {profile.dof}"
        )
    if profile.joint_types and len(profile.joint_types) != profile.dof:
        errors.append(
            f"joint_types has {len(profile.joint_types)} entries, "
            f"expected {profile.dof}"
        )
    seen = set()
    for action in profile.actions:
        if action.action_id in seen:
            errors.append(f"duplicate action id: {action.action_id}")
        seen.add(action.action_id)
        if action.mode == "open_file":
            if not action.target:
                errors.append(f"{action.action_id}: missing target")
                continue
            target = Path(expand_path(action.target, root, python_exe))
            if not target.exists():
                errors.append(
                    f"{action.action_id}: missing file {target}"
                )
        elif action.mode == "new_console":
            if not action.command:
                errors.append(f"{action.action_id}: missing command")
                continue
            expanded = [
                expand_path(part, root, python_exe)
                for part in action.command
            ]
            executable = Path(expanded[0])
            if executable.is_absolute() and not executable.exists():
                errors.append(
                    f"{action.action_id}: missing executable {executable}"
                )
        else:
            errors.append(
                f"{action.action_id}: unsupported mode {action.mode}"
            )
    return errors
