"""Unified, profile-driven launcher for the SCARA simulation platform."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from robot_profile import (
    ActionSpec,
    RobotProfile,
    expand_path,
    load_profiles,
    validate_profile,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "platform" / "profiles"
ACTIVE_PROFILE = ROOT / "platform" / "active_profile.txt"
PYTHON_EXE = sys.executable


def enable_terminal() -> None:
    if os.name == "nt":
        os.system("")
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def active_profile_id() -> str:
    if ACTIVE_PROFILE.exists():
        value = ACTIVE_PROFILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    return "scara_teach_4rprr"


def load_active() -> tuple[dict[str, RobotProfile], RobotProfile]:
    profiles = load_profiles(PROFILE_DIR)
    if not profiles:
        raise RuntimeError(f"No profiles found in {PROFILE_DIR}")
    current = active_profile_id()
    if current not in profiles:
        current = next(iter(profiles))
    return profiles, profiles[current]


def separator(char: str = "=", width: int = 78) -> None:
    print(char * width)


def print_header(profile: RobotProfile) -> None:
    separator()
    print(" SCARA 仿真平台统一菜单")
    print(f" 当前配置: {profile.display_name}  [{profile.profile_id}]")
    print(
        f" 轴数: {profile.dof}   状态维: {profile.state_dim}   "
        f"动作维: {profile.action_dim}"
    )
    print(
        " 关节: "
        + ", ".join(
            f"{name}({kind})"
            for name, kind in zip(
                profile.joint_names, profile.joint_types
            )
        )
    )
    if profile.notes:
        print(f" 参数摘要: {profile.notes}")
    separator()


def print_profile_details(profile: RobotProfile) -> None:
    separator("-")
    print(f"profile_id   : {profile.profile_id}")
    print(f"display_name : {profile.display_name}")
    print(f"dof          : {profile.dof}")
    print(f"joint_names  : {profile.joint_names}")
    print(f"joint_types  : {profile.joint_types}")
    print(f"state_dim    : {profile.state_dim}")
    print(f"action_dim   : {profile.action_dim}")
    print(f"end_effector : {profile.end_effector}")
    print(f"controller   : {profile.controller}")
    print(f"notes        : {profile.notes}")
    separator("-")


def print_actions(profile: RobotProfile) -> None:
    for index, action in enumerate(profile.actions, start=1):
        print(f" {index:2d}. {action.label}")
        if action.description:
            print(f"     {action.description}")
    print(" P. 切换机器人轴配置")
    print(" I. 查看当前配置")
    print(" V. 校验当前配置")
    print(" R. 重新加载")
    print(" 0. 退出")
    separator("-")


def switch_profile(profiles: dict[str, RobotProfile]) -> RobotProfile:
    items = list(profiles.values())
    print("\n可用机器人配置：")
    for index, profile in enumerate(items, start=1):
        tag = " *" if profile.profile_id == active_profile_id() else ""
        print(
            f" {index}. {profile.display_name} "
            f"[{profile.profile_id}] DOF={profile.dof}{tag}"
        )
    raw = input("选择配置编号（回车取消）: ").strip()
    if not raw:
        return load_active()[1]
    try:
        selected = items[int(raw) - 1]
    except (ValueError, IndexError):
        print("无效编号。")
        return load_active()[1]
    ACTIVE_PROFILE.write_text(
        selected.profile_id + "\n", encoding="utf-8"
    )
    print(f"已切换: {selected.display_name}")
    return selected


def execute_action(profile: RobotProfile, action: ActionSpec) -> None:
    working_directory = (
        Path(
            expand_path(
                action.working_directory, ROOT, PYTHON_EXE
            )
        )
        if action.working_directory
        else ROOT
    )
    if action.mode == "open_file":
        target = Path(
            expand_path(action.target or "", ROOT, PYTHON_EXE)
        )
        if not target.exists():
            raise FileNotFoundError(target)
        os.startfile(str(target))
        print(f"已打开: {target}")
        return
    if action.mode != "new_console":
        raise ValueError(f"Unsupported action mode: {action.mode}")
    command = [
        expand_path(part, ROOT, PYTHON_EXE)
        for part in (action.command or [])
    ]
    if not command:
        raise ValueError(f"Action {action.action_id} has no command")
    executable = Path(command[0])
    if executable.is_absolute() and not executable.exists():
        raise FileNotFoundError(executable)
    if command[0].lower().endswith("cmd.exe"):
        process_command = command
    else:
        process_command = [
            "C:\\Windows\\System32\\cmd.exe",
            "/k",
            subprocess.list2cmdline(command),
        ]
    subprocess.Popen(
        process_command,
        cwd=str(working_directory),
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        env={
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "SCARA_PROFILE": profile.profile_id,
        },
    )
    print(f"已启动: {action.label}")


def validate_all(profiles: dict[str, RobotProfile]) -> int:
    failed = 0
    for profile in profiles.values():
        errors = validate_profile(profile, ROOT, PYTHON_EXE)
        status = "PASS" if not errors else "FAIL"
        print(f"[{status}] {profile.profile_id} ({profile.dof} DOF)")
        for error in errors:
            print(f"  - {error}")
        failed += bool(errors)
    return 1 if failed else 0


def interactive(profile: RobotProfile) -> int:
    while True:
        profiles, profile = load_active()
        print("\n")
        print_header(profile)
        print_actions(profile)
        choice = input("选择: ").strip().lower()
        if choice == "0":
            return 0
        if choice == "p":
            switch_profile(profiles)
            continue
        if choice == "i":
            print_profile_details(profile)
            input("按 Enter 返回...")
            continue
        if choice == "v":
            errors = validate_profile(profile, ROOT, PYTHON_EXE)
            if errors:
                print("校验失败：")
                for error in errors:
                    print(f" - {error}")
            else:
                print("校验通过：所有动作入口均存在。")
            input("按 Enter 返回...")
            continue
        if choice == "r":
            print("已重新加载配置。")
            continue
        try:
            action = profile.actions[int(choice) - 1]
        except (ValueError, IndexError):
            print("无效选项。")
            continue
        try:
            execute_action(profile, action)
        except Exception as exc:
            print(f"启动失败: {exc}")
            input("按 Enter 返回...")


def main() -> int:
    enable_terminal()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run")
    parser.add_argument("--dry-run")
    parser.add_argument("--profile")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--all-profiles", action="store_true")
    parser.add_argument("--info", action="store_true")
    args = parser.parse_args()
    profiles, profile = load_active()
    if args.profile:
        if args.profile not in profiles:
            print(f"未知配置: {args.profile}")
            return 2
        ACTIVE_PROFILE.write_text(
            args.profile + "\n", encoding="utf-8"
        )
        profile = profiles[args.profile]
    if args.all_profiles:
        return validate_all(profiles)
    if args.validate:
        errors = validate_profile(profile, ROOT, PYTHON_EXE)
        if errors:
            for error in errors:
                print(f"- {error}")
            return 1
        print(f"Profile {profile.profile_id}: PASS")
        return 0
    if args.info:
        print_profile_details(profile)
        return 0
    if args.list:
        for action in profile.actions:
            print(f"{action.action_id}\t{action.label}")
        return 0
    action_id = args.run or args.dry_run
    if action_id:
        actions = {item.action_id: item for item in profile.actions}
        if action_id not in actions:
            print(f"未知动作: {action_id}")
            return 2
        action = actions[action_id]
        if args.dry_run:
            print(f"{action.action_id}: {action.mode}")
            if action.command:
                print(
                    " ".join(
                        expand_path(part, ROOT, PYTHON_EXE)
                        for part in action.command
                    )
                )
            if action.target:
                print(
                    expand_path(action.target, ROOT, PYTHON_EXE)
                )
            return 0
        try:
            execute_action(profile, action)
            return 0
        except Exception as exc:
            print(f"启动失败: {exc}")
            return 1
    return interactive(profile)


if __name__ == "__main__":
    raise SystemExit(main())
