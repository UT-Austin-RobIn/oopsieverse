#!/usr/bin/env python3
"""
Keyboard teleoperation for OmniGibson damage-tracking tasks with live health visualization.

Loads a task from oopsiebench.envs.behavior1k.task_configs, creates OGDamageableEnvironment,
runs keyboard teleop (OmniGibson built-in, no telemoma required), and shows live health bars until Escape.

Usage (from project root):
    python -m oopsiebench.envs.behavior1k.teleop --task pour_water
    python -m oopsiebench.envs.behavior1k.teleop --task shelve_item
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Ensure project root is on path so task_configs can import scripts.task_configs.base
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Reduce OmniGibson physx log noise when running before og is imported
import os
os.environ.setdefault("CARB_LOG_CHANNELS", "omni.physx.plugin=off")


def _load_task_config(task_name: str):
    """Load TaskConfig from oopsiebench.envs.behavior1k.task_configs.<task_name>."""
    # Task config modules import "from scripts.task_configs.base import TaskConfig".
    # Shim that to the local base so they resolve without changing those files.
    if "scripts.task_configs.base" not in sys.modules:
        import oopsiebench.envs.behavior1k.task_configs.base as _local_base
        sys.modules["scripts.task_configs.base"] = _local_base
    mod_path = f"oopsiebench.envs.behavior1k.task_configs.{task_name}"
    try:
        mod = importlib.import_module(mod_path)
    except ModuleNotFoundError:
        raise ValueError(
            f"Unknown task '{task_name}'. "
            f"Add a module under oopsiebench/envs/behavior1k/task_configs/ (e.g. pour_water.py)."
        )
    return mod.get_task_config()


def _build_external_sensors_config(task_cfg, robot_name: str, robot_type: str, image_height: int = 256, image_width: int = 256):
    """Build env.external_sensors list from task_cfg.external_camera_configs."""
    import torch as th
    sensors = []
    for name, cam_cfg in task_cfg.external_camera_configs.items():
        idx = name.split("_")[-1]
        prim_path = (
            f"/controllable__damageable{robot_type.lower()}"
            f"__{robot_name}/base_link/external_sensor{idx}"
        )
        sensors.append({
            "sensor_type": "VisionSensor",
            "name": f"external_sensor{idx}",
            "relative_prim_path": prim_path,
            "modalities": ["rgb", "seg_instance"],
            "sensor_kwargs": {
                "image_height": image_height,
                "image_width": image_width,
                "horizontal_aperture": cam_cfg.get("horizontal_aperture", 15.0),
            },
            "position": th.tensor(cam_cfg["position"], dtype=th.float32),
            "orientation": th.tensor(cam_cfg["orientation"], dtype=th.float32),
            "pose_frame": "world",
        })
    return sensors


def _build_env_config(task_cfg):
    """Build full OG config dict from TaskConfig (same pattern as verify_damageable_env)."""
    import omnigibson as og
    from omnigibson.utils.config_utils import parse_config

    base = parse_config(f"{og.example_config_path}/default_cfg.yaml")
    base["scene"] = {
        "type": "InteractiveTraversableScene",
        **task_cfg.scene_config,
    }
    base["robots"] = [task_cfg.robot_config]
    base["objects"] = list(task_cfg.task_objects.values())
    # DummyTask only; activity_name is for damage tracking (OGDamageableEnvironment.task_name).
    base["task"] = {
        "type": "DummyTask",
        "activity_name": task_cfg.task_name,
    }
    base["env"] = base.get("env", {})
    base["env"]["external_sensors"] = _build_external_sensors_config(
        task_cfg, task_cfg.robot_name, task_cfg.robot_type,
    )
    return base


def main():
    parser = argparse.ArgumentParser(
        description="Keyboard teleop with live health visualization. Press Escape to exit.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (e.g. pour_water, shelve_item, add_firewood).",
    )
    args = parser.parse_args()

    task_cfg = _load_task_config(args.task)

    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm

    gm.USE_GPU_DYNAMICS = task_cfg.use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = task_cfg.enable_transition_rules

    from damagesim.omnigibson.damageable_env import OGDamageableEnvironment

    cfg = _build_env_config(task_cfg)
    env = OGDamageableEnvironment(cfg)
    env.reset()

    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor(task_cfg.viewer_camera_pos, dtype=th.float32),
        orientation=th.tensor(task_cfg.viewer_camera_orn, dtype=th.float32),
    )
    for _ in range(10):
        og.sim.step()

    env.enable_health_visualization()

    from omnigibson.utils.ui_utils import KeyboardRobotController
    import omnigibson.lazy as lazy

    robot = env.robots[0]
    keyboard_controller = KeyboardRobotController(robot)

    # Override Escape to cleanup health viz then shutdown (controller default only calls og.shutdown)
    def on_escape():
        if hasattr(env, "disable_health_visualization"):
            env.disable_health_visualization()
        og.shutdown()

    keyboard_controller.register_custom_keymapping(
        lazy.carb.input.KeyboardInput.ESCAPE,
        "Exit and shutdown",
        on_escape,
    )

    step_count = 0
    while True:
        action, _ = keyboard_controller.get_teleop_action()
        obs, reward, terminated, truncated, info = env.step(
            action,
            n_render_iterations=1,
            episode_step_count=step_count,
            init_skip_steps=0,
        )
        env.update_health_visualization(obs)
        step_count += 1
        og.sim.render()


if __name__ == "__main__":
    main()
