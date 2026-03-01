import os
import h5py
import argparse
import importlib
import numpy as np
import torch as th
import omnigibson as og
from omnigibson.macros import gm
from typing import Dict, List, Optional

from damagesim.omnigibson.damageable_env import (
    OGDamageableDataPlaybackWrapper,
)

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = False

# ── Task-config registry ────────────────────────────────────────────────

# Maps CLI task_name → module path under ``scripts.task_configs``
TASK_REGISTRY: Dict[str, str] = {
    "shelve_item": "scripts.task_configs.shelve_item",
    "add_firewood": "scripts.task_configs.add_firewood",
    "firewood": "scripts.task_configs.add_firewood",  # alias
    "pour_water": "scripts.task_configs.pour_water",
}

def load_task_config(task_name: str):
    """Import the task config module and return its ``TaskConfig``."""
    if task_name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(
            f"Unknown task '{task_name}'. Available tasks: {available}"
        )
    mod = importlib.import_module(TASK_REGISTRY[task_name])
    return mod.get_task_config()

def run_playback(activity_name, teleop_hdf5_path, new_playback_hdf5_path):
    """Create an OG env from HDF5 and replay demonstrations."""

    # gm.USE_GPU_DYNAMICS = task_cfg.use_gpu_dynamics
    # gm.ENABLE_TRANSITION_RULES = task_cfg.enable_transition_rules

    env = OGDamageableDataPlaybackWrapper.create_from_hdf5(
        input_path=teleop_hdf5_path,
        output_path=new_playback_hdf5_path,
        robot_obs_modalities=["proprio"],
        n_render_iterations=1,
        only_successes=False,
        activity_name=activity_name,
    )
    env.playback_dataset(record_data=True)
    env.save_data()

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

args = parse_args()

# loop over all hdf5 files in the current directory
# print(file)
activity_name = "shelve_item"
file_name = f"{activity_name}_unsafe.hdf5"
task_cfg = load_task_config(activity_name)
teleop_hdf5_path = f"tests/data/teleop_data/behavior1k/{file_name}"
original_playback_hdf5_path = f"tests/data/playback_data/behavior1k/{file_name}"
new_playback_hdf5_path = f"tests/data/tmp/behavior1k/{file_name}"

run_playback(activity_name, teleop_hdf5_path, new_playback_hdf5_path)

# Compare healths between original playback hdf5 and new playback hdf5
f_original_playback = h5py.File(original_playback_hdf5_path, "r")
f_new_playback = h5py.File(new_playback_hdf5_path, "r")
for demo_key in sorted(f_original_playback["data"].keys()):
    demo_idx = int(demo_key.split("_")[-1])
    print(f"Comparing episode {demo_idx} …")

    # Check if healths are the same
    health_original_playback = f_original_playback[f"data/{demo_key}/obs/health"][-1]
    health_new_playback = f_new_playback[f"data/{demo_key}/obs/health"][-1]
    if not np.allclose(health_original_playback, health_new_playback, atol=5.0):
        print(f"Healths are not the same for episode {demo_idx}")
        print(f"Original health: {health_original_playback}")
        print(f"New health: {health_new_playback}")
        print(f"Difference: {np.abs(health_original_playback - health_new_playback)}")
        raise ValueError(f"Healths are not the same for episode {demo_idx}")

f_original_playback.close()
f_new_playback.close()

print("All tests passed!")
og.shutdown()