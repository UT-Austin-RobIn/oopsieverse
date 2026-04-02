"""
Tests for RoboCasa damage tracking across registered environments.

Verifies:
1. Objects and fixtures are DamageableMixin instances; only those configured
   in DAMAGEABLE_OBJECTS for each task have track_damage=True.
2. Replaying recorded teleop HDF5 data produces final health values matching
   recorded damage_data/* values without hardcoded object keys.
"""

from pathlib import Path

import h5py
import numpy as np
import json
import pytest

from damagesim.core.damageable_mixin import DamageableMixin
from damagesim.robosuite.params import DAMAGEABLE_OBJECTS
from robosuite.controllers import load_composite_controller_config

from oopsiebench.envs.registry import EnvironmentRegistry


TELEOP_DATA_DIR = Path("oopsiebench/test_data/robocasa/")
TRIAL_FILES = sorted(TELEOP_DATA_DIR.glob("*.hdf5"))
ROBOCASA_ENVS = EnvironmentRegistry.list_envs()


def _env_name_from_path(path):
    """Extract environment name from HDF5 filename (e.g. 'open_single_door_safe.hdf5' -> 'open_single_door')."""
    stem = path.stem
    for suffix in ("_safe", "_unsafe"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _sync_damage_evaluator_velocities(env):
    """Sync evaluators' prev_linear_velocities with current sim state to prevent
    spurious impact detection when restoring states during playback."""
    for obj in env.get_damageable_objects():
        for evaluator in obj.damage_evaluators:
            if hasattr(evaluator, 'prev_linear_velocities') and hasattr(evaluator, '_get_part_linear_velocity'):
                for part_name in evaluator._get_damageable_part_names():
                    evaluator.prev_linear_velocities[part_name] = evaluator._get_part_linear_velocity(part_name)


def _create_env(env_name):
    env_config = EnvironmentRegistry.get(env_name)
    return env_config.damageable_class(
        robots=env_config.robot,
        controller_configs=load_composite_controller_config(robot=env_config.robot),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_segmentation=False,
        control_freq=env_config.control_freq,
    )


def _restore_env_for_playback(env, env_name, ep_meta, model_xml, expected_state_size, expected_health_cols, source_name):
    """Restore the exact recorded scene for playback.

    Sets ep_meta before the first reset so _load_model() produces the correct
    Python objects, then loads the exact MuJoCo model from the recording via
    reset_from_xml_string (which re-initialises damage tracking internally).
    """
    assert ep_meta, (
        f"[{env_name}] Cannot replay {source_name}: missing ep_meta. "
        f"Re-record this fixture with the current teleop script."
    )
    env.set_ep_meta(ep_meta)
    env.reset()

    if model_xml is not None:
        env.reset_from_xml_string(model_xml)
        env.sim.reset()

    obs, _ = env.get_observations()
    actual_state_size = int(env.sim.get_state().flatten().shape[0])
    actual_health_cols = obs["health"].shape[0]

    if actual_state_size != expected_state_size:
        env.close()
        raise AssertionError(
            f"[{env_name}] Cannot replay {source_name}: state vector size mismatch "
            f"(expected {expected_state_size}, got {actual_state_size})."
        )
    if actual_health_cols != expected_health_cols:
        env.close()
        raise AssertionError(
            f"[{env_name}] Cannot replay {source_name}: health column mismatch "
            f"(expected {expected_health_cols}, got {actual_health_cols})."
        )
    return env


def _expected_should_track(env, obj_name, obj_category, is_robot, tracked_names, tracked_categories):
    if not tracked_names and not tracked_categories:
        return True
    if obj_category in tracked_categories:
        return True
    if obj_name in tracked_names:
        return True
    if obj_name:
        name_lower = obj_name.lower()
        for cat in tracked_categories:
            if cat and (cat.lower() in name_lower or name_lower in cat.lower()):
                return True
    if "agent" in tracked_categories and is_robot:
        return True
    return False


@pytest.mark.parametrize("env_name", ROBOCASA_ENVS, ids=ROBOCASA_ENVS)
def test_damageable_objects_initialized(env_name):
    """All objects/fixtures are DamageableMixin; only those in DAMAGEABLE_OBJECTS track damage."""
    env = _create_env(env_name)
    env.reset()

    default_cfg = DAMAGEABLE_OBJECTS.get("default", {})
    task_cfg = DAMAGEABLE_OBJECTS.get(env.task_name, {})
    tracked_names = set(default_cfg.get("names", []) + task_cfg.get("names", []))
    tracked_categories = set(default_cfg.get("categories", []) + task_cfg.get("categories", []))

    try:
        for obj in env._get_all_objects():
            assert isinstance(obj, DamageableMixin), (
                f"Object '{getattr(obj, 'name', obj)}' is not a DamageableMixin instance"
            )

            obj_name = getattr(obj, "name", "") or ""
            obj_category = getattr(obj, "category", "") or ""
            is_robot = env._is_robot(obj)

            should_track = _expected_should_track(
                env=env,
                obj_name=obj_name,
                obj_category=obj_category,
                is_robot=is_robot,
                tracked_names=tracked_names,
                tracked_categories=tracked_categories,
            )

            assert obj.track_damage == should_track, (
                f"[{env_name}] Object '{obj_name}': expected track_damage={should_track}, "
                f"got {obj.track_damage}"
            )
    finally:
        env.close()


@pytest.mark.parametrize(
    "hdf5_path",
    TRIAL_FILES,
    ids=[p.name for p in TRIAL_FILES],
)
def test_playback_damage_values_match(hdf5_path):
    """Replay demo_0 from each fixture and verify final obs health matches recorded values."""
    env_name = _env_name_from_path(hdf5_path)
    playback_env = _create_env(env_name)

    with h5py.File(hdf5_path, "r") as f:
        demo = f["data/demo_0"]
        states = demo["states"][:]
        actions = demo["actions"][1:]
        model_xml = demo.attrs.get("model_file")
        ep_meta = json.loads(demo.attrs.get("ep_meta", "{}"))
        all_health = demo["obs/health"][:]  # shape (T, N)

    num_actions = len(actions)
    assert len(states) == num_actions + 1, (
        f"Expected num_states == num_actions + 1, got {len(states)} states and {num_actions} actions"
    )
    expected_state_size = int(states[0].size)

    playback_env = _restore_env_for_playback(
        env=playback_env,
        env_name=env_name,
        ep_meta=ep_meta,
        model_xml=model_xml,
        expected_state_size=expected_state_size,
        expected_health_cols=all_health.shape[1],
        source_name=hdf5_path.name,
    )

    try:
        playback_env.sim.set_state_from_flattened(states[0])
        playback_env.sim.forward()
        _sync_damage_evaluator_velocities(playback_env)

        for i in range(num_actions):
            playback_env.sim.set_state_from_flattened(states[i])
            playback_env.sim.forward()
            _sync_damage_evaluator_velocities(playback_env)
            obs, _, _, info = playback_env.step(actions[i])

        observed_health = np.sort(obs["health"].astype(np.float64))
        expected_health = np.sort(all_health[-1].astype(np.float64))
        assert observed_health.shape == expected_health.shape, (
            f"[{env_name}] Health vector shape mismatch in {hdf5_path.name}: "
            f"expected {expected_health.shape[0]} values, got {observed_health.shape[0]}."
        )
        np.testing.assert_allclose(
            observed_health,
            expected_health,
            atol=5.0,
            err_msg=(
                f"[{env_name}] Health mismatch in {hdf5_path.name}: "
                f"max |diff|={float(np.abs(observed_health - expected_health).max()):.3f}"
            ),
        )
    finally:
        playback_env.close()
