"""
Canonical playback HDF5 observation schema shared by Robocasa and B1K writers.

Whitelisted ``obs`` keys written to playback HDF5:
  - proprio          (T+1, 23): sin(7)+cos(7)+eef_pos(3)+eef_quat(4)+gripper_qpos(2)
  - eef_pos           (T+1, 3)
  - eef_quat          (T+1, 4)  — always stored as xyzw
  - health            (T+1, ...)
  - cam/{name}/rgb
  - cam/{name}/seg

Per-demo length contract (``T`` = number of actions):
  - State-aligned (length T+1): ``obs/*``, ``info/*`` (``obs_info``, ``damage_info``, …),
    ``task_completion``
  - Transition-aligned (length T): ``action``, ``reward``, and when present
    ``terminated`` / ``truncated``

Index ``t`` means: ``obs[t]`` / ``info[t]`` / ``task_completion[t]`` describe the
same state; ``action[t]`` is taken from that state and leads toward ``obs[t+1]``.

B1K OmniGibson quaternions are converted from wxyz → xyzw at write time.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

try:
    import torch as th
except ImportError:  # pragma: no cover
    th = None

PROPRIO_DIM = 23
PROPRIO_EEF_QUAT_SLICE = slice(17, 21)

_WHITELIST_EXACT = frozenset({"proprio", "eef_pos", "eef_quat", "health"})


def playback_proprio_keys(arm: str = "0") -> list[str]:
    """OG proprio modalities without grasp (23-D)."""
    return [
        f"arm_{arm}_qpos_sin",
        f"arm_{arm}_qpos_cos",
        f"eef_{arm}_pos",
        f"eef_{arm}_quat",
        f"gripper_{arm}_qpos",
    ]


def wxyz_to_xyzw(q: Any) -> Any:
    """Convert quaternion from (w, x, y, z) to (x, y, z, w)."""
    if th is not None and isinstance(q, th.Tensor):
        return th.cat([q[..., 1:], q[..., :1]], dim=-1)
    q = np.asarray(q)
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def _as_float32_array(v: Any) -> Any:
    if th is not None and isinstance(v, th.Tensor):
        return v
    if isinstance(v, np.ndarray):
        if v.dtype == np.float64:
            return v.astype(np.float32)
        return v
    return v


def _maybe_drop_alpha(rgb: Any) -> Any:
    """Keep RGB only when a trailing alpha channel is present."""
    if th is not None and isinstance(rgb, th.Tensor):
        if rgb.ndim >= 3 and rgb.shape[-1] == 4:
            return rgb[..., :3]
        return rgb
    rgb = np.asarray(rgb)
    if rgb.ndim >= 3 and rgb.shape[-1] == 4:
        return rgb[..., :3]
    return rgb


def is_cam_obs_key(key: str) -> bool:
    parts = key.split("/")
    return len(parts) == 3 and parts[0] == "cam" and parts[2] in ("rgb", "seg")


def is_image_obs_key(key: str) -> bool:
    """True for image-like obs keys (canonical or legacy)."""
    if is_cam_obs_key(key):
        return True
    return key.endswith(
        ("rgb", "depth", "seg", "seg_instance", "seg_semantic", "_image")
    )


def filter_playback_obs(obs: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only the canonical playback obs whitelist."""
    return {
        k: v
        for k, v in obs.items()
        if k in _WHITELIST_EXACT or is_cam_obs_key(k)
    }


def _canonicalize_proprio_vector(proprio: Any, *, convert_quat_wxyz_to_xyzw: bool) -> Any:
    """Ensure 23-D proprio and optionally convert packed eef quat wxyz→xyzw."""
    if th is not None and isinstance(proprio, th.Tensor):
        p = proprio.flatten().float()
        if p.numel() > PROPRIO_DIM:
            p = p[:PROPRIO_DIM]
        elif p.numel() < PROPRIO_DIM:
            raise ValueError(f"proprio has {p.numel()} dims; expected >= {PROPRIO_DIM}")
        if convert_quat_wxyz_to_xyzw:
            q = p[PROPRIO_EEF_QUAT_SLICE]
            p = p.clone()
            p[PROPRIO_EEF_QUAT_SLICE] = wxyz_to_xyzw(q)
        return p

    p = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if p.size > PROPRIO_DIM:
        p = p[:PROPRIO_DIM].copy()
    elif p.size < PROPRIO_DIM:
        raise ValueError(f"proprio has {p.size} dims; expected >= {PROPRIO_DIM}")
    else:
        p = p.copy()
    if convert_quat_wxyz_to_xyzw:
        p[PROPRIO_EEF_QUAT_SLICE] = wxyz_to_xyzw(p[PROPRIO_EEF_QUAT_SLICE])
    return p


def b1k_camera_name_from_key(key: str) -> str:
    """
    Map OG flattened camera keys to a short camera name.

    Examples:
      external::external_sensor0::rgb -> external_sensor0
      franka0::franka0:eef_link:Camera:0::seg_instance -> eef
    """
    parts = key.split("::")
    if len(parts) < 2:
        return key
    sensor = parts[-2] if len(parts) >= 3 else parts[0]
    if "eef_link" in sensor or ":Camera:" in sensor:
        return "eef"
    return sensor


def canonicalize_robocasa_obs(obs: Mapping[str, Any]) -> Dict[str, Any]:
    """Rename Robocasa/robosuite obs keys into the canonical playback schema."""
    out: Dict[str, Any] = {}

    if "proprio" in obs:
        out["proprio"] = _canonicalize_proprio_vector(
            obs["proprio"], convert_quat_wxyz_to_xyzw=False
        )
    elif "robot0_proprio" in obs:
        out["proprio"] = _canonicalize_proprio_vector(
            obs["robot0_proprio"], convert_quat_wxyz_to_xyzw=False
        )

    if "eef_pos" in obs:
        out["eef_pos"] = _as_float32_array(obs["eef_pos"])
    elif "robot0_base_to_eef_pos" in obs:
        out["eef_pos"] = _as_float32_array(obs["robot0_base_to_eef_pos"])
    elif "robot0_eef_pos" in obs:
        out["eef_pos"] = _as_float32_array(obs["robot0_eef_pos"])

    if "eef_quat" in obs:
        out["eef_quat"] = _as_float32_array(obs["eef_quat"])
    elif "robot0_base_to_eef_quat" in obs:
        out["eef_quat"] = _as_float32_array(obs["robot0_base_to_eef_quat"])
    elif "robot0_eef_quat" in obs:
        out["eef_quat"] = _as_float32_array(obs["robot0_eef_quat"])

    if "health" in obs:
        out["health"] = obs["health"]

    for key, val in obs.items():
        if key.endswith("_image"):
            name = key[: -len("_image")]
            out[f"cam/{name}/rgb"] = _maybe_drop_alpha(val)
        elif key.endswith("_segmentation_class"):
            name = key[: -len("_segmentation_class")]
            out[f"cam/{name}/seg"] = val
        elif key.endswith("_segmentation_instance"):
            name = key[: -len("_segmentation_instance")]
            out[f"cam/{name}/seg"] = val

    return filter_playback_obs(out)


def canonicalize_b1k_obs(obs: Mapping[str, Any]) -> Dict[str, Any]:
    """Rename OmniGibson flattened obs keys into the canonical playback schema."""
    out: Dict[str, Any] = {}

    if "eef_pos" in obs:
        out["eef_pos"] = _as_float32_array(obs["eef_pos"])

    # OG relative eef orientation is wxyz; canonical schema stores xyzw.
    if "eef_ori" in obs:
        out["eef_quat"] = _as_float32_array(wxyz_to_xyzw(obs["eef_ori"]))
    elif "eef_quat" in obs:
        out["eef_quat"] = _as_float32_array(wxyz_to_xyzw(obs["eef_quat"]))

    for key, val in obs.items():
        if key == "health":
            out["health"] = val
        elif key == "proprio" or key.endswith("::proprio"):
            out["proprio"] = _canonicalize_proprio_vector(
                val, convert_quat_wxyz_to_xyzw=True
            )
        elif key.endswith("::rgb") or key == "rgb":
            name = b1k_camera_name_from_key(key) if "::" in key else "rgb"
            out[f"cam/{name}/rgb"] = _maybe_drop_alpha(val)
        elif key.endswith("::seg_instance") or key.endswith("::seg_semantic"):
            name = b1k_camera_name_from_key(key)
            out[f"cam/{name}/seg"] = val

    return filter_playback_obs(out)


def canonicalize_traj_obs(traj_data: list, suite: str) -> list:
    """In-place canonicalize ``obs`` on each trajectory step. Returns the same list."""
    if suite == "robocasa":
        fn = canonicalize_robocasa_obs
    elif suite == "b1k":
        fn = canonicalize_b1k_obs
    else:
        raise ValueError(f"Unknown suite '{suite}' (expected 'robocasa' or 'b1k')")

    for step in traj_data:
        if isinstance(step, dict) and "obs" in step and step["obs"] is not None:
            step["obs"] = fn(step["obs"])
    return traj_data


def cam_rgb_key(camera_name: str) -> str:
    return f"cam/{camera_name}/rgb"


def cam_seg_key(camera_name: str) -> str:
    return f"cam/{camera_name}/seg"


def canonical_seg_key_from_obs_info_camera(camera_name: str) -> str | None:
    """
    Map ``info/obs_info`` camera_name to canonical playback seg key.

    Examples:
      external_sensor0 -> cam/external_sensor0/seg
      franka0:eef_link:Camera:0 -> cam/eef/seg
      proprio -> None (skip)
    """
    if not camera_name or camera_name == "proprio":
        return None
    if "eef_link" in camera_name or ":Camera:" in camera_name:
        return cam_seg_key("eef")
    return cam_seg_key(camera_name)


def list_playback_cameras(obs_grp) -> list[str]:
    """Return camera names under ``obs/cam`` (empty if missing)."""
    if "cam" not in obs_grp:
        return []
    return sorted(obs_grp["cam"].keys())
