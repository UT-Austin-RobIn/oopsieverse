"""
Shared data-structures and defaults for task configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskConfig:
    """
    Everything the unified playback / visualisation script needs to know
    about a particular task.
    """

    # ── Identity ────────────────────────────────────────────────────────
    task_name: str

    # ── OG macros ───────────────────────────────────────────────────────
    use_gpu_dynamics: bool = False
    enable_transition_rules: bool = False

    # ── Scene ───────────────────────────────────────────────────────────
    scene_config: Dict[str, Any] = field(default_factory=dict)

    # ── Robot ───────────────────────────────────────────────────────────
    robot_config: Dict[str, Any] = field(default_factory=dict)
    robot_name: str = "franka0"
    robot_type: str = "FrankaPanda"

    # ── Task objects ────────────────────────────────────────────────────
    task_objects: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ── Cameras ─────────────────────────────────────────────────────────
    viewer_camera_pos: List[float] = field(default_factory=lambda: [0, 0, 0])
    viewer_camera_orn: List[float] = field(default_factory=lambda: [0, 0, 0, 1])
    external_camera_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ── Visualization config ────────────────────────────────────────────
    target_objects_health_with_links: List[str] = field(default_factory=list)
    target_objects_health: List[str] = field(default_factory=list)
    target_objects_forces: List[str] = field(default_factory=list)
    force_keys: List[str] = field(default_factory=lambda: ["filtered_qs_forces"])
    target_contact_bodies: List[str] = field(default_factory=list)
    # For electrical tasks
    target_objects_water_contacts: List[str] = field(default_factory=list)
    # For thermal tasks
    target_objects_temperature: List[str] = field(default_factory=list)

    # ── Default HDF5 paths ──────────────────────────────────────────────
    default_collect_hdf5: str = ""
    default_playback_hdf5: str = ""
    default_video_dir: str = ""

    # ── Optional task-specific playback wrapper class ───────────────────
    playback_wrapper_cls: Optional[Any] = None

    # ── Optional task-specific hooks ────────────────────────────────────
    # Called after env is created during playback (before playback starts)
    post_playback_env_setup: Optional[Any] = None  # Callable[[env], None]

