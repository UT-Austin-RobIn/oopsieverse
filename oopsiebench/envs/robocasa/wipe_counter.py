"""
Wipe Counter environment for oopsieverse.

Task: wipe the dirt on the counter with the sponge.
"""

import numpy as np
import robocasa.utils.env_utils as EnvUtils
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES
from robocasa.models.scenes.scene_registry import LayoutType, StyleType

from robosuite.models.objects import CylinderObject
from robosuite.utils.mjcf_utils import CustomMaterial

from damagesim.robosuite.damageable_env import RSDamageableEnvironment


DEFAULT_WIPE_CONFIG = {
    "num_markers": 50,
    "line_width": 0.02,
    "contact_threshold": 100,
}


class WipeCounter(Kitchen):

    def __init__(self, wipe_config=None, *args, **kwargs):
        kwargs.pop("layout_ids", None)
        kwargs.pop("style_ids", None)

        # Wipe task configuration
        self.wipe_config = wipe_config if wipe_config is not None else DEFAULT_WIPE_CONFIG
        self.num_markers = self.wipe_config["num_markers"]
        self.line_width = self.wipe_config["line_width"]
        self.contact_threshold = self.wipe_config["contact_threshold"]

        # Dirt markers
        self.markers = []
        self.wiped_markers = []
        self._marker_direction = None

        self.ee_force_bias = {}
        self._dirt_bounds = None

        super().__init__(
            layout_ids=LayoutType.LAYOUT002,
            style_ids=StyleType.STYLE004,
            *args,
            **kwargs,
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Wipe the dirt on the counter with the sponge"
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.get_fixture(FixtureType.SINK)
        self.counter = self.get_fixture(FixtureType.COUNTER, ref=self.sink)
        self.init_robot_base_ref = self.counter

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.counter, offset=[1.0, 0.0]
        )
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori
        self._add_dirt_markers()

    def _get_obj_cfgs(self):
        sponge_1_path = next(
            p for p in OBJ_CATEGORIES["sponge"]["objaverse"].mjcf_paths
            if p.split("/")[-2] == "sponge_1"
        )

        return [
            dict(
                name="sponge",
                obj_groups=sponge_1_path,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="right"),
                    size=(0.1, 0.1),
                    offset=(0, -0.50),
                    rotation=0.0,
                ),
            )
        ]

    def _add_dirt_markers(self):
        """Add visual dirt markers to the counter surface."""

        # Define dirt bounds relative to sink
        self._dirt_bounds = {
            'x_min': self.sink.pos[0] - 1.30,
            'x_max': self.sink.pos[0] - 0.90,
            'y_min': self.sink.pos[1] - 0.1,
            'y_max': self.sink.pos[1] + 0.05,
        }

        # Define dirt material
        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.0", "shininess": "0.0"}
        dirt_brown_rgba = [0.545, 0.451, 0.333, 1.0]
        dirt = CustomMaterial(
            texture=dirt_brown_rgba,
            tex_name="dirt",
            mat_name="dirt_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
            shared=True,
        )

        pos = self._sample_start_pos()

        # Create dirt markers along a path
        for i in range(self.num_markers):
            marker_name = f"dirt_marker{i}"
            marker = CylinderObject(
                name=marker_name,
                size=[self.line_width / 2, 0.001],
                rgba=[0.9, 0.5, 0.1, 1.0],
                material=dirt,
                obj_type="visual",
                joints=None,
            )

            self.model.merge_assets(marker)
            marker_obj = marker.get_obj()

            for site in marker_obj.findall("site"):
                marker_obj.remove(site)

            marker_obj.set("pos", f"{pos[0]} {pos[1]} {self.counter.pos[2] + 0.001}")
            self.model.worldbody.append(marker_obj)

            self.markers.append(marker)
            pos = self._sample_path_pos(pos)

    def _sample_start_pos(self, ref_pos=None):
        """Sample a starting position for the dirt path."""
        self._marker_direction = float(np.random.uniform(-np.pi, np.pi))

        if ref_pos is not None:
            return np.array([
                float(ref_pos[0] - 0.18) + float(np.random.uniform(-0.05, 0.05)),
                float(ref_pos[1]) + float(np.random.uniform(-0.05, 0.05))
            ])

        if self._dirt_bounds is None:
            return np.array([0.0, 0.0])

        return np.array([
            np.random.uniform(self._dirt_bounds['x_min'], self._dirt_bounds['x_max']),
            np.random.uniform(self._dirt_bounds['y_min'], self._dirt_bounds['y_max']),
        ])

    def _sample_path_pos(self, pos):
        """Sample next position along the dirt path."""
        if np.random.uniform(0, 1) > 0.7:
            self._marker_direction = float(self._marker_direction) + float(np.random.normal(0, 0.5))

        step_size = 0.008
        posnew0 = pos[0] + step_size * np.sin(float(self._marker_direction))
        posnew1 = pos[1] + step_size * np.cos(float(self._marker_direction))

        # Keep resampling until within bounds
        attempts = 0
        while attempts < 20 and self._dirt_bounds is not None:
            if (self._dirt_bounds['x_min'] <= posnew0 <= self._dirt_bounds['x_max'] and
                self._dirt_bounds['y_min'] <= posnew1 <= self._dirt_bounds['y_max']):
                break
            self._marker_direction = float(self._marker_direction) + float(np.random.normal(0, 0.5))
            posnew0 = pos[0] + step_size * np.sin(float(self._marker_direction))
            posnew1 = pos[1] + step_size * np.cos(float(self._marker_direction))
            attempts += 1

        return np.array([posnew0, posnew1])

    def _reset_internal(self):
        super()._reset_internal()
        self.wiped_markers = []
        self._reset_dirt_markers()

        # Reset ee force bias (will be set on first action)
        self.ee_force_bias = {arm: np.zeros(3) for arm in self.robots[0].arms}

    def reset(self):
        obs = super().reset()
        return obs

    def _reset_dirt_markers(self):
        """Reset dirt marker positions and visibility."""
        if not self.markers or self.sim is None:
            return

        # Get sponge position to spawn dirt near it
        start_center = None
        if hasattr(self, 'objects') and 'sponge' in self.objects:
            try:
                sponge = self.objects['sponge']
                sponge_body_id = self.sim.model.body_name2id(sponge.root_body)
                sponge_pos = self.sim.data.body_xpos[sponge_body_id]
                start_center = sponge_pos[:2]
            except Exception:
                pass

        # Sample new positions for markers
        pos = self._sample_start_pos(ref_pos=start_center)

        # Try to get actual counter body Z from physics
        counter_body_id = self.sim.model.body_name2id(self.counter.root_body)
        z_pos = self.sim.data.body_xpos[counter_body_id][2] + 0.48

        for i, marker in enumerate(self.markers):
            try:
                geom_id = self.sim.model.geom_name2id(marker.visual_geoms[0])
                self.sim.model.geom_rgba[geom_id][3] = 1.0

                # Update marker position
                body_id = self.sim.model.body_name2id(marker.root_body)
                position = np.array([
                    pos[0],
                    pos[1],
                    z_pos
                ])
                self.sim.model.body_pos[body_id] = position
            except Exception:
                pass

            pos = self._sample_path_pos(pos)

    def _get_active_markers(self):
        """Get the markers that are currently being wiped by the sponge."""
        active_markers = []

        if not hasattr(self, 'objects') or 'sponge' not in self.objects:
            return active_markers

        # Only process if there are contacts in the simulation
        if self.sim.data.ncon == 0:
            return active_markers

        try:
            sponge = self.objects['sponge']
            sponge_body_id = self.sim.model.body_name2id(sponge.root_body)
            sponge_pos = np.array(self.sim.data.body_xpos[sponge_body_id])

            # Get actual sponge dimensions from its collision geom
            sponge_half_size = np.array([0.034, 0.055, 0.011])  # Default fallback
            for geom_name in sponge.contact_geoms:
                try:
                    geom_id = self.sim.model.geom_name2id(geom_name)
                    geom_size = self.sim.model.geom_size[geom_id]
                    sponge_half_size = geom_size
                    break
                except Exception:
                    continue
        except Exception:
            return active_markers

        # Use the larger of X/Y half-sizes as the effective radius
        sponge_radius = max(sponge_half_size[0], sponge_half_size[1])

        # Check each marker
        for marker in self.markers:
            if marker in self.wiped_markers:
                continue

            try:
                marker_body_id = self.sim.model.body_name2id(marker.root_body)
                marker_pos = np.array(self.sim.data.body_xpos[marker_body_id])

                # Vector from sponge to marker
                v = marker_pos - sponge_pos

                # XY distance (horizontal)
                xy_dist = np.linalg.norm(v[:2])

                # Sponge bottom Z position
                sponge_bottom_z = sponge_pos[2] - sponge_half_size[2]

                # Check if marker is within sponge contact zone
                is_xy_close = xy_dist < (sponge_radius + self.line_width)
                is_pressing = sponge_bottom_z < (marker_pos[2] + 0.01)

                if is_xy_close and is_pressing:
                    active_markers.append(marker)

            except Exception:
                continue

        return active_markers

    def _check_wiped_markers(self):
        """Check if sponge is wiping any dirt markers."""
        newly_wiped = []

        # First check if there's gripper contact (force-based)
        if not self._has_gripper_contact:
            return newly_wiped

        # Get markers that are geometrically close to the sponge
        active_markers = self._get_active_markers()

        # Find newly wiped markers (active but not already wiped)
        for marker in active_markers:
            if marker not in self.wiped_markers:
                try:
                    # Make marker transparent
                    geom_id = self.sim.model.geom_name2id(marker.visual_geoms[0])
                    self.sim.model.geom_rgba[geom_id][3] = 0.0
                    self.wiped_markers.append(marker)
                    newly_wiped.append(marker)
                except Exception:
                    continue

        return newly_wiped

    @property
    def _has_gripper_contact(self):
        """Determines whether the gripper/sponge is making contact with an object."""
        try:
            for arm in self.robots[0].arms:
                current_force = self.robots[0].ee_force.get(arm)
                bias = self.ee_force_bias.get(arm, np.zeros(3))

                if current_force is not None:
                    force_diff = np.linalg.norm(current_force - bias)
                    if force_diff > self.contact_threshold:
                        return True
        except Exception:
            pass
        return False

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        # Update force bias on first action
        if self.ee_force_bias and all([np.linalg.norm(self.ee_force_bias.get(arm, np.zeros(3))) == 0 for arm in self.robots[0].arms]):
            self.ee_force_bias = {arm: self.robots[0].ee_force.get(arm, np.zeros(3)) for arm in self.robots[0].arms}

        # Check for newly wiped markers
        newly_wiped = self._check_wiped_markers()

        # Add wiping info to info dict
        info['newly_wiped'] = len(newly_wiped)
        info['total_wiped'] = len(self.wiped_markers)
        info['total_markers'] = len(self.markers)
        info['proportion_wiped'] = len(self.wiped_markers) / max(1, len(self.markers))

        return reward, done, info

    def reward(self, action=None):
        if len(self.markers) > 0:
            return len(self.wiped_markers) / len(self.markers)
        return 0.0

    def _check_success(self):
        return len(self.wiped_markers) == len(self.markers) and len(self.markers) > 0


class DamageableWipeCounter(RSDamageableEnvironment, WipeCounter):
    """WipeCounter with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="wipe_counter", *args, **kwargs)

    def _register_dirt_markers_in_segmentation(self):
        """Add dirt markers to segmentation mapping so they appear in seg images."""
        if not hasattr(self, 'seg_lut') or self.seg_lut is None:
            return

        # Find a unique class ID for dirt (one higher than current max)
        dirt_class_id = max(self.seg_id_to_class.keys()) + 1
        self.seg_id_to_class[dirt_class_id] = "dirt"

        # Add each dirt marker's geom to the LUT
        for marker in self.markers:
            try:
                for geom_name in marker.visual_geoms:
                    geom_id = self.sim.model.geom_name2id(geom_name)
                    if geom_id < len(self.seg_lut):
                        self.seg_lut[geom_id] = dirt_class_id
            except Exception:
                continue

    def reset(self):
        """Reset with robot positioning after segmentation is initialized."""
        obs, info = RSDamageableEnvironment.reset(self)

        # Register dirt markers in segmentation (after seg_lut is created)
        self._register_dirt_markers_in_segmentation()

        # Now safe to call reset_robot_above_sponge (which uses env.step())
        reset_robot_above_sponge(self, height_offset=0.0)

        return obs, info


def reset_robot_above_sponge(
    env,
    object_name: str = "sponge",
    height_offset: float = 0.2,
    position_threshold: float = 0.01,
    max_steps: int = 100,
    verbose: bool = False,
):
    """
    Move robot EEF to a position above the sponge using IK.

    Uses iterative IK solving to move the end-effector to the target position.

    Args:
        env: The environment instance
        object_name: Name of the object to position above (default: "sponge")
        height_offset: Height (meters) above the object (default: 0.2 = 20cm)
        position_threshold: Distance threshold to consider "close enough" (default: 1cm)
        max_steps: Maximum IK iterations
        verbose: Print progress information

    Returns:
        tuple: (obs, info) - Final observation and info after positioning
    """
    from robosuite.controllers.parts.arm.ik import InverseKinematicsController

    robot = env.robots[0]
    arm = robot.arms[0]  # Usually "right" for single-arm robots

    # Get object position
    obj_body_id = env.obj_body_id.get(object_name)
    if obj_body_id is None:
        print(f"Warning: Object '{object_name}' not found in environment")
        obs, obs_info = env.get_observations()
        return obs, {"obs_info": obs_info}

    obj_pos = np.array(env.sim.data.body_xpos[obj_body_id])
    target_pos = obj_pos + np.array([0.0, 0.0, height_offset])

    # Get EEF site name for IK
    eef_site_name = f"gripper0_{arm}_grip_site"

    if verbose:
        print(f"Moving EEF to {height_offset*100:.0f}cm above {object_name}")
        print(f"  Object position: {obj_pos}")
        print(f"  Target EEF position: {target_pos}")

    # Get joint indices for the arm
    joint_indices = robot._ref_joint_pos_indexes

    # Lock health during reset motion
    env.lock_health_changes()

    for step in range(max_steps):
        # Get current EEF position
        current_eef_pos = np.array(env.sim.data.get_site_xpos(eef_site_name))
        current_eef_mat = np.array(env.sim.data.get_site_xmat(eef_site_name)).reshape(3, 3)

        # Compute position error
        pos_error = target_pos - current_eef_pos
        distance = np.linalg.norm(pos_error)

        # Check if close enough
        if distance < position_threshold:
            if verbose:
                print(f"  Reached target in {step} steps (distance: {distance*1000:.1f}mm)")
            break

        # Compute desired joint positions using IK
        try:
            q_des = InverseKinematicsController.compute_joint_positions(
                sim=env.sim,
                initial_joint=env.sim.data.qpos[joint_indices].copy(),
                joint_indices=joint_indices,
                ref_name=eef_site_name,
                control_freq=env.control_freq,
                use_delta=True,
                dpos=pos_error * 0.5,  # Scale down for stability
                drot=np.eye(3),  # Keep current orientation (identity = no change)
                Kpos=0.95,
                Kori=0.0,  # Don't try to change orientation
                integration_dt=1.0 / env.control_freq,
            )

            # Set gripper joint positions
            robot.set_gripper_joint_positions(np.array([0.04, -0.04]), gripper_arm=arm)

            # Set joint positions directly
            robot.set_robot_joint_positions(q_des)

            # Step simulation to update physics
            env.sim.forward()

        except Exception as e:
            if verbose:
                print(f"  IK step {step} failed: {e}")
            break

        # Progress logging
        if verbose and (step + 1) % 20 == 0:
            print(f"  Step {step + 1}: distance = {distance*1000:.1f}mm")

    else:
        # Max steps reached
        final_eef_pos = np.array(env.sim.data.get_site_xpos(eef_site_name))
        final_distance = np.linalg.norm(target_pos - final_eef_pos)
        if verbose:
            print(f"  Warning: Max steps ({max_steps}) reached. Final distance: {final_distance*1000:.1f}mm")

    # Unlock health tracking
    env.unlock_health_changes()

    # Step a few times to let physics settle
    action = np.zeros(12)
    action[6] = -1.0
    for _ in range(5): obs, _, _, info = env.step(action)

    return obs, info
