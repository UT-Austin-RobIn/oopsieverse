"""
Continuous gripper control devices for robosuite/RoboCasa teleoperation.

Provides drop-in replacements for robosuite's Keyboard and SpaceMouse devices
that track gripper position as a continuous float in [-1, 1] instead of the
default binary open/close toggle.

Classes:
    ContinuousGripperMixin       — shared base logic (position tracking, input2action override)
    ContinuousGripperKeyboard    — keyboard device with -/=/spacebar gripper bindings
    ContinuousGripperSpaceMouse  — SpaceMouse device with left/right button hold gripper control
"""

import threading

import numpy as np

from robosuite.devices import Keyboard, SpaceMouse
from robosuite.devices.device import Device

try:
    import hid
except ModuleNotFoundError as exc:
    hid = None  # deferred error raised only when ContinuousGripperSpaceMouse is instantiated

from pynput.keyboard import Key, Listener


# ═══════════════════════════════════════════════════════════════════════
# Gripper patching
# ═══════════════════════════════════════════════════════════════════════

def patch_gripper_for_position_control():
    """Monkey-patch PandaGripper class to use position-based control.

    Replaces velocity-based format_action with direct position mapping
    for continuous gripper support. Patches the class so it survives
    env.reset() which recreates gripper instances via hard_reset.
    """
    from robosuite.models.grippers.panda_gripper import PandaGripper

    def _position_format_action(self, action):
        assert len(action) == self.dof
        action_value = action[0] if hasattr(action, '__len__') else action
        self.current_action = np.clip(
            np.array([-1.0, 1.0]) * action_value, -1.0, 1.0
        )
        return self.current_action

    PandaGripper.format_action = _position_format_action


# ═══════════════════════════════════════════════════════════════════════
# Base mixin
# ═══════════════════════════════════════════════════════════════════════

class ContinuousGripperMixin:
    """
    Mixin providing shared continuous gripper control logic.

    Tracks gripper positions as continuous floats in [-1.0 (open), 1.0 (closed)]
    for every robot/arm pair in the environment.  Designed for multiple
    inheritance with robosuite Device subclasses.
    """

    def _init_gripper_positions(self, num_robots, all_robot_arms):
        """Initialize gripper position tracking for all robots and arms.

        Args:
            num_robots: Number of robots in the environment.
            all_robot_arms: List of arm lists, one per robot.
        """
        self.gripper_positions = {
            robot_idx: {
                arm_idx: -1.0
                for arm_idx in range(len(all_robot_arms[robot_idx]))
            }
            for robot_idx in range(num_robots)
        }

    def input2action(self, mirror_actions=False):
        """Convert device input to robot actions, preserving continuous gripper values.

        Overrides the parent class to prevent binary gripper conversion
        (``grasp = 1 if grasp else -1``) and instead pass the continuous
        gripper position straight through in [-1, 1].

        Args:
            mirror_actions: Whether to mirror actions across the y-axis.

        Returns:
            dict: Action dictionary with continuous gripper values, or None on reset.
        """
        robot = self.env.robots[self.active_robot]
        active_arm = self.active_arm
        state = self.get_controller_state()

        dpos, rotation, raw_drotation, grasp, reset = (
            state["dpos"],
            state.get("rotation", None),
            state["raw_drotation"],
            state["grasp"],
            state["reset"],
        )

        if mirror_actions:
            dpos[0] *= -1
            dpos[1] *= -1
            raw_drotation[0] *= -1
            raw_drotation[1] *= -1

        if reset:
            return None

        gripper_dof = robot.gripper[active_arm].dof
        drotation = raw_drotation[[1, 0, 2]]
        drotation[2] = -drotation[2]
        dpos, drotation = self._postprocess_device_outputs(dpos, drotation)

        ac_dict = {}
        for arm in robot.arms:
            arm_action = self.get_arm_action(
                robot,
                arm,
                norm_delta=np.zeros(6),
            )
            ac_dict[f"{arm}_abs"] = arm_action["abs"]
            ac_dict[f"{arm}_delta"] = arm_action["delta"]
            ac_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)

        if robot.is_mobile:
            base_mode = bool(state["base_mode"])
            if base_mode is True:
                arm_norm_delta = np.zeros(6)
                base_ac = np.array([dpos[0], dpos[1], drotation[2]])
            else:
                arm_norm_delta = np.concatenate([dpos, drotation])
                base_ac = np.zeros(3)
            ac_dict["base"] = base_ac
            ac_dict["base_mode"] = np.array([1 if base_mode is True else -1])
        else:
            arm_norm_delta = np.concatenate([dpos, drotation])

        arm_action = self.get_arm_action(
            robot,
            active_arm,
            norm_delta=arm_norm_delta,
        )
        ac_dict[f"{active_arm}_abs"] = arm_action["abs"]
        ac_dict[f"{active_arm}_delta"] = arm_action["delta"]
        # Preserve continuous gripper position instead of binary conversion
        ac_dict[f"{active_arm}_gripper"] = np.array([grasp] * gripper_dof)

        for k, v in ac_dict.items():
            if "abs" not in k:
                ac_dict[k] = np.clip(v, -1, 1)

        return ac_dict

    def _reset_gripper_positions(self):
        """Reset all gripper positions to fully open (-1.0)."""
        if hasattr(self, "gripper_positions"):
            for robot_idx in self.gripper_positions:
                for arm_idx in self.gripper_positions[robot_idx]:
                    self.gripper_positions[robot_idx][arm_idx] = -1.0


# ═══════════════════════════════════════════════════════════════════════
# Keyboard device
# ═══════════════════════════════════════════════════════════════════════

class ContinuousGripperKeyboard(ContinuousGripperMixin, Keyboard):
    """Keyboard device with continuous gripper control.

    Extends the standard robosuite Keyboard device to track gripper position
    as a continuous float in [-1.0, 1.0] instead of binary open/close.

    Key bindings (in addition to standard keyboard controls):
        ``-``        Open gripper incrementally.
        ``=``        Close gripper incrementally.
        spacebar     Toggle between fully open (-1) and fully closed (1).

    Args:
        env (RobotEnv): The environment containing the robot(s) to control.
        pos_sensitivity (float): Position command scaling.
        rot_sensitivity (float): Rotation command scaling.
        gripper_step_size (float): Increment per keypress (default 0.05).
    """

    def __init__(self, env, pos_sensitivity=1.0, rot_sensitivity=1.0, gripper_step_size=0.05):
        self.gripper_step_size = gripper_step_size
        super().__init__(env, pos_sensitivity, rot_sensitivity)
        self._init_gripper_positions(self.num_robots, self.all_robot_arms)

    @staticmethod
    def _display_controls():
        """Print keyboard control bindings."""
        def print_command(char, info):
            print(f"{char:30s}\t{info}")

        print("\nKeyboard Controls:")
        print_command("Ctrl+q", "reset simulation")
        print_command("-", "open gripper incrementally")
        print_command("=", "close gripper incrementally")
        print_command("spacebar", "toggle gripper (fully open/closed)")
        print_command("up-right-down-left", "move horizontally in x-y plane")
        print_command(".-;", "move vertically")
        print_command("o-p / y-h / e-r", "rotate (yaw/pitch/roll)")
        print_command("b", "toggle arm/base mode (if applicable)")
        print_command("s", "switch active arm (if multi-armed)")
        print()

    def on_press(self, key):
        """Handle key presses for continuous gripper control."""
        try:
            if hasattr(key, "char") and key.char == "-":
                self.gripper_positions[self.active_robot][self.active_arm_index] -= self.gripper_step_size
                self.gripper_positions[self.active_robot][self.active_arm_index] = np.clip(
                    self.gripper_positions[self.active_robot][self.active_arm_index], -1.0, 1.0
                )
                return
            elif hasattr(key, "char") and key.char == "=":
                self.gripper_positions[self.active_robot][self.active_arm_index] += self.gripper_step_size
                self.gripper_positions[self.active_robot][self.active_arm_index] = np.clip(
                    self.gripper_positions[self.active_robot][self.active_arm_index], -1.0, 1.0
                )
                return
        except Exception:
            pass
        super().on_press(key)

    def on_release(self, key):
        """Handle key releases; spacebar toggles gripper fully open/closed."""
        if key == Key.space:
            current_pos = self.gripper_positions[self.active_robot][self.active_arm_index]
            self.gripper_positions[self.active_robot][self.active_arm_index] = (
                -1.0 if current_pos > 0.0 else 1.0
            )
            return
        super().on_release(key)

    def get_controller_state(self):
        """Return controller state with continuous gripper value."""
        dpos = self.pos - self.last_pos
        self.last_pos = np.array(self.pos)
        raw_drotation = self.raw_drotation - self.last_drotation
        self.last_drotation = np.array(self.raw_drotation)
        grasp = self.gripper_positions[self.active_robot][self.active_arm_index]
        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=grasp,
            reset=self._reset_state,
            base_mode=int(self.base_mode),
        )

    def _reset_internal_state(self):
        """Reset internal state including gripper positions."""
        super()._reset_internal_state()
        self._reset_gripper_positions()


# ═══════════════════════════════════════════════════════════════════════
# SpaceMouse device
# ═══════════════════════════════════════════════════════════════════════

class ContinuousGripperSpaceMouse(ContinuousGripperMixin, SpaceMouse):
    """SpaceMouse device with continuous gripper control.

    Extends the standard robosuite SpaceMouse device to track gripper position
    as a continuous float in [-1.0, 1.0] instead of binary open/close.

    Button bindings:
        Left button (hold)   Open gripper continuously.
        Right button (hold)  Close gripper continuously.

    Args:
        env (RobotEnv): The environment containing the robot(s) to control.
        vendor_id (int): SpaceMouse vendor ID (defaults to robosuite macro).
        product_id (int): SpaceMouse product ID (defaults to robosuite macro).
        pos_sensitivity (float): Position command scaling.
        rot_sensitivity (float): Rotation command scaling.
        gripper_step_size (float): Increment per update cycle (default 0.01).
    """

    def __init__(
        self,
        env,
        vendor_id=None,
        product_id=None,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
        gripper_step_size=0.01,
    ):
        if hid is None:
            raise ImportError(
                "Unable to load module hid, required to interface with SpaceMouse. "
                "Install with: pip install hidapi"
            )

        import robosuite.macros as macros

        if vendor_id is None:
            vendor_id = macros.SPACEMOUSE_VENDOR_ID
        if product_id is None:
            product_id = macros.SPACEMOUSE_PRODUCT_ID

        # Track button hold states BEFORE Device.__init__ because run() starts during init
        self.left_button_held = False
        self.right_button_held = False
        self.gripper_step_size = gripper_step_size
        self.listener = None

        # Call Device.__init__ directly to avoid SpaceMouse.__init__ re-opening device
        Device.__init__(self, env)

        print("Opening SpaceMouse device")
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.device = hid.device()
        try:
            self.device.open(self.vendor_id, self.product_id)
        except OSError:
            print(
                "Failed to open SpaceMouse device. "
                "Consider killing other processes like 3DconnexionHelper "
                "(killall 3DconnexionHelper)"
            )
            raise

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        print("Manufacturer: %s" % self.device.get_manufacturer_string())
        print("Product: %s" % self.device.get_product_string())

        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()

        self.single_click_and_hold = False
        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False

        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

        try:
            self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()
            print("Keyboard listener started successfully")
        except Exception as e:
            print(f"Warning: Could not start keyboard listener: {e}")
            print("Keyboard controls (q, b, s) will not be available")
            print("SpaceMouse buttons will still work for gripper control")
            self.listener = None

        self._init_gripper_positions(self.num_robots, self.all_robot_arms)

    @staticmethod
    def _display_controls():
        """Print SpaceMouse control bindings."""
        def print_command(char, info):
            print(f"{char:30s}\t{info}")

        print("\nSpaceMouse Controls:")
        print_command("Left button (hold)", "open gripper continuously")
        print_command("Right button (hold)", "close gripper continuously")
        print_command("Move laterally", "move arm horizontally")
        print_command("Move vertically", "move arm vertically")
        print_command("Twist", "rotate arm (roll/pitch/yaw)")
        print_command("q (keyboard)", "quit/reset simulation")
        print_command("b", "toggle arm/base mode (if applicable)")
        print_command("s", "switch active arm (if multi-armed)")
        print()

    @property
    def control_gripper(self):
        """Continuous gripper position in [-1.0, 1.0] (open to closed)."""
        return self.gripper_positions[self.active_robot][self.active_arm_index]

    def on_press(self, key):
        """Delegate all key presses to parent."""
        super().on_press(key)

    def on_release(self, key):
        """Handle 'q' for reset; delegate everything else to parent."""
        try:
            if hasattr(key, "char") and key.char == "q":
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()
                return
        except AttributeError:
            pass
        super().on_release(key)

    def run(self):
        """Listener thread: reads SpaceMouse HID packets and applies continuous gripper updates.

        Left button held → open gripper; right button held → close gripper.
        Runs at ~60 Hz (16 ms HID read timeout) for smooth control.
        """
        while True:
            d = self.device.read(13, timeout_ms=16)
            if self._enabled:
                if d is not None and len(d) > 0:
                    if self.product_id == 50741:
                        # Older SpaceMouse model
                        from robosuite.devices.spacemouse import convert
                        if d[0] == 1:
                            self.x = convert(d[3], d[4])
                            self.y = convert(d[1], d[2])
                            self.z = convert(d[5], d[6]) * -1.0
                        elif d[0] == 2:
                            self.roll = convert(d[1], d[2])
                            self.pitch = convert(d[3], d[4])
                            self.yaw = convert(d[5], d[6])
                        self._control = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]
                    else:
                        from robosuite.devices.spacemouse import convert
                        if d[0] == 1:
                            self.x = convert(d[3], d[4])
                            self.y = convert(d[1], d[2])
                            self.z = convert(d[5], d[6]) * -1.0
                            self.roll = convert(d[7], d[8])
                            self.pitch = convert(d[9], d[10])
                            self.yaw = convert(d[11], d[12])
                            self._control = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]

                    if d[0] == 3:
                        # Side button events: 1=left pressed, 2=right pressed, 0=released
                        if d[1] == 1:
                            self.left_button_held = True
                            self.right_button_held = False
                        elif d[1] == 2:
                            self.left_button_held = False
                            self.right_button_held = True
                        elif d[1] == 0:
                            self.left_button_held = False
                            self.right_button_held = False

                # Apply continuous gripper changes each loop (~60 Hz)
                if self.left_button_held:
                    self.gripper_positions[self.active_robot][self.active_arm_index] -= self.gripper_step_size
                    self.gripper_positions[self.active_robot][self.active_arm_index] = np.clip(
                        self.gripper_positions[self.active_robot][self.active_arm_index], -1.0, 1.0
                    )
                if self.right_button_held:
                    self.gripper_positions[self.active_robot][self.active_arm_index] += self.gripper_step_size
                    self.gripper_positions[self.active_robot][self.active_arm_index] = np.clip(
                        self.gripper_positions[self.active_robot][self.active_arm_index], -1.0, 1.0
                    )

    def _reset_internal_state(self):
        """Reset controller state including gripper positions and button hold flags."""
        super()._reset_internal_state()
        self._reset_gripper_positions()
        self.left_button_held = False
        self.right_button_held = False
