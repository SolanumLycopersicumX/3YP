import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pybullet as pb
except Exception as e:
    pb = None
try:
    import pybullet_data
except Exception:
    pybullet_data = None
import os


@dataclass
class PBConfig:
    step_size: float = 0.03
    max_steps: int = 50
    y_limits: Tuple[float, float] = (-0.5, 0.5)
    z_limits: Tuple[float, float] = (0.0, 1.0)
    substeps: int = 24
    time_step: float = 1.0 / 240.0
    use_gui: bool = True
    urdf_path: Optional[str] = None
    ee_link_index: Optional[int] = None
    fixed_base: bool = True
    gravity: Tuple[float, float, float] = (0, 0, -9.81)
    camera_distance: float = 1.0
    camera_yaw: float = 45
    camera_pitch: float = -30
    camera_target: Tuple[float, float, float] = (0.0, 0.0, 0.5)
    render_width: int = 640
    render_height: int = 480
    start_y: Optional[float] = None
    start_z: Optional[float] = None
    home_mid: bool = False
    home_joint_positions: Optional[Dict[int, float]] = None
    # Joint-drive mode config
    joint_drive: bool = False
    joint_lr_name: str = "shoulder_pan"
    joint_ud_name: str = "wrist_flex"
    joint_step: float = 0.05
    invert_lr: bool = False
    invert_ud: bool = False
    # Grid floor (debug lines)
    grid_floor: bool = False
    grid_size: float = 1.5
    grid_step: float = 0.1
    grid_height: float = 0.0
    grid_color: Tuple[float, float, float] = (0.6, 0.6, 0.6)
    grid_line_width: float = 1.0
    # Ground plane options
    load_plane: bool = True
    plane_urdf: Optional[str] = None


class PyBulletArmEnv(gym.Env):
    """A minimal PyBullet-based arm env to visualize Y/Z end-effector motion.

    - Action space: Discrete(4) [0:left, 1:right, 2:up, 3:down]
    - Observation: np.array([y, z], dtype=np.float32) of EE target
    - Reward: 0 (visualization only)
    - Termination: after max_steps
    - Rendering: human -> PyBullet GUI; rgb_array -> camera image
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}

    def __init__(self, cfg: PBConfig, render_mode: Optional[str] = None):
        super().__init__()
        if pb is None:
            raise ImportError("pybullet is not installed. Please install with `pip install pybullet pybullet_data`. ")

        self.cfg = cfg
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        low = np.array([self.cfg.y_limits[0], self.cfg.z_limits[0]], dtype=np.float32)
        high = np.array([self.cfg.y_limits[1], self.cfg.z_limits[1]], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._client = None
        self._robot = None
        self._plane = None
        self._ee_link_index = None
        self._q_indices: List[int] = []
        self._lower_limits: List[float] = []
        self._upper_limits: List[float] = []
        self._joint_ranges: List[float] = []
        self._rest_q: List[float] = []
        self._name_to_index: Dict[str, int] = {}
        self._idx_lr: Optional[int] = None
        self._idx_ud: Optional[int] = None

        self._target_y = 0.0
        self._target_z = 0.5
        self._step_count = 0

        # For rgb_array rendering
        self._view_matrix = None
        self._proj_matrix = None

        self._connect()
        self._reset_sim()

    # ---- gym API ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._reset_sim()
        obs = np.array([self._target_y, self._target_z], dtype=np.float32)
        return obs, {}

    def step(self, action: int):
        # If joint-drive enabled, map上下左右到关节角变化
        if self.cfg.joint_drive and (self._idx_lr is not None or self._idx_ud is not None):
            d_lr = 0.0
            d_ud = 0.0
            if action == 0:      # left
                d_lr = self.cfg.joint_step * (-1.0 if self.cfg.invert_lr else 1.0)
            elif action == 1:    # right
                d_lr = -self.cfg.joint_step * (-1.0 if self.cfg.invert_lr else 1.0)
            elif action == 2:    # up
                d_ud = self.cfg.joint_step * (-1.0 if self.cfg.invert_ud else 1.0)
            elif action == 3:    # down
                d_ud = -self.cfg.joint_step * (-1.0 if self.cfg.invert_ud else 1.0)

            if self._idx_lr is not None:
                cur = pb.getJointState(self._robot, self._idx_lr)[0]
                try:
                    jpos = self._q_indices.index(self._idx_lr)
                    lo = self._lower_limits[jpos]; hi = self._upper_limits[jpos]
                except ValueError:
                    lo, hi = -3.14, 3.14
                target = float(np.clip(cur + d_lr, lo, hi))
                pb.setJointMotorControl2(self._robot, self._idx_lr, controlMode=pb.POSITION_CONTROL, targetPosition=target, force=300)
            if self._idx_ud is not None:
                cur = pb.getJointState(self._robot, self._idx_ud)[0]
                try:
                    jpos = self._q_indices.index(self._idx_ud)
                    lo = self._lower_limits[jpos]; hi = self._upper_limits[jpos]
                except ValueError:
                    lo, hi = -3.14, 3.14
                target = float(np.clip(cur + d_ud, lo, hi))
                pb.setJointMotorControl2(self._robot, self._idx_ud, controlMode=pb.POSITION_CONTROL, targetPosition=target, force=300)
            for _ in range(self.cfg.substeps):
                pb.stepSimulation()
            self._step_count += 1
            obs = np.array([self._target_y, self._target_z], dtype=np.float32)
            terminated = self._step_count >= self.cfg.max_steps
            truncated = False
            reward = 0.0
            info = {}
            return obs, reward, terminated, truncated, info

        # 默认：使用 Y/Z 目标 + IK 控制
        # Convention: x+ 前, y+ 左, z+ 上
        # 0:left  -> +y, 1:right -> -y, 2:up -> +z, 3:down -> -z
        dy = (self.cfg.step_size if action == 0 else -self.cfg.step_size if action == 1 else 0.0)
        dz = (self.cfg.step_size if action == 2 else -self.cfg.step_size if action == 3 else 0.0)

        self._target_y = float(np.clip(self._target_y + dy, *self.cfg.y_limits))
        self._target_z = float(np.clip(self._target_z + dz, *self.cfg.z_limits))

        self._goto_target(self._target_y, self._target_z)
        self._step_count += 1

        obs = np.array([self._target_y, self._target_z], dtype=np.float32)
        terminated = self._step_count >= self.cfg.max_steps
        truncated = False
        reward = 0.0
        info = {"y": self._target_y, "z": self._target_z, "dy": dy, "dz": dz}

        if self.render_mode == "human":
            time.sleep(max(0.0, self.cfg.time_step * self.cfg.substeps))

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._view_matrix is None:
                self._setup_camera()
            w, h = self.cfg.render_width, self.cfg.render_height
            img = pb.getCameraImage(width=w, height=h, viewMatrix=self._view_matrix, projectionMatrix=self._proj_matrix)
            rgba = np.asarray(img[2], dtype=np.uint8).reshape(h, w, 4)
            rgb = rgba[..., :3]
            return rgb
        # human mode uses PyBullet GUI directly
        return None

    def get_status(self):
        """Return a summary dict for TUI/logging.
        Keys: ee_xyz (tuple), target_yz (tuple), step (int), max_steps (int).
        """
        ee = (np.nan, np.nan, np.nan)
        try:
            link = pb.getLinkState(self._robot, self._ee_link_index)
            if isinstance(link, (list, tuple)):
                ee = link[0]
        except Exception:
            pass
        eps = 1e-4
        hit_y_min = (self._target_y <= self.cfg.y_limits[0] + eps)
        hit_y_max = (self._target_y >= self.cfg.y_limits[1] - eps)
        hit_z_min = (self._target_z <= self.cfg.z_limits[0] + eps)
        hit_z_max = (self._target_z >= self.cfg.z_limits[1] - eps)
        return {
            "ee_xyz": (float(ee[0]), float(ee[1]), float(ee[2])) if ee is not None else (np.nan, np.nan, np.nan),
            "target_yz": (float(self._target_y), float(self._target_z)),
            "step": int(self._step_count),
            "max_steps": int(self.cfg.max_steps),
            "hit_y_min": hit_y_min,
            "hit_y_max": hit_y_max,
            "hit_z_min": hit_z_min,
            "hit_z_max": hit_z_max,
        }

    def close(self):
        if self._client is not None:
            pb.disconnect(self._client)
            self._client = None

    # ---- internal helpers ----
    def _connect(self):
        mode = pb.GUI if (self.render_mode == "human" and self.cfg.use_gui) else pb.DIRECT
        self._client = pb.connect(mode)
        if pybullet_data is not None:
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _reset_sim(self):
        pb.resetSimulation(physicsClientId=self._client)
        pb.setTimeStep(self.cfg.time_step)
        pb.setGravity(*self.cfg.gravity)
        # Load or create a ground plane
        self._plane = None
        if self.cfg.load_plane:
            loaded = False
            # 1) Preferred: user-provided plane URDF
            if self.cfg.plane_urdf is not None:
                try:
                    plane_path = os.path.abspath(self.cfg.plane_urdf)
                    plane_dir = os.path.dirname(plane_path)
                    if plane_dir:
                        pb.setAdditionalSearchPath(plane_dir)
                    self._plane = pb.loadURDF(plane_path)
                    loaded = True
                except Exception:
                    loaded = False
            # 2) Fallback: repository local asset (assets/pybullet/plane.urdf)
            if not loaded:
                try:
                    here = os.path.dirname(os.path.abspath(__file__))
                    local_plane = os.path.join(here, "assets", "pybullet", "plane.urdf")
                    if os.path.exists(local_plane):
                        pb.setAdditionalSearchPath(os.path.dirname(local_plane))
                        self._plane = pb.loadURDF(local_plane)
                        loaded = True
                except Exception:
                    loaded = False
            # 3) Next: pybullet_data/plane.urdf
            if (not loaded) and (pybullet_data is not None):
                try:
                    self._plane = pb.loadURDF("plane.urdf")
                    loaded = True
                except Exception:
                    loaded = False
            # 4) Last resort: procedural infinite plane
            if not loaded:
                try:
                    col_id = pb.createCollisionShape(pb.GEOM_PLANE, planeNormal=[0, 0, 1])
                    self._plane = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id)
                except Exception:
                    self._plane = None

        # Resolve robot URDF
        if self.cfg.urdf_path is not None:
            urdf = self.cfg.urdf_path
            # Ensure meshes/textures referenced relatively are resolvable
            try:
                urdf_dir = os.path.dirname(os.path.abspath(urdf))
                if urdf_dir:
                    pb.setAdditionalSearchPath(urdf_dir)
            except Exception:
                pass
        else:
            if pybullet_data is None:
                raise RuntimeError(
                    "No --urdf provided and pybullet_data is not available. "
                    "Install pybullet (it includes pybullet_data) or pass --urdf to your robot URDF."
                )
            urdf = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")

        # Spawn robot at origin
        self._robot = pb.loadURDF(urdf, basePosition=[0, 0, 0], useFixedBase=self.cfg.fixed_base)

        # Collect controllable joint indices
        self._q_indices = []
        self._lower_limits, self._upper_limits, self._joint_ranges = [], [], []
        n_j = pb.getNumJoints(self._robot)
        last_idx = 0
        for j in range(n_j):
            info = pb.getJointInfo(self._robot, j)
            joint_type = info[2]
            jname = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
            self._name_to_index[jname] = j
            if joint_type in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC):
                self._q_indices.append(j)
                last_idx = j
                lo = float(info[8]); hi = float(info[9])
                lo_f = lo if np.isfinite(lo) else -3.14
                hi_f = hi if np.isfinite(hi) else 3.14
                self._lower_limits.append(lo_f)
                self._upper_limits.append(hi_f)
                rng = (hi - lo) if (np.isfinite(hi) and np.isfinite(lo) and hi > lo) else (hi_f - lo_f)
                self._joint_ranges.append(rng if rng > 0 else 6.28)

        # Pick end-effector link
        self._ee_link_index = self.cfg.ee_link_index if self.cfg.ee_link_index is not None else last_idx

        # Initial target: current EE position
        # Initial target from config or current EE pose
        if self.cfg.start_y is not None and self.cfg.start_z is not None:
            self._target_y = float(np.clip(self.cfg.start_y, *self.cfg.y_limits))
            self._target_z = float(np.clip(self.cfg.start_z, *self.cfg.z_limits))
        else:
            link_state = pb.getLinkState(self._robot, self._ee_link_index)
            _, _, _, _, _, world_pos, _ = link_state[:7] + (link_state[4],)
            if world_pos is not None:
                self._target_y = float(np.clip(world_pos[1], *self.cfg.y_limits))
                self._target_z = float(np.clip(world_pos[2], *self.cfg.z_limits))
            else:
                self._target_y = 0.0
                self._target_z = 0.5

        # remember home pose for pulse/return-to-center
        self._home_y = float(self._target_y)
        self._home_z = float(self._target_z)
        # initialize rest poses with current states to stabilize IK
        self._rest_q = []
        for j in self._q_indices:
            try:
                self._rest_q.append(float(pb.getJointState(self._robot, j)[0]))
            except Exception:
                self._rest_q.append(0.0)

        self._setup_camera()
        self._step_count = 0
        # Apply home joint positions or mid pose if requested
        self._apply_home_pose()
        if self.cfg.joint_drive:
            self._idx_lr = self._name_to_index.get(self.cfg.joint_lr_name)
            self._idx_ud = self._name_to_index.get(self.cfg.joint_ud_name)
        # Optional grid on floor
        if self.cfg.grid_floor:
            try:
                self._draw_grid()
            except Exception:
                pass

    def _setup_camera(self):
        self._view_matrix = pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.cfg.camera_target,
            distance=self.cfg.camera_distance,
            yaw=self.cfg.camera_yaw,
            pitch=self.cfg.camera_pitch,
            roll=0,
            upAxisIndex=2,
        )
        self._proj_matrix = pb.computeProjectionMatrixFOV(
            fov=60, aspect=self.cfg.render_width / self.cfg.render_height, nearVal=0.01, farVal=10
        )

    def _goto_target(self, y: float, z: float):
        # Map target in YZ to world coordinates; keep X fixed near base
        target_pos = np.array([0.3, float(y), float(z)], dtype=np.float32)
        # Keep current end-effector orientation
        cur = pb.getLinkState(self._robot, self._ee_link_index)
        orn = cur[5] if len(cur) > 5 else pb.getQuaternionFromEuler([0, 0, 0])
        # IK to compute joint targets
        try:
            q = pb.calculateInverseKinematics(
                self._robot,
                self._ee_link_index,
                target_pos,
                orn,
                lowerLimits=self._lower_limits,
                upperLimits=self._upper_limits,
                jointRanges=self._joint_ranges,
                restPoses=self._rest_q,
                maxNumIterations=60,
            )
        except TypeError:
            q = pb.calculateInverseKinematics(self._robot, self._ee_link_index, target_pos, orn)
        # Apply to controllable joints
        for i, j_idx in enumerate(self._q_indices):
            if i < len(q):
                pb.setJointMotorControl2(self._robot, j_idx, controlMode=pb.POSITION_CONTROL, targetPosition=q[i], force=200)
        # Update rest poses to reduce null-space drift next step
        self._rest_q = [float(q[i]) if i < len(q) else self._rest_q[i] for i in range(len(self._q_indices))]
        # Step simulation
        for _ in range(self.cfg.substeps):
            pb.stepSimulation()

    def _apply_home_pose(self):
        if self.cfg.home_joint_positions:
            for j_idx, pos in self.cfg.home_joint_positions.items():
                try:
                    pb.setJointMotorControl2(self._robot, int(j_idx), controlMode=pb.POSITION_CONTROL, targetPosition=float(pos), force=300)
                except Exception:
                    pass
            for _ in range(self.cfg.substeps * 5):
                pb.stepSimulation()
        elif self.cfg.home_mid:
            n_j = pb.getNumJoints(self._robot)
            for j in range(n_j):
                info = pb.getJointInfo(self._robot, j)
                jtype = info[2]
                if jtype in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC):
                    lo, hi = float(info[8]), float(info[9])
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        mid = 0.5 * (lo + hi)
                        pb.setJointMotorControl2(self._robot, j, controlMode=pb.POSITION_CONTROL, targetPosition=mid, force=300)
            for _ in range(self.cfg.substeps * 5):
                pb.stepSimulation()

    # public helper for pulse mode
    def center(self):
        y = float(np.clip(getattr(self, "_home_y", 0.0), *self.cfg.y_limits))
        z = float(np.clip(getattr(self, "_home_z", 0.0), *self.cfg.z_limits))
        self._target_y, self._target_z = y, z
        self._goto_target(y, z)

    def _draw_grid(self):
        """Draw a simple XY grid at z=grid_height using user debug lines."""
        sz = float(self.cfg.grid_size)
        st = float(self.cfg.grid_step)
        z = float(self.cfg.grid_height)
        r, g, b = self.cfg.grid_color
        lw = self.cfg.grid_line_width
        # Lines parallel to X (vary Y)
        y_vals = np.arange(-sz, sz + 1e-6, st)
        for y in y_vals:
            pb.addUserDebugLine([ -sz, y, z], [ sz, y, z], [r, g, b], lineWidth=lw)
        # Lines parallel to Y (vary X)
        x_vals = np.arange(-sz, sz + 1e-6, st)
        for x in x_vals:
            pb.addUserDebugLine([ x, -sz, z], [ x,  sz, z], [r, g, b], lineWidth=lw)
