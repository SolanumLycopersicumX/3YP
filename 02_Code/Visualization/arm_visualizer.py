import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ArmFrame:
    arm_rgb: np.ndarray | None
    trajectory_yz: list[tuple[float, float]]
    status: dict[str, object] = field(default_factory=dict)


def load_pybullet_env():
    simulation_dir = PROJECT_ROOT / "02_Code" / "Simulation"
    simulation_path = str(simulation_dir)
    if simulation_path not in sys.path:
        sys.path.insert(0, simulation_path)

    from pybullet_arm_env import PBConfig, PyBulletArmEnv

    return PyBulletArmEnv, PBConfig


def default_so101_urdf_path() -> Path:
    return PROJECT_ROOT / "02_Code" / "lerobot" / "examples" / "assets" / "SO101" / "so101_new_calib.urdf"


class ArmVisualizer:
    def __init__(
        self,
        env_cls=None,
        cfg_cls=None,
        step_size=0.03,
        urdf_path: str | Path | None = None,
        camera_yaw=135.0,
        camera_pitch=-25.0,
        camera_distance=0.85,
        camera_target=(0.0, 0.0, 0.25),
    ):
        self.env_cls = env_cls
        self.cfg_cls = cfg_cls
        self.step_size = step_size
        self.urdf_path = self._resolve_urdf_path(urdf_path)
        self.arm_model = "SO-101" if self.urdf_path is not None else "KUKA iiwa"
        self.camera_yaw = float(camera_yaw)
        self.camera_pitch = float(camera_pitch)
        self.camera_distance = float(camera_distance)
        self.camera_target = tuple(float(value) for value in camera_target)
        self.env = None
        self.fallback_y = 0.0
        self.fallback_z = 0.0
        self.trajectory_yz: list[tuple[float, float]] = []
        self.arm_mode = "uninitialized"
        self.fallback_error = None
        self._env_init_attempted = False

    def _ensure_env(self):
        if self.env is not None or self.arm_mode == "fallback":
            return self.env

        try:
            if self.env_cls is None or self.cfg_cls is None:
                self.env_cls, self.cfg_cls = load_pybullet_env()
            config = self.cfg_cls(
                step_size=self.step_size,
                use_gui=False,
                urdf_path=str(self.urdf_path) if self.urdf_path is not None else None,
                arm_model=self.arm_model,
                camera_yaw=self.camera_yaw,
                camera_pitch=self.camera_pitch,
                camera_distance=self.camera_distance,
                camera_target=self.camera_target,
            )
            self.env = self.env_cls(config, render_mode="rgb_array")
            self.arm_mode = "pybullet"
            self.fallback_error = None
        except Exception as exc:
            self.env = None
            self.arm_mode = "fallback"
            self.fallback_error = str(exc)
        finally:
            self._env_init_attempted = True

        return self.env

    def reset(self) -> ArmFrame:
        self.fallback_y = 0.0
        self.fallback_z = 0.0
        self.trajectory_yz = []

        env = self._ensure_env()
        if env is None:
            return self._fallback_frame()

        try:
            obs = self._extract_obs(env.reset())
            self._append_obs(obs)
            return self._env_frame()
        except Exception as exc:
            return self._degrade_to_fallback(exc)

    def step(self, action: int | None) -> ArmFrame:
        if action is not None and action not in {0, 1, 2, 3}:
            raise ValueError(f"Invalid arm action: {action}")

        env = self._ensure_env()

        if action is None:
            if env is None:
                return self._fallback_frame()
            try:
                return self._env_frame()
            except Exception as exc:
                return self._degrade_to_fallback(exc)

        if env is None:
            self._step_fallback(action)
            return self._fallback_frame()

        try:
            obs = self._extract_obs(env.step(action))
            self._append_obs(obs)
            return self._env_frame()
        except Exception as exc:
            return self._degrade_to_fallback(exc)

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.env = None

    def _env_frame(self) -> ArmFrame:
        return ArmFrame(
            arm_rgb=self.env.render(),
            trajectory_yz=list(self.trajectory_yz),
            status=self._status(),
        )

    def _fallback_frame(self) -> ArmFrame:
        return ArmFrame(
            arm_rgb=None,
            trajectory_yz=list(self.trajectory_yz),
            status=self._status(),
        )

    def _status(self) -> dict[str, object]:
        status: dict[str, object] = {"arm_mode": self.arm_mode}
        status["arm_model"] = self.arm_model
        status["camera_yaw"] = self.camera_yaw
        status["camera_pitch"] = self.camera_pitch
        status["camera_distance"] = self.camera_distance
        if self.fallback_error:
            status["error"] = self.fallback_error
        return status

    def _degrade_to_fallback(self, exc: Exception) -> ArmFrame:
        self._close_env_best_effort()
        self.arm_mode = "fallback"
        self.fallback_error = str(exc)
        return self._fallback_frame()

    def _close_env_best_effort(self):
        try:
            self.close()
        except Exception:
            pass

    def _step_fallback(self, action: int):
        if action == 0:
            self.fallback_y += self.step_size
        elif action == 1:
            self.fallback_y -= self.step_size
        elif action == 2:
            self.fallback_z += self.step_size
        elif action == 3:
            self.fallback_z -= self.step_size
        self.trajectory_yz.append((self.fallback_y, self.fallback_z))

    def _append_obs(self, obs):
        self.trajectory_yz.append((float(obs[0]), float(obs[1])))

    @staticmethod
    def _resolve_urdf_path(urdf_path: str | Path | None) -> Path | None:
        path = Path(urdf_path) if urdf_path is not None else default_so101_urdf_path()
        return path if path.exists() else None

    @staticmethod
    def _extract_obs(result):
        if isinstance(result, tuple):
            return result[0]
        return result
