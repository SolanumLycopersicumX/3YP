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


class ArmVisualizer:
    def __init__(self, env_cls=None, cfg_cls=None, step_size=0.03):
        self.env_cls = env_cls
        self.cfg_cls = cfg_cls
        self.step_size = step_size
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
            config = self.cfg_cls(step_size=self.step_size, use_gui=False)
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
    def _extract_obs(result):
        if isinstance(result, tuple):
            return result[0]
        return result
