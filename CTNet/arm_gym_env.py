import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib
import os

# Backend selection:
# - Respect user-set MPLBACKEND
# - If on headless *nix (no DISPLAY), fall back to 'Agg'
# - Otherwise, keep default interactive backend (TkAgg/QtAgg, etc.)
if not os.environ.get("MPLBACKEND"):
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces


@dataclass
class ArmConfig:
    step_size: float = 0.03
    max_steps: int = 50
    y_limits: Tuple[float, float] = (-1.0, 1.0)
    z_limits: Tuple[float, float] = (-1.0, 1.0)
    start_y: Optional[float] = None
    start_z: Optional[float] = None


class SimpleArm2DEnv(gym.Env):
    """A minimal Gymnasium environment for a 2D end-effector in YZ plane.

    - Action space: Discrete(4) [0:left, 1:right, 2:up, 3:down]
    - Observation: np.array([y, z], dtype=np.float32)
    - Reward: always 0 (visualization-only)
    - Termination: after max_steps
    - Render: matplotlib YZ plane, returns RGB array when render_mode='rgb_array'
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, config: Optional[ArmConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or ArmConfig()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        low = np.array([self.cfg.y_limits[0], self.cfg.z_limits[0]], dtype=np.float32)
        high = np.array([self.cfg.y_limits[1], self.cfg.z_limits[1]], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._y = 0.0
        self._z = 0.0
        self._step = 0

        # Lazy-constructed figure
        self._fig = None
        self._ax = None
        self._pred_line = None
        self._marker = None
        self._path_y = [0.0]
        self._path_z = [0.0]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._y = float(self.cfg.start_y) if self.cfg.start_y is not None else 0.0
        self._z = float(self.cfg.start_z) if self.cfg.start_z is not None else 0.0
        self._step = 0
        self._path_y = [0.0]
        self._path_z = [0.0]
        # remember home pose (center) for pulse mode
        self._home_y = self._y
        self._home_z = self._z
        obs = np.array([self._y, self._z], dtype=np.float32)
        info = {}
        if self.render_mode == "human":
            self._ensure_fig()
            self._draw()
            plt.pause(0.001)
        return obs, info

    def step(self, action: int):
        # Map actions to displacements
        # Convention: x+ 前, y+ 左, z+ 上
        # 0:left  -> +y, 1:right -> -y, 2:up -> +z, 3:down -> -z
        dy = (self.cfg.step_size if action == 0 else -self.cfg.step_size if action == 1 else 0.0)
        dz = (self.cfg.step_size if action == 2 else -self.cfg.step_size if action == 3 else 0.0)

        # Update state with clamping
        self._y = float(np.clip(self._y + dy, *self.cfg.y_limits))
        self._z = float(np.clip(self._z + dz, *self.cfg.z_limits))
        self._path_y.append(self._y)
        self._path_z.append(self._z)
        self._step += 1

        terminated = self._step >= self.cfg.max_steps
        truncated = False
        reward = 0.0
        obs = np.array([self._y, self._z], dtype=np.float32)
        info = {"y": self._y, "z": self._z, "dy": dy, "dz": dz}

        if self.render_mode == "human":
            self._ensure_fig()
            self._draw()
            plt.pause(0.001)

        if self.render_mode == "rgb_array":
            # Gymnasium expects render() to be called separately; we just keep state ready
            pass

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return
        self._ensure_fig()
        self._draw()
        if self.render_mode == "human":
            plt.pause(0.001)
            return None
        elif self.render_mode == "rgb_array":
            # Draw the canvas and fetch RGBA buffer, then drop alpha for RGB.
            self._fig.canvas.draw()
            try:
                # Preferred: use buffer_rgba (works across recent matplotlib)
                rgba = np.asarray(self._fig.canvas.buffer_rgba())
                image = rgba[..., :3].copy()
            except Exception:
                # Fallbacks for older/newer backends
                try:
                    buf = self._fig.canvas.tostring_rgb()
                    w, h = self._fig.canvas.get_width_height()
                    image = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
                except Exception:
                    # Last resort: ARGB -> RGB
                    buf = self._fig.canvas.tostring_argb()
                    w, h = self._fig.canvas.get_width_height()
                    argb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                    rgb = np.empty((h, w, 3), dtype=np.uint8)
                    # ARGB -> RGB: drop alpha and roll channels
                    rgb[..., 0] = argb[..., 1]
                    rgb[..., 1] = argb[..., 2]
                    rgb[..., 2] = argb[..., 3]
                    image = rgb
            return image

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._pred_line = None
            self._marker = None

    # --- helpers ---
    def _ensure_fig(self):
        if self._fig is not None:
            return
        self._fig, self._ax = plt.subplots(figsize=(5, 5))
        self._ax.set_title("SimpleArm2DEnv (YZ plane)")
        self._ax.set_xlabel("Y")
        self._ax.set_ylabel("Z")
        self._ax.grid(alpha=0.3)
        self._ax.set_xlim(self.cfg.y_limits)
        self._ax.set_ylim(self.cfg.z_limits)
        (self._pred_line,) = self._ax.plot([], [], "-o", color="#1f77b4", label="Path")
        (self._marker,) = self._ax.plot([], [], "o", color="#d62728", markersize=10)
        self._ax.legend()
        if self.render_mode == "human":
            try:
                plt.show(block=False)
            except Exception:
                pass

    def _draw(self):
        if self._fig is None:
            return
        self._pred_line.set_data(self._path_y, self._path_z)
        self._marker.set_data([self._y], [self._z])
        self._fig.tight_layout()

    # public helper for pulse mode
    def center(self):
        """Return the marker to the home pose and update the plot."""
        self._y = float(np.clip(getattr(self, "_home_y", 0.0), *self.cfg.y_limits))
        self._z = float(np.clip(getattr(self, "_home_z", 0.0), *self.cfg.z_limits))
        self._path_y.append(self._y)
        self._path_z.append(self._z)
        if self.render_mode in ("human", "rgb_array"):
            self.render()

    def get_status(self):
        return {
            "ee_xyz": (float('nan'), float(self._y), float(self._z)),
            "target_yz": (float(self._y), float(self._z)),
            "step": int(self._step),
            "max_steps": int(self.cfg.max_steps),
        }
