import unittest
from unittest.mock import patch

import numpy as np

from arm_visualizer import ArmVisualizer, default_so101_urdf_path


class FakeConfig:
    def __init__(self, step_size, use_gui, **kwargs):
        self.step_size = step_size
        self.use_gui = use_gui
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeEnv:
    def __init__(self, config, render_mode):
        self.config = config
        self.render_mode = render_mode
        self.y = 0.0
        self.z = 0.0
        self.closed = False

    def reset(self):
        self.y = 0.0
        self.z = 0.0
        return np.array([self.y, self.z], dtype=float)

    def step(self, action):
        if action == 0:
            self.y += self.config.step_size
        elif action == 1:
            self.y -= self.config.step_size
        elif action == 2:
            self.z += self.config.step_size
        elif action == 3:
            self.z -= self.config.step_size
        return np.array([self.y, self.z], dtype=float), 0.0, False, False, {}

    def render(self):
        return np.zeros((12, 16, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


class FailingEnv:
    def __init__(self, config, render_mode):
        raise RuntimeError("pybullet unavailable")


class StepFailingEnv(FakeEnv):
    def step(self, action):
        raise RuntimeError("step failed")


class TestArmVisualizer(unittest.TestCase):
    def test_fake_env_updates_trajectory_and_frame(self):
        visualizer = ArmVisualizer(env_cls=FakeEnv, cfg_cls=FakeConfig, step_size=0.1)

        visualizer.reset()
        frame = visualizer.step(2)

        self.assertEqual(frame.arm_rgb.shape, (12, 16, 3))
        self.assertEqual(frame.arm_rgb.dtype, np.uint8)
        self.assertEqual(frame.trajectory_yz[-1], (0.0, 0.1))

    def test_default_so101_urdf_exists(self):
        self.assertTrue(default_so101_urdf_path().exists())

    def test_passes_so101_urdf_and_camera_controls_to_pybullet_config(self):
        visualizer = ArmVisualizer(
            env_cls=FakeEnv,
            cfg_cls=FakeConfig,
            step_size=0.1,
            camera_yaw=120.0,
            camera_pitch=-20.0,
            camera_distance=1.4,
            camera_target=(0.1, 0.2, 0.3),
        )

        visualizer.reset()
        config = visualizer.env.config

        self.assertEqual(config.urdf_path, str(default_so101_urdf_path()))
        self.assertEqual(config.camera_yaw, 120.0)
        self.assertEqual(config.camera_pitch, -20.0)
        self.assertEqual(config.camera_distance, 1.4)
        self.assertEqual(config.camera_target, (0.1, 0.2, 0.3))
        self.assertEqual(config.arm_model, "SO-101")

    def test_fallback_without_env_still_tracks_actions(self):
        visualizer = ArmVisualizer(env_cls=FailingEnv, cfg_cls=FakeConfig, step_size=0.03)

        visualizer.reset()
        frame = visualizer.step(0)

        self.assertIsNone(frame.arm_rgb)
        self.assertEqual(frame.trajectory_yz[-1], (0.03, 0.0))
        self.assertEqual(frame.status["arm_mode"], "fallback")

    def test_loader_failure_uses_fallback_frame(self):
        visualizer = ArmVisualizer()

        with patch("arm_visualizer.load_pybullet_env", side_effect=ImportError("missing pybullet")):
            frame = visualizer.reset()

        self.assertIsNone(frame.arm_rgb)
        self.assertEqual(frame.status["arm_mode"], "fallback")
        self.assertIn("missing pybullet", frame.status["error"])

    def test_invalid_action_raises_without_changing_trajectory(self):
        visualizer = ArmVisualizer(env_cls=FakeEnv, cfg_cls=FakeConfig, step_size=0.1)
        visualizer.reset()
        trajectory_len = len(visualizer.trajectory_yz)

        with self.assertRaises(ValueError):
            visualizer.step(99)

        self.assertEqual(len(visualizer.trajectory_yz), trajectory_len)

    def test_step_failure_degrades_to_fallback_frame(self):
        visualizer = ArmVisualizer(env_cls=StepFailingEnv, cfg_cls=FakeConfig, step_size=0.1)
        visualizer.reset()

        frame = visualizer.step(2)

        self.assertIsNone(frame.arm_rgb)
        self.assertEqual(frame.status["arm_mode"], "fallback")
        self.assertIn("step failed", frame.status["error"])
        self.assertIsNone(visualizer.env)


if __name__ == "__main__":
    unittest.main()
