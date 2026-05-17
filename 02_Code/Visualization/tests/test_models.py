import unittest

import numpy as np

from models import (
    ActionDecision,
    DashboardFrame,
    SourceEpoch,
    build_action_decision,
    class_to_action,
)


class TestDashboardModels(unittest.TestCase):
    def test_class_to_action_mapping(self):
        self.assertEqual(class_to_action(0), (0, "left"))
        self.assertEqual(class_to_action(1), (1, "right"))
        self.assertEqual(class_to_action(2), (2, "up"))
        self.assertEqual(class_to_action(3), (3, "down"))

        with self.assertRaises(ValueError):
            class_to_action(4)

    def test_action_decision_keeps_ctnet_and_scripted_actions_separate(self):
        decision = build_action_decision(pred_class=2, scripted_demo_action=1)

        self.assertIsInstance(decision, ActionDecision)
        self.assertEqual(decision.ctnet_predicted_action, 2)
        self.assertEqual(decision.ctnet_predicted_action_name, "up")
        self.assertEqual(decision.scripted_demo_action, 1)
        self.assertEqual(decision.scripted_demo_action_name, "right")
        self.assertEqual(decision.executed_action, 1)
        self.assertEqual(decision.executed_action_name, "right")
        self.assertEqual(decision.executed_action_source, "scripted demo")

    def test_action_decision_uses_ctnet_when_no_scripted_action(self):
        decision = build_action_decision(pred_class=3)

        self.assertEqual(decision.ctnet_predicted_action, 3)
        self.assertEqual(decision.ctnet_predicted_action_name, "down")
        self.assertIsNone(decision.scripted_demo_action)
        self.assertIsNone(decision.scripted_demo_action_name)
        self.assertEqual(decision.executed_action, 3)
        self.assertEqual(decision.executed_action_name, "down")
        self.assertEqual(decision.executed_action_source, "CTNet prediction")

    def test_dashboard_frame_can_represent_offline_replay(self):
        raw_eeg = np.zeros((64, 160), dtype=float)
        preprocessed = np.ones((1, 64, 160), dtype=float)
        arm_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        trajectory_yz = [(0.0, 0.0), (0.1, 0.2)]

        frame = DashboardFrame(
            mode="offline replay",
            raw_eeg=raw_eeg,
            preprocessed_eeg_for_display=preprocessed,
            model_input_shape=(1, 1, 64, 160),
            sampling_rate=160.0,
            channel_names=["C3", "C4"],
            pred_class=2,
            pred_name="Hands/Up",
            probabilities=np.array([0.1, 0.2, 0.6, 0.1]),
            confidence=0.6,
            true_label=2,
            true_name="Hands/Up",
            ctnet_predicted_action=2,
            ctnet_predicted_action_name="up",
            scripted_demo_action=1,
            scripted_demo_action_name="right",
            executed_action=1,
            executed_action_name="right",
            executed_action_source="scripted demo",
            arm_rgb=arm_rgb,
            trajectory_yz=trajectory_yz,
            replay_index=3,
            replay_total=8,
            status={"source": "fixture"},
        )

        self.assertIs(frame.raw_eeg, raw_eeg)
        self.assertEqual(frame.mode, "offline replay")
        self.assertEqual(frame.replay_index, 3)
        self.assertEqual(frame.replay_total, 8)
        self.assertEqual(frame.status["source"], "fixture")

    def test_source_epoch_carries_replay_metadata(self):
        raw_eeg = np.zeros((64, 160), dtype=float)

        epoch = SourceEpoch(
            raw_eeg=raw_eeg,
            sampling_rate=160.0,
            channel_names=["C3", "C4"],
            true_label=1,
            subject=4,
            epoch_index=12,
            replay_index=5,
            replay_total=20,
        )

        self.assertIs(epoch.raw_eeg, raw_eeg)
        self.assertEqual(epoch.replay_index, 5)
        self.assertEqual(epoch.replay_total, 20)


if __name__ == "__main__":
    unittest.main()
