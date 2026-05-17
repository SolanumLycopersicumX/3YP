import unittest

import numpy as np

from data_sources import OfflinePhysioNetSource, SyntheticBrainFlowSource


def fake_physionet_loader(subject):
    data = np.arange(5 * 3 * 8, dtype=np.float32).reshape(5, 3, 8)
    labels = np.array([0, 1, 2, 3, 0])
    return data, labels


class FakeStream:
    def __init__(self):
        self.sampling_rate = 250
        self.n_eeg_channels = 2
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def get_eeg_epoch(self, duration_sec=0.5, apply_filter=False):
        return np.ones((2, 125), dtype=np.float32)


class TestDataSources(unittest.TestCase):
    def test_offline_source_steps_through_range(self):
        source = OfflinePhysioNetSource(
            subject=4,
            start_epoch=1,
            stop_epoch=3,
            loader=fake_physionet_loader,
        )

        current = source.current()
        self.assertEqual(current.true_label, 1)
        self.assertEqual(current.epoch_index, 1)
        self.assertEqual(current.replay_index, 0)
        self.assertEqual(current.replay_total, 3)

        stepped = source.step()
        self.assertEqual(stepped.true_label, 2)
        self.assertEqual(stepped.replay_index, 1)

    def test_offline_reset_returns_to_first_epoch(self):
        source = OfflinePhysioNetSource(
            subject=4,
            start_epoch=1,
            stop_epoch=3,
            loader=fake_physionet_loader,
        )

        source.step()
        reset = source.reset()

        self.assertEqual(reset.epoch_index, 1)

    def test_synthetic_source_uses_stream_without_true_label(self):
        stream = FakeStream()
        source = SyntheticBrainFlowSource(
            stream_factory=lambda: stream,
            duration_sec=0.5,
        )

        epoch = source.step()

        self.assertTrue(stream.started)
        self.assertEqual(epoch.raw_eeg.shape, (2, 125))
        self.assertIsNone(epoch.true_label)
        self.assertEqual(epoch.replay_index, 0)

        source.stop()

        self.assertFalse(stream.started)


if __name__ == "__main__":
    unittest.main()
