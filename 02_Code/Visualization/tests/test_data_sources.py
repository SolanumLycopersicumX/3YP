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


class FailingStartStream:
    def __init__(self):
        self.stop_called = False

    def start(self):
        raise RuntimeError("start failed")

    def stop(self):
        self.stop_called = True


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

    def test_offline_source_stays_at_last_selected_epoch(self):
        source = OfflinePhysioNetSource(
            subject=4,
            start_epoch=1,
            stop_epoch=2,
            loader=fake_physionet_loader,
        )

        source.step()
        at_end = source.step()
        still_at_end = source.step()

        self.assertEqual(at_end.epoch_index, 2)
        self.assertEqual(still_at_end.epoch_index, 2)
        self.assertEqual(still_at_end.replay_index, 1)

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

    def test_offline_source_rejects_invalid_epoch_range(self):
        invalid_ranges = [
            {"start_epoch": -1, "stop_epoch": 2},
            {"start_epoch": 3, "stop_epoch": 1},
            {"start_epoch": 0, "stop_epoch": 5},
        ]

        for kwargs in invalid_ranges:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    OfflinePhysioNetSource(
                        subject=4,
                        loader=fake_physionet_loader,
                        **kwargs,
                    )

    def test_offline_source_rejects_empty_data_dimensions(self):
        shapes = [(0, 3, 8), (5, 0, 8), (5, 3, 0)]

        for shape in shapes:
            with self.subTest(shape=shape):
                data = np.zeros(shape, dtype=np.float32)
                labels = np.zeros(shape[0], dtype=np.int64)

                with self.assertRaises(ValueError):
                    OfflinePhysioNetSource(
                        subject=4,
                        loader=lambda subject, data=data, labels=labels: (data, labels),
                    )

    def test_offline_source_rejects_malformed_data_and_labels(self):
        cases = [
            (np.zeros((5, 3), dtype=np.float32), np.zeros(5, dtype=np.int64)),
            (np.zeros((5, 3, 8), dtype=np.float32), np.zeros(4, dtype=np.int64)),
            (np.zeros((5, 3, 8), dtype=np.float32), np.zeros(6, dtype=np.int64)),
            (np.zeros((5, 3, 8), dtype=np.float32), np.zeros((5, 2), dtype=np.int64)),
        ]

        for data, labels in cases:
            with self.subTest(data_shape=data.shape, labels_shape=labels.shape):
                with self.assertRaises(ValueError):
                    OfflinePhysioNetSource(
                        subject=4,
                        loader=lambda subject, data=data, labels=labels: (data, labels),
                    )

    def test_offline_source_flattens_single_column_labels(self):
        data = np.arange(5 * 3 * 8, dtype=np.float32).reshape(5, 3, 8)
        labels = np.array([[0], [1], [2], [3], [0]])

        source = OfflinePhysioNetSource(
            subject=4,
            start_epoch=2,
            loader=lambda subject: (data, labels),
        )

        self.assertEqual(source.current().true_label, 2)

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

    def test_synthetic_source_clears_stream_when_start_fails(self):
        stream = FailingStartStream()
        source = SyntheticBrainFlowSource(stream_factory=lambda: stream)

        with self.assertRaises(RuntimeError):
            source.start()

        self.assertIsNone(source.stream)
        self.assertTrue(stream.stop_called)


if __name__ == "__main__":
    unittest.main()
