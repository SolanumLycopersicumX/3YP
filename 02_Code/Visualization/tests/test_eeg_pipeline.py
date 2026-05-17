import unittest

import numpy as np
import torch

from eeg_pipeline import (
    EEGPipeline,
    adapt_channels,
    create_model_input,
    preprocess_for_display,
    resample_time,
)


class FakeModel(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        logits = torch.tensor([[0.1, 3.0, 0.2, -1.0]], dtype=torch.float32)
        return logits.repeat(batch_size, 1)


class TestEEGPipeline(unittest.TestCase):
    def test_preprocess_for_display_preserves_shape(self):
        rng = np.random.default_rng(123)
        raw = rng.normal(size=(4, 320))

        processed = preprocess_for_display(raw, sampling_rate=160)

        self.assertEqual(processed.shape, raw.shape)
        self.assertEqual(processed.dtype, np.float32)

    def test_resample_time_changes_only_time_axis(self):
        data = np.arange(30, dtype=np.float32).reshape(3, 10)

        resampled = resample_time(data, 25)

        self.assertEqual(resampled.shape, (3, 25))

    def test_adapt_channels_pads_and_trims(self):
        padded = adapt_channels(np.ones((2, 5), dtype=np.float32), 4)
        trimmed = adapt_channels(np.ones((6, 5), dtype=np.float32), 4)

        self.assertEqual(padded.shape, (4, 5))
        np.testing.assert_array_equal(padded[:2], np.ones((2, 5), dtype=np.float32))
        np.testing.assert_array_equal(padded[2:], np.zeros((2, 5), dtype=np.float32))
        self.assertEqual(trimmed.shape, (4, 5))

    def test_create_model_input_uses_metadata_normalization(self):
        raw = np.full((2, 4), 3.0, dtype=np.float32)

        model_input = create_model_input(
            raw,
            target_channels=4,
            target_samples=8,
            norm_mean=1.0,
            norm_std=2.0,
        )

        self.assertEqual(model_input.shape, (1, 1, 4, 8))
        self.assertEqual(model_input.dtype, np.float32)
        self.assertAlmostEqual(float(model_input[0, 0, 0, 0]), 1.0)

    def test_predict_with_fake_model_returns_class_probability_and_action(self):
        pipeline = EEGPipeline(
            model=FakeModel(),
            target_channels=4,
            target_samples=8,
        )

        result = pipeline.predict(np.zeros((4, 8), dtype=np.float32))

        self.assertEqual(result.pred_class, 1)
        self.assertEqual(result.pred_name, "Right")
        self.assertEqual(result.ctnet_predicted_action, 1)
        self.assertEqual(result.ctnet_predicted_action_name, "right")
        self.assertEqual(result.probabilities.shape, (4,))
        self.assertGreater(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
