import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from plotting import (
    export_csv,
    export_jsonl,
    make_eeg_figure,
    make_probability_figure,
    make_trajectory_figure,
)


class TestPlotting(unittest.TestCase):
    def test_eeg_figure_contains_axes(self):
        raw = np.zeros((3, 100), dtype=float)
        preprocessed = np.ones((3, 100), dtype=float)

        fig = make_eeg_figure(
            raw,
            preprocessed,
            sampling_rate=100.0,
            channel_names=["C1", "C2", "C3"],
            max_channels=2,
        )

        self.assertEqual(len(fig.axes), 2)

    def test_eeg_figure_rejects_shape_mismatch(self):
        raw = np.zeros((3, 100), dtype=float)
        preprocessed = np.ones((2, 100), dtype=float)

        with self.assertRaises(ValueError):
            make_eeg_figure(
                raw,
                preprocessed,
                sampling_rate=100.0,
                channel_names=["C1", "C2", "C3"],
            )

    def test_trajectory_figure_contains_one_axis(self):
        fig = make_trajectory_figure([(0.0, 0.0), (0.2, 0.1)])

        self.assertEqual(len(fig.axes), 1)

    def test_probability_figure_contains_one_axis(self):
        fig = make_probability_figure(np.array([0.1, 0.2, 0.3, 0.4]))

        self.assertEqual(len(fig.axes), 1)

    def test_export_jsonl_writes_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "records.jsonl"

            export_jsonl(path, [{"label": "Left", "confidence": 0.8}])

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records, [{"label": "Left", "confidence": 0.8}])

    def test_export_csv_writes_union_headers_in_first_seen_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "records.csv"

            export_csv(
                path,
                [
                    {"label": "Left", "confidence": 0.8},
                    {"epoch": 2, "label": "Right"},
                ],
            )

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0], {"label": "Left", "confidence": "0.8", "epoch": ""})
            self.assertEqual(rows[1], {"label": "Right", "confidence": "", "epoch": "2"})


if __name__ == "__main__":
    unittest.main()
