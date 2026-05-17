# EEG Arm Observer Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit dashboard that simultaneously shows raw EEG versus preprocessed EEG, PyBullet arm simulation, Y-Z trajectory, and CTNet classification/action results for offline PhysioNet replay and BrainFlow synthetic streaming.

**Architecture:** Add a focused `02_Code/Visualization/` package with separate modules for shared models, EEG preprocessing/classification, data sources, arm simulation, plotting/export, and the Streamlit app. The dashboard reuses existing PhysioNet loading, OpenBCI/BrainFlow streaming, and PyBullet environment code through thin wrappers so existing training and control scripts stay untouched.

**Tech Stack:** Python 3.10, Streamlit, NumPy, SciPy, Torch, Matplotlib, MNE, BrainFlow, PyBullet, standard-library `unittest`.

---

## Source Spec

Design source: `docs/superpowers/specs/2026-05-17-eeg-arm-observer-dashboard-design.md`

## Scope Check

This is one coherent subsystem: a local observer dashboard. The two data modes share the same frame model, EEG pipeline, plotting helpers, and arm visualizer, so they belong in one implementation plan rather than separate plans.

## File Structure

- Create `02_Code/Visualization/__init__.py`: package marker and short module description.
- Create `02_Code/Visualization/models.py`: shared constants, dataclasses, class/action mapping, and action-source selection.
- Create `02_Code/Visualization/eeg_pipeline.py`: visible EEG preprocessing, model-input creation, CTNet model loading, and prediction.
- Create `02_Code/Visualization/data_sources.py`: offline PhysioNet replay source and BrainFlow synthetic source with a common API.
- Create `02_Code/Visualization/arm_visualizer.py`: PyBullet `rgb_array` wrapper with trajectory state and fallback behavior.
- Create `02_Code/Visualization/plotting.py`: Matplotlib figures and JSON/CSV export helpers.
- Create `02_Code/Visualization/dashboard_app.py`: Streamlit UI, sidebar controls, session state, and panel rendering.
- Create `02_Code/Visualization/README.md`: usage, dependencies, modes, and limitations.
- Create `02_Code/Visualization/tests/test_models.py`: mapping and frame tests.
- Create `02_Code/Visualization/tests/test_eeg_pipeline.py`: preprocessing and fake-model inference tests.
- Create `02_Code/Visualization/tests/test_data_sources.py`: replay and synthetic-source wrapper tests using fakes.
- Create `02_Code/Visualization/tests/test_arm_visualizer.py`: trajectory and fallback tests using a fake environment.
- Create `02_Code/Visualization/tests/test_plotting.py`: figure and export tests.
- Modify `02_Code/Utils/requirements.txt`: add Streamlit and BrainFlow dashboard dependencies.
- Modify `README.md`: add one quick-start line for the dashboard.

## Task 1: Shared Models and Action Mapping

**Files:**
- Create: `02_Code/Visualization/__init__.py`
- Create: `02_Code/Visualization/models.py`
- Create: `02_Code/Visualization/tests/test_models.py`

- [ ] **Step 1: Write failing model tests**

Create `02_Code/Visualization/tests/test_models.py`:

```python
import unittest

import numpy as np

from models import (
    CLASS_NAMES,
    DashboardFrame,
    SourceEpoch,
    build_action_decision,
    class_to_action,
)


class ModelTests(unittest.TestCase):
    def test_class_to_action_mapping(self):
        self.assertEqual(class_to_action(0), (0, "left"))
        self.assertEqual(class_to_action(1), (1, "right"))
        self.assertEqual(class_to_action(2), (2, "up"))
        self.assertEqual(class_to_action(3), (3, "down"))
        with self.assertRaises(ValueError):
            class_to_action(4)

    def test_action_decision_keeps_ctnet_and_scripted_actions_separate(self):
        decision = build_action_decision(pred_class=2, scripted_demo_action=1)
        self.assertEqual(decision.ctnet_predicted_action, 2)
        self.assertEqual(decision.ctnet_predicted_action_name, "up")
        self.assertEqual(decision.scripted_demo_action, 1)
        self.assertEqual(decision.scripted_demo_action_name, "right")
        self.assertEqual(decision.executed_action, 1)
        self.assertEqual(decision.executed_action_name, "right")
        self.assertEqual(decision.executed_action_source, "scripted demo")

    def test_action_decision_uses_ctnet_when_no_scripted_action(self):
        decision = build_action_decision(pred_class=3, scripted_demo_action=None)
        self.assertEqual(decision.ctnet_predicted_action, 3)
        self.assertIsNone(decision.scripted_demo_action)
        self.assertEqual(decision.executed_action, 3)
        self.assertEqual(decision.executed_action_source, "CTNet prediction")

    def test_dashboard_frame_can_represent_offline_replay(self):
        raw = np.zeros((2, 4), dtype=np.float32)
        frame = DashboardFrame(
            mode="offline",
            raw_eeg=raw,
            preprocessed_eeg_for_display=raw.copy(),
            model_input_shape=(1, 1, 64, 1000),
            sampling_rate=160.0,
            channel_names=["C3", "C4"],
            pred_class=0,
            pred_name=CLASS_NAMES[0],
            probabilities=np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32),
            confidence=0.7,
            true_label=0,
            true_name=CLASS_NAMES[0],
            ctnet_predicted_action=0,
            ctnet_predicted_action_name="left",
            scripted_demo_action=None,
            scripted_demo_action_name=None,
            executed_action=0,
            executed_action_name="left",
            executed_action_source="CTNet prediction",
            arm_rgb=None,
            trajectory_yz=[(0.0, 0.0)],
            replay_index=0,
            replay_total=3,
            status={"source": "S001"},
        )
        self.assertEqual(frame.mode, "offline")
        self.assertEqual(frame.replay_total, 3)

    def test_source_epoch_carries_replay_metadata(self):
        epoch = SourceEpoch(
            raw_eeg=np.ones((2, 8), dtype=np.float32),
            sampling_rate=160.0,
            channel_names=["C3", "C4"],
            true_label=1,
            subject=1,
            epoch_index=5,
            replay_index=2,
            replay_total=4,
            status={"mode": "offline"},
        )
        self.assertEqual(epoch.replay_index, 2)
        self.assertEqual(epoch.replay_total, 4)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing model tests**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m unittest 02_Code/Visualization/tests/test_models.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'models'`.

- [ ] **Step 3: Implement shared models**

Create `02_Code/Visualization/__init__.py`:

```python
"""Visualization dashboard package for EEG and robotic-arm observation."""
```

Create `02_Code/Visualization/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


CLASS_NAMES = ["Left", "Right", "Hands/Up", "Feet/Down"]
ACTION_NAMES = ["left", "right", "up", "down"]
CLASS_TO_ACTION = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
}
SCRIPTED_ACTION_SEQUENCE = [0, 2, 1, 3, 0, 1, 2, 3]


@dataclass(frozen=True)
class SourceEpoch:
    raw_eeg: np.ndarray
    sampling_rate: float
    channel_names: list[str]
    true_label: int | None
    subject: int | None
    epoch_index: int | None
    replay_index: int | None
    replay_total: int | None
    status: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionDecision:
    ctnet_predicted_action: int | None
    ctnet_predicted_action_name: str | None
    scripted_demo_action: int | None
    scripted_demo_action_name: str | None
    executed_action: int | None
    executed_action_name: str | None
    executed_action_source: str


@dataclass(frozen=True)
class DashboardFrame:
    mode: str
    raw_eeg: np.ndarray
    preprocessed_eeg_for_display: np.ndarray
    model_input_shape: tuple[int, ...]
    sampling_rate: float
    channel_names: list[str]
    pred_class: int | None
    pred_name: str | None
    probabilities: np.ndarray | None
    confidence: float | None
    true_label: int | None
    true_name: str | None
    ctnet_predicted_action: int | None
    ctnet_predicted_action_name: str | None
    scripted_demo_action: int | None
    scripted_demo_action_name: str | None
    executed_action: int | None
    executed_action_name: str | None
    executed_action_source: str
    arm_rgb: np.ndarray | None
    trajectory_yz: list[tuple[float, float]]
    replay_index: int | None
    replay_total: int | None
    status: dict[str, Any] = field(default_factory=dict)


def action_name(action: int | None) -> str | None:
    if action is None:
        return None
    if action < 0 or action >= len(ACTION_NAMES):
        raise ValueError(f"Unsupported action: {action}")
    return ACTION_NAMES[action]


def class_to_action(pred_class: int) -> tuple[int, str]:
    if pred_class not in CLASS_TO_ACTION:
        raise ValueError(f"Unsupported class: {pred_class}")
    action = CLASS_TO_ACTION[pred_class]
    return action, action_name(action) or "unknown"


def class_name(label: int | None) -> str | None:
    if label is None:
        return None
    if label < 0 or label >= len(CLASS_NAMES):
        raise ValueError(f"Unsupported class label: {label}")
    return CLASS_NAMES[label]


def scripted_action_for_step(step_index: int) -> int:
    return SCRIPTED_ACTION_SEQUENCE[step_index % len(SCRIPTED_ACTION_SEQUENCE)]


def build_action_decision(
    pred_class: int | None,
    scripted_demo_action: int | None = None,
) -> ActionDecision:
    ctnet_action = None
    ctnet_action_name = None
    if pred_class is not None:
        ctnet_action, ctnet_action_name = class_to_action(pred_class)

    scripted_name = action_name(scripted_demo_action)
    if scripted_demo_action is not None:
        executed_action = scripted_demo_action
        executed_action_name = scripted_name
        executed_source = "scripted demo"
    else:
        executed_action = ctnet_action
        executed_action_name = ctnet_action_name
        executed_source = "CTNet prediction" if ctnet_action is not None else "none"

    return ActionDecision(
        ctnet_predicted_action=ctnet_action,
        ctnet_predicted_action_name=ctnet_action_name,
        scripted_demo_action=scripted_demo_action,
        scripted_demo_action_name=scripted_name,
        executed_action=executed_action,
        executed_action_name=executed_action_name,
        executed_action_source=executed_source,
    )
```

- [ ] **Step 4: Run model tests**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m unittest 02_Code/Visualization/tests/test_models.py -v
```

Expected: PASS, 5 tests.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add 02_Code/Visualization/__init__.py 02_Code/Visualization/models.py 02_Code/Visualization/tests/test_models.py
git commit -m "Add dashboard shared models"
```

## Task 2: EEG Pipeline and CTNet Prediction

**Files:**
- Create: `02_Code/Visualization/eeg_pipeline.py`
- Create: `02_Code/Visualization/tests/test_eeg_pipeline.py`

- [ ] **Step 1: Write failing EEG pipeline tests**

Create `02_Code/Visualization/tests/test_eeg_pipeline.py`:

```python
import unittest

import numpy as np

from eeg_pipeline import (
    EEGPipeline,
    adapt_channels,
    create_model_input,
    preprocess_for_display,
    resample_time,
)


class FakeModel:
    def eval(self):
        return self

    def __call__(self, batch):
        import torch

        logits = torch.tensor([[0.1, 2.0, 0.2, 0.3]], dtype=torch.float32)
        features = torch.zeros((1, 4), dtype=torch.float32)
        return features, logits


class EEGPipelineTests(unittest.TestCase):
    def test_preprocess_for_display_preserves_shape(self):
        rng = np.random.default_rng(7)
        raw = rng.normal(size=(4, 320)).astype(np.float32)
        filtered = preprocess_for_display(raw, sampling_rate=160.0)
        self.assertEqual(filtered.shape, raw.shape)
        self.assertEqual(filtered.dtype, np.float32)

    def test_resample_time_changes_only_time_axis(self):
        data = np.ones((3, 10), dtype=np.float32)
        out = resample_time(data, target_samples=25)
        self.assertEqual(out.shape, (3, 25))

    def test_adapt_channels_pads_and_trims(self):
        small = np.ones((2, 5), dtype=np.float32)
        padded = adapt_channels(small, target_channels=4)
        self.assertEqual(padded.shape, (4, 5))
        self.assertTrue(np.allclose(padded[2:], 0.0))

        large = np.ones((6, 5), dtype=np.float32)
        trimmed = adapt_channels(large, target_channels=4)
        self.assertEqual(trimmed.shape, (4, 5))

    def test_create_model_input_uses_metadata_normalization(self):
        raw = np.full((2, 10), 3.0, dtype=np.float32)
        model_input = create_model_input(
            raw,
            target_channels=4,
            target_samples=8,
            norm_mean=1.0,
            norm_std=2.0,
        )
        self.assertEqual(model_input.shape, (1, 1, 4, 8))
        self.assertAlmostEqual(float(model_input[0, 0, 0, 0]), 1.0, places=5)

    def test_predict_with_fake_model_returns_class_probability_and_action(self):
        pipeline = EEGPipeline(model=FakeModel(), norm_mean=0.0, norm_std=1.0)
        result = pipeline.predict(np.zeros((64, 1000), dtype=np.float32))
        self.assertEqual(result.pred_class, 1)
        self.assertEqual(result.pred_name, "Right")
        self.assertEqual(result.ctnet_predicted_action, 1)
        self.assertEqual(result.ctnet_predicted_action_name, "right")
        self.assertEqual(result.probabilities.shape, (4,))
        self.assertGreater(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing EEG pipeline tests**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m unittest 02_Code/Visualization/tests/test_eeg_pipeline.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'eeg_pipeline'`.

- [ ] **Step 3: Implement EEG preprocessing and prediction**

Create `02_Code/Visualization/eeg_pipeline.py`:

```python
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, resample, sosfiltfilt

from models import build_action_decision, class_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "04_Trained_Models/physionet_ctnet_109sub/pool/model_pool.pth"
DEFAULT_TARGET_CHANNELS = 64
DEFAULT_TARGET_SAMPLES = 1000


@dataclass(frozen=True)
class PredictionResult:
    pred_class: int
    pred_name: str
    probabilities: np.ndarray
    confidence: float
    ctnet_predicted_action: int
    ctnet_predicted_action_name: str
    model_input_shape: tuple[int, ...]


def preprocess_for_display(
    raw_eeg: np.ndarray,
    sampling_rate: float,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
) -> np.ndarray:
    raw = np.asarray(raw_eeg, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Expected EEG shape (channels, samples), got {raw.shape}")
    nyquist = sampling_rate / 2.0
    if l_freq <= 0 or h_freq >= nyquist or l_freq >= h_freq:
        return raw.copy()
    sos = butter(4, [l_freq / nyquist, h_freq / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, raw, axis=1).astype(np.float32)


def resample_time(data: np.ndarray, target_samples: int) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.shape[1] == target_samples:
        return arr.copy()
    return resample(arr, target_samples, axis=1).astype(np.float32)


def adapt_channels(data: np.ndarray, target_channels: int) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    channels, samples = arr.shape
    if channels == target_channels:
        return arr.copy()
    if channels > target_channels:
        return arr[:target_channels].copy()
    pad = np.zeros((target_channels - channels, samples), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def create_model_input(
    preprocessed_eeg: np.ndarray,
    target_channels: int = DEFAULT_TARGET_CHANNELS,
    target_samples: int = DEFAULT_TARGET_SAMPLES,
    norm_mean: float = 0.0,
    norm_std: float = 1.0,
) -> np.ndarray:
    adapted = adapt_channels(preprocessed_eeg, target_channels=target_channels)
    resampled = resample_time(adapted, target_samples=target_samples)
    safe_std = norm_std if abs(norm_std) > 1e-12 else 1.0
    normalized = (resampled - norm_mean) / safe_std
    return normalized[np.newaxis, np.newaxis, :, :].astype(np.float32)


def load_norm_metadata(model_path: Path) -> dict[str, Any]:
    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    norm_path = model_path.parent / "norm_params.json"
    if norm_path.exists():
        raw = json.loads(norm_path.read_text(encoding="utf-8"))
        return {"norm_mean": raw.get("mean", 0.0), "norm_std": raw.get("std", 1.0)}

    return {"norm_mean": 0.0, "norm_std": 1.0}


def _prepare_ctnet_unpickle_context() -> None:
    code_root = PROJECT_ROOT / "02_Code"
    eeg_root = code_root / "EEG_Classification"
    for path in (code_root, eeg_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import CTNet_model
        import __main__

        for attr_name in dir(CTNet_model):
            if not attr_name.startswith("_"):
                setattr(__main__, attr_name, getattr(CTNet_model, attr_name))
    except Exception:
        return


def load_ctnet_model(model_path: Path = DEFAULT_MODEL_PATH, device: str = "cpu"):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing CTNet model file: {model_path}")
    _prepare_ctnet_unpickle_context()
    import torch

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model


class EEGPipeline:
    def __init__(
        self,
        model=None,
        model_path: Path = DEFAULT_MODEL_PATH,
        device: str = "cpu",
        norm_mean: float | None = None,
        norm_std: float | None = None,
        target_channels: int = DEFAULT_TARGET_CHANNELS,
        target_samples: int = DEFAULT_TARGET_SAMPLES,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.model = model
        self.target_channels = target_channels
        self.target_samples = target_samples
        metadata = load_norm_metadata(self.model_path)
        self.norm_mean = float(metadata.get("norm_mean", metadata.get("mean", 0.0))) if norm_mean is None else norm_mean
        self.norm_std = float(metadata.get("norm_std", metadata.get("std", 1.0))) if norm_std is None else norm_std

    def ensure_model(self):
        if self.model is None:
            self.model = load_ctnet_model(self.model_path, self.device)
        return self.model

    def make_model_input(self, preprocessed_eeg: np.ndarray) -> np.ndarray:
        return create_model_input(
            preprocessed_eeg,
            target_channels=self.target_channels,
            target_samples=self.target_samples,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
        )

    def predict(self, preprocessed_eeg: np.ndarray) -> PredictionResult:
        import torch

        model_input = self.make_model_input(preprocessed_eeg)
        tensor = torch.from_numpy(model_input).float().to(self.device)
        model = self.ensure_model()
        with torch.no_grad():
            output = model(tensor)
            logits = output[1] if isinstance(output, tuple) else output
            probs_tensor = torch.softmax(logits, dim=1).detach().cpu()[0]
        probabilities = probs_tensor.numpy().astype(np.float32)
        pred_class = int(np.argmax(probabilities))
        decision = build_action_decision(pred_class=pred_class)
        return PredictionResult(
            pred_class=pred_class,
            pred_name=class_name(pred_class) or str(pred_class),
            probabilities=probabilities,
            confidence=float(probabilities[pred_class]),
            ctnet_predicted_action=decision.ctnet_predicted_action or 0,
            ctnet_predicted_action_name=decision.ctnet_predicted_action_name or "unknown",
            model_input_shape=tuple(model_input.shape),
        )
```

- [ ] **Step 4: Run EEG pipeline tests**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m unittest 02_Code/Visualization/tests/test_eeg_pipeline.py -v
```

Expected: PASS, 5 tests.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add 02_Code/Visualization/eeg_pipeline.py 02_Code/Visualization/tests/test_eeg_pipeline.py
git commit -m "Add EEG dashboard pipeline"
```

## Task 3: Offline and Synthetic Data Sources

**Files:**
- Create: `02_Code/Visualization/data_sources.py`
- Create: `02_Code/Visualization/tests/test_data_sources.py`

- [ ] **Step 1: Write failing data-source tests**

Create `02_Code/Visualization/tests/test_data_sources.py`:

```python
import unittest

import numpy as np

from data_sources import OfflinePhysioNetSource, SyntheticBrainFlowSource


def fake_physionet_loader(subject):
    data = np.arange(5 * 3 * 8, dtype=np.float32).reshape(5, 3, 8)
    labels = np.array([0, 1, 2, 3, 0], dtype=np.int64)
    return data, labels


class FakeStream:
    sampling_rate = 250
    n_eeg_channels = 2

    def __init__(self):
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def get_eeg_epoch(self, duration_sec=4.0, apply_filter=False):
        samples = int(self.sampling_rate * duration_sec)
        return np.ones((self.n_eeg_channels, samples), dtype=np.float32)


class DataSourceTests(unittest.TestCase):
    def test_offline_source_steps_through_range(self):
        source = OfflinePhysioNetSource(loader=fake_physionet_loader, subject=3, start_epoch=1, stop_epoch=3)
        first = source.current()
        self.assertEqual(first.true_label, 1)
        self.assertEqual(first.epoch_index, 1)
        self.assertEqual(first.replay_index, 0)
        self.assertEqual(first.replay_total, 3)

        second = source.step()
        self.assertEqual(second.true_label, 2)
        self.assertEqual(second.replay_index, 1)

    def test_offline_reset_returns_to_first_epoch(self):
        source = OfflinePhysioNetSource(loader=fake_physionet_loader, subject=3, start_epoch=1, stop_epoch=3)
        source.step()
        source.reset()
        self.assertEqual(source.current().epoch_index, 1)

    def test_synthetic_source_uses_stream_without_true_label(self):
        stream = FakeStream()
        source = SyntheticBrainFlowSource(stream_factory=lambda: stream, duration_sec=0.5)
        source.start()
        epoch = source.step()
        self.assertTrue(stream.started)
        self.assertIsNone(epoch.true_label)
        self.assertEqual(epoch.raw_eeg.shape, (2, 125))
        self.assertEqual(epoch.replay_index, 0)
        source.stop()
        self.assertFalse(stream.started)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing data-source tests**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m unittest 02_Code/Visualization/tests/test_data_sources.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'data_sources'`.

- [ ] **Step 3: Implement data sources**

Create `02_Code/Visualization/data_sources.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

from models import SourceEpoch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_channel_names(n_channels: int) -> list[str]:
    return [f"Ch {i + 1}" for i in range(n_channels)]


def load_physionet_subject(subject: int) -> tuple[np.ndarray, np.ndarray]:
    code_root = PROJECT_ROOT / "02_Code"
    eeg_root = code_root / "EEG_Classification"
    for path in (code_root, eeg_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from physionet_loader import load_subject_4class_mne

    return load_subject_4class_mne(subject)


class OfflinePhysioNetSource:
    def __init__(
        self,
        subject: int,
        start_epoch: int = 0,
        stop_epoch: int | None = None,
        loader: Callable[[int], tuple[np.ndarray, np.ndarray]] = load_physionet_subject,
        sampling_rate: float = 160.0,
    ):
        data, labels = loader(subject)
        if data.ndim != 3:
            raise ValueError(f"Expected PhysioNet data shape (epochs, channels, samples), got {data.shape}")
        total_epochs = data.shape[0]
        end = total_epochs - 1 if stop_epoch is None else stop_epoch
        if start_epoch < 0 or end < start_epoch or end >= total_epochs:
            raise ValueError(f"Invalid epoch range {start_epoch}:{end} for {total_epochs} epochs")
        self.subject = subject
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.indices = list(range(start_epoch, end + 1))
        self.sampling_rate = sampling_rate
        self.position = 0
        self.channel_names = default_channel_names(self.data.shape[1])

    def reset(self) -> SourceEpoch:
        self.position = 0
        return self.current()

    def current(self) -> SourceEpoch:
        epoch_index = self.indices[self.position]
        raw = self.data[epoch_index]
        return SourceEpoch(
            raw_eeg=raw,
            sampling_rate=self.sampling_rate,
            channel_names=self.channel_names,
            true_label=int(self.labels[epoch_index]),
            subject=self.subject,
            epoch_index=epoch_index,
            replay_index=self.position,
            replay_total=len(self.indices),
            status={"mode": "offline", "source": f"S{self.subject:03d}"},
        )

    def step(self) -> SourceEpoch:
        if self.position < len(self.indices) - 1:
            self.position += 1
        return self.current()


def create_brainflow_synthetic_stream():
    physical_root = PROJECT_ROOT / "02_Code" / "Physical_Control"
    if str(physical_root) not in sys.path:
        sys.path.insert(0, str(physical_root))
    from openbci_stream import OpenBCIStream

    return OpenBCIStream(board_type="synthetic", target_channels=64, target_samples=1000)


class SyntheticBrainFlowSource:
    def __init__(
        self,
        stream_factory: Callable[[], object] = create_brainflow_synthetic_stream,
        duration_sec: float = 4.0,
    ):
        self.stream_factory = stream_factory
        self.duration_sec = duration_sec
        self.stream = None
        self.position = 0

    def start(self) -> None:
        if self.stream is None:
            self.stream = self.stream_factory()
            self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream = None

    def reset(self) -> None:
        self.position = 0

    def step(self) -> SourceEpoch:
        self.start()
        assert self.stream is not None
        raw = self.stream.get_eeg_epoch(duration_sec=self.duration_sec, apply_filter=False).astype(np.float32)
        epoch = SourceEpoch(
            raw_eeg=raw,
            sampling_rate=float(self.stream.sampling_rate),
            channel_names=default_channel_names(raw.shape[0]),
            true_label=None,
            subject=None,
            epoch_index=None,
            replay_index=self.position,
            replay_total=None,
            status={"mode": "synthetic", "ground_truth": "unavailable"},
        )
        self.position += 1
        return epoch
```

- [ ] **Step 4: Run data-source tests**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m unittest 02_Code/Visualization/tests/test_data_sources.py -v
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add 02_Code/Visualization/data_sources.py 02_Code/Visualization/tests/test_data_sources.py
git commit -m "Add dashboard data sources"
```

## Task 4: Arm Visualizer with PyBullet Fallback

**Files:**
- Create: `02_Code/Visualization/arm_visualizer.py`
- Create: `02_Code/Visualization/tests/test_arm_visualizer.py`

- [ ] **Step 1: Write failing arm visualizer tests**

Create `02_Code/Visualization/tests/test_arm_visualizer.py`:

```python
import unittest

import numpy as np

from arm_visualizer import ArmVisualizer


class FakeEnv:
    def __init__(self, cfg, render_mode=None):
        self.cfg = cfg
        self.render_mode = render_mode
        self.y = 0.0
        self.z = 0.0

    def reset(self):
        self.y = 0.0
        self.z = 0.0
        return np.array([self.y, self.z], dtype=np.float32), {}

    def step(self, action):
        if action == 0:
            self.y += 0.1
        elif action == 1:
            self.y -= 0.1
        elif action == 2:
            self.z += 0.1
        elif action == 3:
            self.z -= 0.1
        return np.array([self.y, self.z], dtype=np.float32), 0.0, False, False, {}

    def render(self):
        return np.zeros((12, 16, 3), dtype=np.uint8)

    def close(self):
        pass


class ArmVisualizerTests(unittest.TestCase):
    def test_fake_env_updates_trajectory_and_frame(self):
        visualizer = ArmVisualizer(env_cls=FakeEnv)
        visualizer.reset()
        frame = visualizer.step(2)
        self.assertIsNotNone(frame.arm_rgb)
        self.assertEqual(frame.arm_rgb.shape, (12, 16, 3))
        self.assertEqual(frame.trajectory_yz[-1], (0.0, 0.1))

    def test_fallback_without_env_still_tracks_actions(self):
        def failing_env_cls(cfg, render_mode=None):
            raise RuntimeError("pybullet unavailable")

        visualizer = ArmVisualizer(env_cls=failing_env_cls)
        visualizer.reset()
        frame = visualizer.step(0)
        self.assertIsNone(frame.arm_rgb)
        self.assertEqual(frame.trajectory_yz[-1], (0.03, 0.0))
        self.assertEqual(frame.status["arm_mode"], "fallback")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing arm visualizer tests**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m unittest 02_Code/Visualization/tests/test_arm_visualizer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'arm_visualizer'`.

- [ ] **Step 3: Implement arm visualizer**

Create `02_Code/Visualization/arm_visualizer.py`:

```python
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ArmFrame:
    arm_rgb: np.ndarray | None
    trajectory_yz: list[tuple[float, float]]
    status: dict[str, object] = field(default_factory=dict)


def load_pybullet_env():
    simulation_root = PROJECT_ROOT / "02_Code" / "Simulation"
    if str(simulation_root) not in sys.path:
        sys.path.insert(0, str(simulation_root))
    from pybullet_arm_env import PBConfig, PyBulletArmEnv

    return PyBulletArmEnv, PBConfig


class ArmVisualizer:
    def __init__(self, env_cls=None, cfg_cls=None, step_size: float = 0.03):
        self.env_cls = env_cls
        self.cfg_cls = cfg_cls
        self.step_size = step_size
        self.env = None
        self.fallback_y = 0.0
        self.fallback_z = 0.0
        self.trajectory_yz: list[tuple[float, float]] = [(0.0, 0.0)]
        self.arm_mode = "uninitialized"

    def _ensure_env(self) -> None:
        if self.env is not None or self.arm_mode == "fallback":
            return
        try:
            env_cls = self.env_cls
            cfg_cls = self.cfg_cls
            if env_cls is None or cfg_cls is None:
                env_cls, cfg_cls = load_pybullet_env()
            cfg = cfg_cls(step_size=self.step_size, use_gui=False)
            self.env = env_cls(cfg, render_mode="rgb_array")
            self.arm_mode = "pybullet"
        except Exception as exc:
            self.env = None
            self.arm_mode = "fallback"
            self.fallback_error = str(exc)

    def reset(self) -> ArmFrame:
        self.trajectory_yz = [(0.0, 0.0)]
        self.fallback_y = 0.0
        self.fallback_z = 0.0
        self._ensure_env()
        if self.env is not None:
            obs, _ = self.env.reset()
            self.trajectory_yz = [(float(obs[0]), float(obs[1]))]
            return ArmFrame(self.env.render(), list(self.trajectory_yz), {"arm_mode": self.arm_mode})
        return ArmFrame(None, list(self.trajectory_yz), {"arm_mode": self.arm_mode, "error": getattr(self, "fallback_error", "")})

    def step(self, action: int | None) -> ArmFrame:
        self._ensure_env()
        if action is None:
            return ArmFrame(None, list(self.trajectory_yz), {"arm_mode": self.arm_mode})
        if self.env is not None:
            obs, _, _, _, _ = self.env.step(int(action))
            self.trajectory_yz.append((float(obs[0]), float(obs[1])))
            return ArmFrame(self.env.render(), list(self.trajectory_yz), {"arm_mode": self.arm_mode})

        if action == 0:
            self.fallback_y += self.step_size
        elif action == 1:
            self.fallback_y -= self.step_size
        elif action == 2:
            self.fallback_z += self.step_size
        elif action == 3:
            self.fallback_z -= self.step_size
        self.trajectory_yz.append((float(self.fallback_y), float(self.fallback_z)))
        return ArmFrame(None, list(self.trajectory_yz), {"arm_mode": "fallback", "error": getattr(self, "fallback_error", "")})

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
```

- [ ] **Step 4: Run arm visualizer tests**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m unittest 02_Code/Visualization/tests/test_arm_visualizer.py -v
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add 02_Code/Visualization/arm_visualizer.py 02_Code/Visualization/tests/test_arm_visualizer.py
git commit -m "Add dashboard arm visualizer"
```

## Task 5: Plotting and Export Helpers

**Files:**
- Create: `02_Code/Visualization/plotting.py`
- Create: `02_Code/Visualization/tests/test_plotting.py`

- [ ] **Step 1: Write failing plotting tests**

Create `02_Code/Visualization/tests/test_plotting.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from plotting import (
    export_jsonl,
    make_eeg_figure,
    make_probability_figure,
    make_trajectory_figure,
)


class PlottingTests(unittest.TestCase):
    def test_eeg_figure_contains_axes(self):
        raw = np.zeros((3, 100), dtype=np.float32)
        preprocessed = np.ones((3, 100), dtype=np.float32)
        fig = make_eeg_figure(raw, preprocessed, sampling_rate=100.0, channel_names=["A", "B", "C"], max_channels=2)
        self.assertEqual(len(fig.axes), 2)

    def test_trajectory_figure_contains_one_axis(self):
        fig = make_trajectory_figure([(0.0, 0.0), (0.1, 0.0)])
        self.assertEqual(len(fig.axes), 1)

    def test_probability_figure_contains_one_axis(self):
        fig = make_probability_figure(np.array([0.1, 0.6, 0.2, 0.1], dtype=np.float32))
        self.assertEqual(len(fig.axes), 1)

    def test_export_jsonl_writes_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.jsonl"
            export_jsonl(path, [{"mode": "offline", "executed_action": "left"}])
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[0])["executed_action"], "left")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing plotting tests**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m unittest 02_Code/Visualization/tests/test_plotting.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'plotting'`.

- [ ] **Step 3: Implement plotting and export helpers**

Create `02_Code/Visualization/plotting.py`:

```python
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models import CLASS_NAMES


def _selected_channel_indices(n_channels: int, max_channels: int) -> list[int]:
    return list(range(min(n_channels, max_channels)))


def make_eeg_figure(
    raw_eeg: np.ndarray,
    preprocessed_eeg: np.ndarray,
    sampling_rate: float,
    channel_names: list[str],
    max_channels: int = 8,
):
    raw = np.asarray(raw_eeg, dtype=np.float32)
    processed = np.asarray(preprocessed_eeg, dtype=np.float32)
    if raw.shape != processed.shape:
        raise ValueError(f"raw and preprocessed EEG shapes differ: {raw.shape} vs {processed.shape}")
    indices = _selected_channel_indices(raw.shape[0], max_channels)
    times = np.arange(raw.shape[1], dtype=np.float32) / float(sampling_rate)
    fig, axes = plt.subplots(len(indices), 1, figsize=(8, max(2.5, 1.2 * len(indices))), sharex=True)
    axes_list = np.atleast_1d(axes)
    for ax, idx in zip(axes_list, indices):
        offset_raw = raw[idx]
        offset_processed = processed[idx]
        name = channel_names[idx] if idx < len(channel_names) else f"Ch {idx + 1}"
        ax.plot(times, offset_raw, color="#666666", linewidth=0.8, label="raw")
        ax.plot(times, offset_processed, color="#0072B2", linewidth=0.9, label="preprocessed")
        ax.set_ylabel(name)
        ax.grid(alpha=0.2)
    axes_list[0].legend(loc="upper right", fontsize=8)
    axes_list[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def make_trajectory_figure(trajectory_yz: list[tuple[float, float]]):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    if trajectory_yz:
        arr = np.asarray(trajectory_yz, dtype=np.float32)
        ax.plot(arr[:, 0], arr[:, 1], "-o", color="#0072B2", markersize=3)
        ax.scatter(arr[-1, 0], arr[-1, 1], color="#D55E00", s=50, label="current")
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.set_title("End-effector Trajectory")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_probability_figure(probabilities: np.ndarray | None):
    probs = np.zeros(4, dtype=np.float32) if probabilities is None else np.asarray(probabilities, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.bar(CLASS_NAMES, probs, color=["#0072B2", "#D55E00", "#009E73", "#CC79A7"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("CTNet Classification")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def export_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def export_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
```

- [ ] **Step 4: Run plotting tests**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m unittest 02_Code/Visualization/tests/test_plotting.py -v
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add 02_Code/Visualization/plotting.py 02_Code/Visualization/tests/test_plotting.py
git commit -m "Add dashboard plotting helpers"
```

## Task 6: Streamlit Dashboard App

**Files:**
- Create: `02_Code/Visualization/dashboard_app.py`

- [ ] **Step 1: Add import smoke command before implementation**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m py_compile 02_Code/Visualization/dashboard_app.py
```

Expected: FAIL with `No such file or directory`.

- [ ] **Step 2: Implement dashboard app**

Create `02_Code/Visualization/dashboard_app.py`:

```python
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit("streamlit is required. Install with: pip install streamlit") from exc

from arm_visualizer import ArmVisualizer
from data_sources import OfflinePhysioNetSource, SyntheticBrainFlowSource
from eeg_pipeline import DEFAULT_MODEL_PATH, EEGPipeline, preprocess_for_display
from models import DashboardFrame, build_action_decision, class_name, scripted_action_for_step
from plotting import export_csv, export_jsonl, make_eeg_figure, make_probability_figure, make_trajectory_figure


EXPORT_DIR = Path("03_Experiments/Visualization")


def init_state() -> None:
    st.session_state.setdefault("source", None)
    st.session_state.setdefault("source_key", None)
    st.session_state.setdefault("arm", ArmVisualizer())
    st.session_state.setdefault("pipeline", None)
    st.session_state.setdefault("records", [])
    st.session_state.setdefault("last_frame", None)


def get_pipeline(model_path: str, device: str) -> EEGPipeline:
    key = (model_path, device)
    current = st.session_state.get("pipeline")
    if current is None or st.session_state.get("pipeline_key") != key:
        st.session_state["pipeline"] = EEGPipeline(model_path=Path(model_path), device=device)
        st.session_state["pipeline_key"] = key
    return st.session_state["pipeline"]


def get_source(mode: str, subject: int, start_epoch: int, stop_epoch: int, duration_sec: float):
    key = (mode, subject, start_epoch, stop_epoch, duration_sec)
    if st.session_state.get("source_key") == key and st.session_state.get("source") is not None:
        return st.session_state["source"]
    old_source = st.session_state.get("source")
    if hasattr(old_source, "stop"):
        old_source.stop()
    if mode == "Offline PhysioNet":
        source = OfflinePhysioNetSource(subject=subject, start_epoch=start_epoch, stop_epoch=stop_epoch)
    else:
        source = SyntheticBrainFlowSource(duration_sec=duration_sec)
    st.session_state["source"] = source
    st.session_state["source_key"] = key
    st.session_state["records"] = []
    st.session_state["arm"] = ArmVisualizer()
    st.session_state["arm"].reset()
    return source


def build_dashboard_frame(mode: str, source_epoch, pipeline: EEGPipeline, scripted_demo: bool) -> DashboardFrame:
    raw = source_epoch.raw_eeg
    preprocessed = preprocess_for_display(raw, sampling_rate=source_epoch.sampling_rate)
    prediction = pipeline.predict(preprocessed)
    scripted_action = None
    if mode == "BrainFlow synthetic" and scripted_demo:
        scripted_action = scripted_action_for_step(source_epoch.replay_index or 0)
    decision = build_action_decision(prediction.pred_class, scripted_demo_action=scripted_action)
    arm_frame = st.session_state["arm"].step(decision.executed_action)
    true_name = class_name(source_epoch.true_label)
    status = dict(source_epoch.status)
    status.update(arm_frame.status)
    frame = DashboardFrame(
        mode=mode,
        raw_eeg=raw,
        preprocessed_eeg_for_display=preprocessed,
        model_input_shape=prediction.model_input_shape,
        sampling_rate=source_epoch.sampling_rate,
        channel_names=source_epoch.channel_names,
        pred_class=prediction.pred_class,
        pred_name=prediction.pred_name,
        probabilities=prediction.probabilities,
        confidence=prediction.confidence,
        true_label=source_epoch.true_label,
        true_name=true_name,
        ctnet_predicted_action=decision.ctnet_predicted_action,
        ctnet_predicted_action_name=decision.ctnet_predicted_action_name,
        scripted_demo_action=decision.scripted_demo_action,
        scripted_demo_action_name=decision.scripted_demo_action_name,
        executed_action=decision.executed_action,
        executed_action_name=decision.executed_action_name,
        executed_action_source=decision.executed_action_source,
        arm_rgb=arm_frame.arm_rgb,
        trajectory_yz=arm_frame.trajectory_yz,
        replay_index=source_epoch.replay_index,
        replay_total=source_epoch.replay_total,
        status=status,
    )
    return frame


def record_frame(frame: DashboardFrame) -> None:
    st.session_state["records"].append(
        {
            "mode": frame.mode,
            "replay_index": frame.replay_index,
            "pred_name": frame.pred_name,
            "confidence": frame.confidence,
            "true_name": frame.true_name,
            "ctnet_action": frame.ctnet_predicted_action_name,
            "scripted_action": frame.scripted_demo_action_name,
            "executed_action": frame.executed_action_name,
            "executed_action_source": frame.executed_action_source,
            "trajectory_len": len(frame.trajectory_yz),
        }
    )


def render_dashboard(frame: DashboardFrame, max_channels: int) -> None:
    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)

    with top_left:
        st.subheader("PyBullet Arm Simulation")
        if frame.arm_rgb is not None:
            st.image(frame.arm_rgb, channels="RGB", use_container_width=True)
        else:
            st.info("PyBullet frame unavailable. Showing trajectory fallback below.")
        st.caption(f"Arm mode: {frame.status.get('arm_mode', 'unknown')}")

    with top_right:
        st.subheader("Raw EEG vs Preprocessed EEG")
        st.pyplot(
            make_eeg_figure(
                frame.raw_eeg,
                frame.preprocessed_eeg_for_display,
                frame.sampling_rate,
                frame.channel_names,
                max_channels=max_channels,
            )
        )
        st.caption(f"Displayed preprocessing: 8-30 Hz bandpass. Model input shape: {frame.model_input_shape}")

    with bottom_left:
        st.subheader("Y-Z Trajectory")
        st.pyplot(make_trajectory_figure(frame.trajectory_yz))

    with bottom_right:
        st.subheader("Classification and Action")
        st.pyplot(make_probability_figure(frame.probabilities))
        st.metric("Prediction", frame.pred_name or "none", f"{frame.confidence:.2f}" if frame.confidence is not None else None)
        if frame.true_name is not None:
            st.write(f"True label: `{frame.true_name}`")
        else:
            st.warning("Synthetic EEG has no ground-truth motor-imagery label.")
        st.write(f"CTNet action: `{frame.ctnet_predicted_action_name}`")
        st.write(f"Scripted demo action: `{frame.scripted_demo_action_name or 'disabled'}`")
        st.write(f"Executed action: `{frame.executed_action_name}` from `{frame.executed_action_source}`")


def main() -> None:
    st.set_page_config(page_title="EEG Arm Observer", layout="wide")
    init_state()
    st.title("EEG and Robotic Arm Observer")

    with st.sidebar:
        mode = st.radio("Mode", ["Offline PhysioNet", "BrainFlow synthetic"])
        model_path = st.text_input("CTNet model path", value=str(DEFAULT_MODEL_PATH))
        device = st.selectbox("Device", ["cpu", "cuda"], index=0)
        max_channels = st.selectbox("Displayed EEG channels", [8, 16, 32, 64], index=0)
        subject = st.number_input("Subject", min_value=1, max_value=109, value=1, step=1)
        start_epoch = st.number_input("Start epoch", min_value=0, value=0, step=1)
        stop_epoch = st.number_input("Stop epoch", min_value=0, value=3, step=1)
        duration_sec = st.number_input("Synthetic epoch duration", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
        scripted_demo = st.checkbox("Use scripted demo action in synthetic mode", value=True)
        playback_delay = st.slider("Playback delay (s)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

        step_clicked = st.button("Step")
        reset_clicked = st.button("Reset")
        export_clicked = st.button("Export log")

    try:
        source = get_source(mode, int(subject), int(start_epoch), int(stop_epoch), float(duration_sec))
        pipeline = get_pipeline(model_path, device)
        if reset_clicked:
            if hasattr(source, "reset"):
                source.reset()
            st.session_state["arm"] = ArmVisualizer()
            st.session_state["arm"].reset()
            st.session_state["records"] = []
            st.session_state["last_frame"] = None
        if step_clicked or st.session_state["last_frame"] is None:
            epoch = source.step() if mode == "BrainFlow synthetic" else source.current() if st.session_state["last_frame"] is None else source.step()
            frame = build_dashboard_frame(mode, epoch, pipeline, scripted_demo=scripted_demo)
            st.session_state["last_frame"] = frame
            record_frame(frame)
            if playback_delay > 0 and step_clicked:
                time.sleep(playback_delay)
        frame = st.session_state["last_frame"]
        if frame is not None:
            render_dashboard(frame, int(max_channels))
        if export_clicked:
            EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            json_path = EXPORT_DIR / "dashboard_run.jsonl"
            csv_path = EXPORT_DIR / "dashboard_run.csv"
            export_jsonl(json_path, st.session_state["records"])
            export_csv(csv_path, st.session_state["records"])
            st.success(f"Exported {json_path} and {csv_path}")
    except Exception as exc:
        st.error(str(exc))
        if mode == "BrainFlow synthetic":
            st.info("Synthetic mode requires BrainFlow. Install with `pip install brainflow`.")
        if mode == "Offline PhysioNet":
            st.info("Offline mode requires MNE and local PhysioNet EDF data under `06_Data/physionet_raw/`.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Compile the dashboard app**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m py_compile 02_Code/Visualization/dashboard_app.py
```

Expected: PASS with no output.

- [ ] **Step 4: Run the full visualization test suite**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m unittest discover 02_Code/Visualization/tests -v
```

Expected: PASS for all visualization tests.

- [ ] **Step 5: Commit Task 6**

Run:

```bash
git add 02_Code/Visualization/dashboard_app.py
git commit -m "Add Streamlit EEG arm dashboard"
```

## Task 7: Dependencies, Documentation, and Smoke Verification

**Files:**
- Create: `02_Code/Visualization/README.md`
- Modify: `02_Code/Utils/requirements.txt`
- Modify: `README.md`

- [ ] **Step 1: Update requirements**

Modify `02_Code/Utils/requirements.txt` by adding these lines after the PyBullet entry:

```text

# Local dashboard UI
streamlit>=1.34.0

# BrainFlow synthetic/OpenBCI dashboard mode
brainflow>=5.10.0
```

- [ ] **Step 2: Add visualization README**

Create `02_Code/Visualization/README.md`:

```markdown
# EEG and Arm Observer Dashboard

This local Streamlit dashboard shows the closed-loop BCI control pipeline in one screen:

- raw EEG versus 8-30 Hz preprocessed EEG
- PyBullet robotic-arm simulation frame
- Y-Z end-effector trajectory
- CTNet class probabilities, predicted class, true label when available, and executed action

## Install

From the repository root:

```bash
pip install -r 02_Code/Utils/requirements.txt
```

## Start

```bash
streamlit run 02_Code/Visualization/dashboard_app.py
```

## Offline PhysioNet Mode

Use `Offline PhysioNet` to replay local PhysioNet epochs. The dashboard loads epochs through `02_Code/EEG_Classification/physionet_loader.py`, displays raw and preprocessed waveforms, runs CTNet inference, and drives the PyBullet arm from the CTNet-derived action.

Controls:

- `Subject`: PhysioNet subject id
- `Start epoch` and `Stop epoch`: ordered replay range
- `Step`: advance by one epoch
- `Reset`: return to the first selected epoch and clear the trajectory

## BrainFlow Synthetic Mode

Use `BrainFlow synthetic` to test the online data path without EEG hardware. The dashboard displays CTNet inference on the synthetic EEG stream. Synthetic EEG has no ground-truth motor-imagery label, so the classification panel marks true label as unavailable.

When `Use scripted demo action in synthetic mode` is enabled, the arm follows a stable scripted sequence while the CTNet prediction remains visible separately. The panel labels the executed action source as either `scripted demo` or `CTNet prediction`.

## Exports

The `Export log` button writes:

- `03_Experiments/Visualization/dashboard_run.jsonl`
- `03_Experiments/Visualization/dashboard_run.csv`

## Limitations

- The dashboard is a local observer and demo tool.
- It does not control the physical arm.
- It does not train or fine-tune CTNet.
- BrainFlow synthetic mode is not live human EEG validation.
```

- [ ] **Step 3: Update top-level README quick start**

Modify `README.md` in the quick-start command block by adding:

```bash
# Launch the EEG + PyBullet observer dashboard
streamlit run 02_Code/Visualization/dashboard_app.py
```

Add one bullet near the quick links or code-directory explanation:

```markdown
- [EEG + Arm Observer Dashboard](02_Code/Visualization/README.md)
```

- [ ] **Step 4: Run full tests**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m unittest discover 02_Code/Visualization/tests -v
```

Expected: PASS for all visualization tests.

- [ ] **Step 5: Run dashboard import smoke check**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization python -m py_compile 02_Code/Visualization/dashboard_app.py
```

Expected: PASS with no output.

- [ ] **Step 6: Run Streamlit server smoke check**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization streamlit run 02_Code/Visualization/dashboard_app.py --server.headless true --server.port 8501
```

Expected: Streamlit prints a local URL. Stop with `Ctrl+C` after confirming the server starts.

- [ ] **Step 7: Commit Task 7**

Run:

```bash
git add 02_Code/Visualization/README.md 02_Code/Utils/requirements.txt README.md
git commit -m "Document EEG arm observer dashboard"
```

## Task 8: Manual Demo Verification

**Files:**
- No file changes expected unless manual verification exposes a defect.

- [ ] **Step 1: Start the dashboard**

Run:

```bash
cd /home/tomato/3YP
PYTHONPATH=02_Code/Visualization streamlit run 02_Code/Visualization/dashboard_app.py
```

Expected: Browser page opens with four panels.

- [ ] **Step 2: Verify offline mode**

In the browser:

1. Select `Offline PhysioNet`.
2. Use subject `1`, start epoch `0`, stop epoch `3`.
3. Click `Step` four times.

Expected:

- top-left panel shows a PyBullet image or a clear fallback notice
- top-right panel shows raw and preprocessed EEG traces
- bottom-left trajectory changes after each step
- bottom-right panel shows prediction, confidence, true label, CTNet action, and executed action

- [ ] **Step 3: Verify synthetic mode**

In the browser:

1. Select `BrainFlow synthetic`.
2. Enable `Use scripted demo action in synthetic mode`.
3. Click `Step` twice.

Expected:

- top-right panel shows rolling synthetic EEG traces
- bottom-right panel shows CTNet prediction and scripted demo action separately
- executed action source is `scripted demo`
- true-label area states that synthetic EEG has no ground-truth motor-imagery label

- [ ] **Step 4: Verify export**

In the browser:

1. Click `Export log`.
2. Confirm files exist:

```bash
test -f 03_Experiments/Visualization/dashboard_run.jsonl
test -f 03_Experiments/Visualization/dashboard_run.csv
```

Expected: Both commands exit successfully.

- [ ] **Step 5: Commit manual fixes only if defects were found**

If code or documentation changed during manual verification:

```bash
git add 02_Code/Visualization README.md 02_Code/Utils/requirements.txt
git commit -m "Fix dashboard demo verification issues"
```

If no files changed, do not create a commit.

## Self-Review

Spec coverage:

- Raw EEG versus preprocessed EEG: Task 2 defines display preprocessing, Task 5 plots it, Task 6 renders it.
- PyBullet simulation: Task 4 implements the visualizer, Task 6 renders the frame.
- Trajectory plot: Task 4 records Y-Z history, Task 5 plots it, Task 6 renders it.
- Classification results: Task 2 predicts, Task 5 plots probabilities, Task 6 renders class/action fields.
- Offline replay: Task 3 defines replay source, Task 6 controls step/reset, Task 8 verifies range stepping.
- BrainFlow synthetic: Task 3 wraps synthetic stream, Task 6 exposes mode, Task 8 verifies scripted action labeling.
- Separate CTNet/scripted/executed actions: Task 1 data model, Task 6 rendering, Task 8 manual check.
- Documentation: Task 7 adds README and top-level quick-start.

Placeholder scan:

- This plan contains concrete file paths, code snippets, commands, and expected results for every implementation step.

Type consistency:

- `DashboardFrame`, `SourceEpoch`, and action fields are defined in Task 1 and reused consistently in Tasks 2, 3, 4, 5, and 6.
