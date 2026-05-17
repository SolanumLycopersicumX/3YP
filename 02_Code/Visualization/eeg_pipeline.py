from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models import build_action_decision, class_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "04_Trained_Models/physionet_ctnet_109sub/pool/model_pool.pth"
)
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


def _validate_2d(data: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (channels, samples)")
    return array


def preprocess_for_display(
    raw_eeg: np.ndarray,
    sampling_rate: float,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
) -> np.ndarray:
    data = _validate_2d(raw_eeg, "raw_eeg").astype(np.float32, copy=True)
    nyquist = float(sampling_rate) / 2.0
    if l_freq <= 0 or h_freq <= l_freq or h_freq >= nyquist:
        return data

    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        return data

    try:
        b, a = butter(4, [l_freq / nyquist, h_freq / nyquist], btype="band")
        filtered = filtfilt(b, a, data, axis=1)
    except ValueError:
        return data
    return filtered.astype(np.float32, copy=False)


def resample_time(data: np.ndarray, target_samples: int) -> np.ndarray:
    array = _validate_2d(data, "data").astype(np.float32, copy=False)
    target_samples = int(target_samples)
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")
    if array.shape[1] == target_samples:
        return array.astype(np.float32, copy=True)

    try:
        from scipy.signal import resample

        return resample(array, target_samples, axis=1).astype(np.float32, copy=False)
    except ImportError:
        old_x = np.linspace(0.0, 1.0, array.shape[1], dtype=np.float32)
        new_x = np.linspace(0.0, 1.0, target_samples, dtype=np.float32)
        rows = [np.interp(new_x, old_x, row).astype(np.float32) for row in array]
        return np.stack(rows, axis=0)


def adapt_channels(data: np.ndarray, target_channels: int) -> np.ndarray:
    array = _validate_2d(data, "data").astype(np.float32, copy=False)
    target_channels = int(target_channels)
    if target_channels <= 0:
        raise ValueError("target_channels must be positive")
    if array.shape[0] == target_channels:
        return array.astype(np.float32, copy=True)
    if array.shape[0] > target_channels:
        return array[:target_channels].astype(np.float32, copy=True)

    padded = np.zeros((target_channels, array.shape[1]), dtype=np.float32)
    padded[: array.shape[0]] = array
    return padded


def create_model_input(
    preprocessed_eeg: np.ndarray,
    target_channels: int = DEFAULT_TARGET_CHANNELS,
    target_samples: int = DEFAULT_TARGET_SAMPLES,
    norm_mean: float = 0.0,
    norm_std: float = 1.0,
) -> np.ndarray:
    data = adapt_channels(preprocessed_eeg, target_channels)
    data = resample_time(data, target_samples)
    norm_std = float(norm_std)
    if norm_std == 0.0:
        norm_std = 1.0
    normalized = (data.astype(np.float32, copy=False) - float(norm_mean)) / norm_std
    return normalized[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)


def load_norm_metadata(model_path: Path) -> dict[str, float]:
    model_path = Path(model_path)
    defaults = {"norm_mean": 0.0, "norm_std": 1.0}

    metadata_path = model_path.with_suffix(".json")
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "norm_mean": float(
                data.get("norm_mean", data.get("mean", defaults["norm_mean"]))
            ),
            "norm_std": float(
                data.get("norm_std", data.get("std", defaults["norm_std"]))
            ),
        }

    norm_params_path = model_path.parent / "norm_params.json"
    if norm_params_path.exists():
        with norm_params_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "norm_mean": float(
                data.get("norm_mean", data.get("mean", defaults["norm_mean"]))
            ),
            "norm_std": float(
                data.get("norm_std", data.get("std", defaults["norm_std"]))
            ),
        }

    return defaults


def _prepare_ctnet_unpickle_context() -> None:
    code_root = PROJECT_ROOT / "02_Code"
    eeg_classification_root = code_root / "EEG_Classification"
    for path in (code_root, eeg_classification_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    try:
        import CTNet_model
    except ImportError:
        return

    main_module = sys.modules.get("__main__")
    if main_module is None:
        return

    for attr_name in dir(CTNet_model):
        if not attr_name.startswith("_"):
            setattr(main_module, attr_name, getattr(CTNet_model, attr_name))


def load_ctnet_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    _prepare_ctnet_unpickle_context()
    loaded: Any = torch.load(model_path, map_location=device, weights_only=False)
    model = loaded.get("model", loaded) if isinstance(loaded, dict) else loaded
    model.to(device)
    model.eval()
    return model


class EEGPipeline:
    def __init__(
        self,
        model: torch.nn.Module | None = None,
        model_path: Path = DEFAULT_MODEL_PATH,
        device: str | torch.device = "cpu",
        norm_mean: float | None = None,
        norm_std: float | None = None,
        target_channels: int = DEFAULT_TARGET_CHANNELS,
        target_samples: int = DEFAULT_TARGET_SAMPLES,
    ) -> None:
        self.model = model
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        norm_metadata = load_norm_metadata(self.model_path)
        self.norm_mean = float(
            norm_mean
            if norm_mean is not None
            else norm_metadata.get("norm_mean", norm_metadata.get("mean", 0.0))
        )
        self.norm_std = float(
            norm_std
            if norm_std is not None
            else norm_metadata.get("norm_std", norm_metadata.get("std", 1.0))
        )
        self.target_channels = int(target_channels)
        self.target_samples = int(target_samples)
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def ensure_model(self) -> torch.nn.Module:
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
        model_input = self.make_model_input(preprocessed_eeg)
        input_tensor = torch.from_numpy(model_input).to(self.device)
        model = self.ensure_model()

        with torch.no_grad():
            output = model(input_tensor)
            logits = output[-1] if isinstance(output, tuple) else output
            probabilities_tensor = torch.softmax(logits, dim=1)

        probabilities = (
            probabilities_tensor.detach().cpu().numpy().astype(np.float32, copy=False)[0]
        )
        pred_class = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_class])
        decision = build_action_decision(pred_class)

        if decision.ctnet_predicted_action is None:
            raise ValueError(f"No action mapping for predicted class: {pred_class}")

        return PredictionResult(
            pred_class=pred_class,
            pred_name=class_name(pred_class) or "",
            probabilities=probabilities,
            confidence=confidence,
            ctnet_predicted_action=decision.ctnet_predicted_action,
            ctnet_predicted_action_name=decision.ctnet_predicted_action_name or "",
            model_input_shape=tuple(model_input.shape),
        )
