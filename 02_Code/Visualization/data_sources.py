from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

from models import SourceEpoch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _add_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def default_channel_names(n_channels: int) -> list[str]:
    return [f"Ch {idx}" for idx in range(1, int(n_channels) + 1)]


def load_physionet_subject(subject) -> tuple[np.ndarray, np.ndarray]:
    code_root = PROJECT_ROOT / "02_Code"
    _add_sys_path(code_root)
    _add_sys_path(code_root / "EEG_Classification")

    from physionet_loader import load_subject_4class_mne

    return load_subject_4class_mne(subject)


class OfflinePhysioNetSource:
    def __init__(
        self,
        subject,
        start_epoch: int = 0,
        stop_epoch: int | None = None,
        loader: Callable = load_physionet_subject,
        sampling_rate: float = 160.0,
    ) -> None:
        data, labels = loader(subject)
        data = np.asarray(data)
        labels = np.asarray(labels).reshape(-1)
        if data.ndim != 3:
            raise ValueError("PhysioNet data must have shape (epochs, channels, samples)")
        if any(dim == 0 for dim in data.shape):
            raise ValueError("PhysioNet data dimensions must be non-empty")

        n_epochs = data.shape[0]
        if labels.shape[0] != n_epochs:
            raise ValueError("PhysioNet labels must match the number of epochs")

        start_epoch = int(start_epoch)
        if stop_epoch is None:
            stop_epoch = n_epochs - 1
        else:
            stop_epoch = int(stop_epoch)

        if (
            start_epoch < 0
            or stop_epoch < start_epoch
            or stop_epoch >= n_epochs
        ):
            raise ValueError("Invalid PhysioNet epoch range")

        self.subject = int(subject)
        self.data = data
        self.labels = labels
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch
        self.sampling_rate = float(sampling_rate)
        self.channel_names = default_channel_names(data.shape[1])
        self.position = 0
        self.replay_total = stop_epoch - start_epoch + 1

    def reset(self) -> SourceEpoch:
        self.position = 0
        return self.current()

    def current(self) -> SourceEpoch:
        epoch_index = self.start_epoch + self.position
        return SourceEpoch(
            raw_eeg=self.data[epoch_index].astype(np.float32, copy=False),
            sampling_rate=self.sampling_rate,
            channel_names=self.channel_names,
            true_label=int(self.labels[epoch_index]),
            subject=self.subject,
            epoch_index=epoch_index,
            replay_index=self.position,
            replay_total=self.replay_total,
            status={"mode": "offline", "source": f"S{self.subject:03d}"},
        )

    def step(self) -> SourceEpoch:
        if self.position < self.replay_total - 1:
            self.position += 1
        return self.current()


def create_brainflow_synthetic_stream():
    physical_control_root = PROJECT_ROOT / "02_Code" / "Physical_Control"
    _add_sys_path(physical_control_root)

    from openbci_stream import OpenBCIStream

    return OpenBCIStream(
        board_type="synthetic",
        target_channels=64,
        target_samples=1000,
    )


class SyntheticBrainFlowSource:
    def __init__(
        self,
        stream_factory: Callable = create_brainflow_synthetic_stream,
        duration_sec: float = 4.0,
    ) -> None:
        self.stream_factory = stream_factory
        self.duration_sec = float(duration_sec)
        self.stream = None
        self.position = 0

    def start(self) -> None:
        if self.stream is None:
            stream = self.stream_factory()
            try:
                stream.start()
            except Exception:
                stop = getattr(stream, "stop", None)
                if callable(stop):
                    try:
                        stop()
                    except Exception:
                        pass
                self.stream = None
                raise
            self.stream = stream

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream = None

    def reset(self) -> None:
        self.position = 0

    def step(self) -> SourceEpoch:
        self.start()
        raw_eeg = self.stream.get_eeg_epoch(
            duration_sec=self.duration_sec,
            apply_filter=False,
        )
        epoch = SourceEpoch(
            raw_eeg=np.asarray(raw_eeg).astype(np.float32, copy=False),
            sampling_rate=float(self.stream.sampling_rate),
            channel_names=default_channel_names(self.stream.n_eeg_channels),
            true_label=None,
            subject=None,
            epoch_index=None,
            replay_index=self.position,
            replay_total=None,
            status={"mode": "synthetic", "ground_truth": "unavailable"},
        )
        self.position += 1
        return epoch
