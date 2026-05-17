from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


CLASS_NAMES = ["Left", "Right", "Hands/Up", "Feet/Down"]
ACTION_NAMES = ["left", "right", "up", "down"]
CLASS_TO_ACTION = {0: 0, 1: 1, 2: 2, 3: 3}
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
    status: dict[str, Any]


def _validate_index(value: int, names: list[str], kind: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{kind} must be an integer")
    if value < 0 or value >= len(names):
        raise ValueError(f"Unknown {kind}: {value}")
    return value


def action_name(action: int | None) -> str | None:
    if action is None:
        return None
    return ACTION_NAMES[_validate_index(action, ACTION_NAMES, "action")]


def class_to_action(pred_class: int | None) -> tuple[int, str] | None:
    if pred_class is None:
        return None
    if not isinstance(pred_class, int):
        raise ValueError("class must be an integer")
    try:
        action = CLASS_TO_ACTION[pred_class]
    except KeyError as exc:
        raise ValueError(f"Unknown class: {pred_class}") from exc
    name = action_name(action)
    if name is None:
        raise ValueError(f"Unknown action: {action}")
    return action, name


def class_name(label: int | None) -> str | None:
    if label is None:
        return None
    return CLASS_NAMES[_validate_index(label, CLASS_NAMES, "class")]


def scripted_action_for_step(step_index: int) -> int:
    if not isinstance(step_index, int):
        raise ValueError("step_index must be an integer")
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    return SCRIPTED_ACTION_SEQUENCE[step_index % len(SCRIPTED_ACTION_SEQUENCE)]


def build_action_decision(
    pred_class: int | None,
    scripted_demo_action: int | None = None,
) -> ActionDecision:
    ctnet_action_pair = class_to_action(pred_class)
    if ctnet_action_pair is None:
        ctnet_action = None
        ctnet_action_name = None
    else:
        ctnet_action, ctnet_action_name = ctnet_action_pair
    scripted_action_name = action_name(scripted_demo_action)

    if scripted_demo_action is not None:
        executed_action = scripted_demo_action
        executed_action_source = "scripted demo"
    elif ctnet_action is not None:
        executed_action = ctnet_action
        executed_action_source = "CTNet prediction"
    else:
        executed_action = None
        executed_action_source = "none"

    return ActionDecision(
        ctnet_predicted_action=ctnet_action,
        ctnet_predicted_action_name=ctnet_action_name,
        scripted_demo_action=scripted_demo_action,
        scripted_demo_action_name=scripted_action_name,
        executed_action=executed_action,
        executed_action_name=action_name(executed_action),
        executed_action_source=executed_action_source,
    )
