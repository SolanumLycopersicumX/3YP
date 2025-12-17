"""将 OVR-CSP 输出的特征转为离线强化学习数据集。

运行示例:
    python scripts/prepare_rl_dataset.py --subject A01T

输出:
    - outputs/preprocessing/<subject>/rl_dataset.npz
        包含 state、action、reward、done 数组以及元数据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "preprocessing"


def load_csp_artifacts(subject: str) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """读取 CSP 特征、标签、元数据。"""
    subject_dir = OUTPUT_ROOT / subject
    feature_path = subject_dir / "csp_features.npz"
    metadata_path = subject_dir / "csp_metadata.json"

    if not feature_path.exists():
        raise FileNotFoundError(f"CSP feature file not found: {feature_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"CSP metadata file not found: {metadata_path}")

    data = np.load(feature_path)
    features = data["features"]
    labels = data["labels"].astype(int)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return features, labels, metadata


def build_rl_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    reward_success: float = 1.0,
    reward_fail: float = -1.0,
    terminal_every_step: bool = True,
) -> Dict[str, np.ndarray]:
    """将特征和标签转换为离线 RL 样本。

    状态: CSP 特征向量
    动作: 标签（整数事件码）
    奖励: 成功 -> reward_success, 失败 -> reward_fail (这里假定都是成功样本)
    done: 每个样本视为独立 episode 的结束
    """
    states = features.astype(np.float32)
    actions = labels.astype(np.int64)
    rewards = np.full_like(actions, reward_success, dtype=np.float32)

    # 初期使用简单设置：每个样本单步结束
    if terminal_every_step:
        dones = np.ones_like(actions, dtype=np.float32)
    else:
        dones = np.zeros_like(actions, dtype=np.float32)

    dataset = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }
    return dataset


def save_rl_dataset(
    subject: str,
    dataset: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    reward_success: float,
    reward_fail: float,
    terminal_every_step: bool,
) -> Path:
    """保存 RL 数据集到 npz，并写入元数据。"""
    subject_dir = OUTPUT_ROOT / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    out_path = subject_dir / "rl_dataset.npz"
    np.savez(
        out_path,
        states=dataset["states"],
        actions=dataset["actions"],
        rewards=dataset["rewards"],
        dones=dataset["dones"],
        reward_success=reward_success,
        reward_fail=reward_fail,
        terminal_every_step=terminal_every_step,
        classes=np.unique(dataset["actions"]),
    )

    info = {
        "subject": subject,
        "num_samples": int(dataset["states"].shape[0]),
        "state_dim": int(dataset["states"].shape[1]),
        "actions": [int(a) for a in np.unique(dataset["actions"])],
        "reward_success": reward_success,
        "reward_fail": reward_fail,
        "terminal_every_step": terminal_every_step,
        "source_metadata": metadata,
        "file": out_path.name,
    }
    metadata_path = subject_dir / "rl_dataset_meta.json"
    metadata_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline RL dataset from CSP features.")
    parser.add_argument("--subject", default="A01T", help="被试编号（默认 A01T）")
    parser.add_argument("--reward-success", type=float, default=1.0, help="正确样本奖励")
    parser.add_argument("--reward-fail", type=float, default=-1.0, help="错误样本奖励")
    parser.add_argument(
        "--terminal-every-step",
        action="store_true",
        help="是否将每个样本视为 episode 结束（默认为 True）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, labels, meta = load_csp_artifacts(args.subject)
    dataset = build_rl_dataset(
        features,
        labels,
        reward_success=args.reward_success,
        reward_fail=args.reward_fail,
        terminal_every_step=args.terminal_every_step or True,
    )
    out_path = save_rl_dataset(
        args.subject,
        dataset,
        meta,
        args.reward_success,
        args.reward_fail,
        args.terminal_every_step or True,
    )
    print(f"Saved RL dataset to {out_path}")


if __name__ == "__main__":
    main()

