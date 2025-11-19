"""OVR-CSP 原型演示脚本。

讲解建议：
    - 本脚本验证“预处理后的 epoch 数据 + 传统特征提取/分类器”能否得到合理结果，
      作为后续 DQN 强化学习的性能基线。
    - 输入数据来自 `preprocessing_pipeline.py` 保存的 `*_epochs-epo.fif`。
    - 核心流程：CSP 空间滤波 -> LDA 分类 -> StratifiedKFold 交叉验证。
    - 运行命令示例：`python scripts/ovr_csp_demo.py --subject A01T --components 6 --splits 5`
    - 输出：终端打印各折准确率、均值±标准差、混淆矩阵；另保存 JSON 结果便于报告引用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path  # 统一处理项目内的相对路径

import numpy as np
from mne import read_epochs  # 直接读取预处理保存的 Epochs
from mne.decoding import CSP  # MNE 内置的 Common Spatial Pattern 变换
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 线性判别分析作为分类器
from sklearn.metrics import classification_report, confusion_matrix  # 评估指标
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score  # 分层交叉验证工具
from sklearn.pipeline import make_pipeline  # 将 CSP 与 LDA 串成一体

# === 讲解建议 ===
# PROJECT_ROOT / OUTPUT_ROOT：与预处理脚本共享同一目录，直接复用已生成的中间文件。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "preprocessing"


# === 讲解建议 ===
# load_epochs：封装读取 Epochs 的逻辑，返回特征矩阵 X、标签 y、事件映射字典，及文件路径。
def load_epochs(subject: str) -> tuple[np.ndarray, np.ndarray, dict[str, int], Path]:
    """读取指定被试的 Epochs FIF 文件并返回数据/标签/事件字典。"""
    ep_path = OUTPUT_ROOT / subject / f"{subject}_epochs-epo.fif"
    if not ep_path.exists():
        raise FileNotFoundError(
            f"Epochs file not found for subject {subject}: {ep_path}"
        )
    epochs = read_epochs(ep_path.as_posix(), preload=True, verbose="ERROR")
    X = epochs.get_data()  # 形状: trials x channels x samples
    y = epochs.events[:, -1]  # 事件标签（整数值）
    event_id = epochs.event_id
    return X, y, event_id, ep_path


# === 讲解建议 ===
# run_ovr_csp：搭建 [CSP → LDA] pipeline，并执行交叉验证。
#   - CSP(n_components)：提取每个类别与其他类别之间差异最大的空间滤波器；
#   - LDA：线性分类，配合 shrinkage 可以稳定协方差估计；
#   - StratifiedKFold：保持各类样本比例一致，评估结果更可信。
def run_ovr_csp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_components: int = 6,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    """执行 StratifiedKFold 交叉验证并返回结果字典。"""
    csp = CSP(n_components=n_components, reg="oas", log=True)
    # LDA 搭配 Ledoit-Wolf shrinkage，适合样本数有限的高维数据
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    pipeline = make_pipeline(csp, lda)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=1)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=1)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)

    return {
        "scores": scores.tolist(),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


# === 讲解建议 ===
# save_results：将 metrics 写入 JSON，供周报/会议引用或进一步可视化。
def save_results(subject: str, metrics: dict[str, object]) -> Path:
    """将评估结果保存为 JSON 文件并返回路径。"""
    subject_dir = OUTPUT_ROOT / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    out_path = subject_dir / "csp_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return out_path


# === 讲解建议 ===
# parse_args：支持命令行参数，方便灵活测试不同被试、CSP 分量数或折数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OVR-CSP evaluation.")
    parser.add_argument(
        "--subject",
        default="A01T",
        help="被试文件名（不含扩展名），默认 A01T。",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=6,
        help="CSP 特征数量（每个类一对剩余选取的空间滤波器数）。",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="StratifiedKFold 折数。",
    )
    return parser.parse_args()


# === 讲解建议 ===
# main：串联所有步骤，打印关键信息并保存指标。
# 可在演示时逐条解释：数据形状→事件映射→各折准确率→平均值±方差→混淆矩阵→结果保存路径。
def main() -> None:
    args = parse_args()
    X, y, event_id, ep_path = load_epochs(args.subject)
    print(f"Loaded epochs from: {ep_path}")
    print(f"Data shape: {X.shape} (trials, channels, samples)")
    print(f"Event ID mapping: {event_id}")

    metrics = run_ovr_csp(
        X,
        y,
        n_components=args.components,
        n_splits=args.splits,
    )
    print("Cross-validation scores:", metrics["scores"])
    print(
        f"Mean accuracy: {metrics['mean_score']:.3f} ± {metrics['std_score']:.3f}"
    )
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))

    metrics_path = save_results(args.subject, metrics)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
