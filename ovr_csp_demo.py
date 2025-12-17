"""OVR-CSP 原型演示脚本。

讲解建议：
    - 目标：验证“预处理后的 Epoch + 经典 OVR-CSP+LDA”可达到何种基线表现，为 RL 阶段提供参照。
    - 输入：`preprocessing_pipeline.py` 输出的 `*_epochs-epo.fif`，包含 4 类（左手/右手/脚/舌）飞秒窗数据。
    - 流程：加载 Epoch → 构建 [CSP → LDA] 管线 → 5 折 StratifiedKFold → 保存指标、特征与模型。
    - 运行：`python scripts/ovr_csp_demo.py --subject A01T --components 6 --splits 5`
    - 产物：控制台打印准确率/混淆矩阵；目录生成 `csp_features.npz`、`csp_pipeline.joblib`、`csp_patterns.png` 等文件，可直接汇报或交给 RL 使用。
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path  # 统一处理项目内的相对路径
from typing import Any, Dict

import matplotlib.pyplot as plt  # 保存 CSP 空间模式图
import numpy as np
from mne import read_epochs  # 直接读取预处理保存的 Epochs
from mne.decoding import CSP  # MNE 内置的 Common Spatial Pattern 变换
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 线性判别分析作为分类器
from sklearn.metrics import classification_report, confusion_matrix  # 评估指标
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)  # 分层交叉验证工具
from sklearn.pipeline import Pipeline, make_pipeline  # 将 CSP 与 LDA 串成一体

try:  # joblib 可序列化 sklearn pipeline；若环境缺失则回退到 pickle
    import joblib
except ImportError:  # pragma: no cover - 仅在极端环境下触发
    joblib = None

# === 讲解建议 ===
# PROJECT_ROOT / OUTPUT_ROOT：与预处理脚本共享同一目录，直接复用已生成的中间文件。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "preprocessing"


# === 讲解建议 ===
# load_epochs：封装读取 Epochs 的逻辑，返回 Epochs、数据、标签、事件映射及文件路径。
def load_epochs(subject: str) -> tuple:
    """读取指定被试的 Epochs FIF 文件并返回核心数据。"""
    ep_path = OUTPUT_ROOT / subject / f"{subject}_epochs-epo.fif"  # Epoch 文件位置
    if not ep_path.exists():
        raise FileNotFoundError(
            f"Epochs file not found for subject {subject}: {ep_path}"
        )
    epochs = read_epochs(ep_path.as_posix(), preload=True, verbose="ERROR")  # MNE 读取 Epochs
    X = epochs.get_data()  # trials × channels × samples，供 sklearn 使用
    y = epochs.events[:, -1]  # 事件标签（整数值）
    event_id = epochs.event_id  # 事件字典（如 'left_hand':7）
    return epochs, X, y, event_id, ep_path


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
) -> tuple[dict[str, object], Pipeline]:
    """执行 StratifiedKFold 交叉验证并返回结果字典与最终训练好的 pipeline。"""
    # 1. 构造 CSP 滤波器：reg='oas' 使用 Ledoit-Wolf 协方差估计，log=True 输出 log-variance 特征
    csp = CSP(n_components=n_components, reg="oas", log=True)
    # 2. LDA 分类器，solver='lsqr' + shrinkage='auto' 适合小样本的脑电任务
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    pipeline = make_pipeline(csp, lda)  # 3. 将两者串接成 sklearn pipeline
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)  # 分层交叉验证，维持类别比例

    scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=1)  # 返回每折准确率
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=1)  # 返回交叉验证预测标签
    cm = confusion_matrix(y, y_pred)  # 混淆矩阵
    report = classification_report(y, y_pred, output_dict=True)  # 精确率/召回率/F1 指标
    pipeline.fit(X, y)  # 4. 用全量数据训练一次，用于导出最终模型、滤波器与特征

    metrics = {
        "scores": scores.tolist(),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    return metrics, pipeline


# === 讲解建议 ===
# export_csp_artifacts：将训练好的 CSP 滤波器、模式矩阵与特征落盘，供 RL 与后续实验直接加载。
def export_csp_artifacts(
    subject: str,
    pipeline: Pipeline,
    epochs,
    X: np.ndarray,
    y: np.ndarray,
    event_id: dict[str, int],
) -> Dict[str, Path]:
    """保存 CSP 相关产物并返回路径字典。"""
    subject_dir = OUTPUT_ROOT / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    csp: CSP = pipeline.named_steps["csp"]  # 从 pipeline 中取出训练好的 CSP 模块
    features = csp.transform(X)  # 生成 log-variance 特征（即 M features）

    # 1. 保存特征 + 标签
    features_path = subject_dir / "csp_features.npz"
    np.savez(
        features_path,
        features=features,
        labels=y,
        classes=csp.classes_,
    )

    # 2. 保存空间滤波器与模式矩阵，方便可视化/复现
    filters_path = subject_dir / "csp_filters.npy"
    np.save(filters_path, csp.filters_)

    patterns_path = subject_dir / "csp_patterns.npy"
    np.save(patterns_path, csp.patterns_)

    # 3. 保存整条 pipeline（CSP+LDA），可直接用于预测
    pipeline_path = subject_dir / "csp_pipeline.joblib"
    if joblib is not None:
        joblib.dump(pipeline, pipeline_path)
    else:  # 回退到 pickle，确保任意环境都能保存模型
        pipeline_path = pipeline_path.with_suffix(".pkl")
        with pipeline_path.open("wb") as f:
            pickle.dump(pipeline, f)

    # 4. 绘制 CSP 空间模式图（查看每个滤波器作用在何处）
    plots_dir = subject_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig = csp.plot_patterns(epochs.info, ch_type="eeg", show=False)
    patterns_fig_path = plots_dir / "csp_patterns.png"
    fig.savefig(patterns_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. 整理元信息，方便后续脚本读取或在汇报中引用
    metadata_path = subject_dir / "csp_metadata.json"
    metadata: Dict[str, Any] = {
        "subject": subject,
        "n_samples": int(features.shape[0]),
        "n_components": int(csp.n_components),
        "feature_dim": int(features.shape[1]),
        "classes": [int(cls) for cls in np.unique(y)],
        "event_id": {str(k): int(v) for k, v in event_id.items()},
        "log_transform": bool(csp.log),
        "regularization": str(csp.reg),
        "features_file": features_path.name,
        "filters_file": filters_path.name,
        "patterns_file": patterns_path.name,
        "pipeline_file": pipeline_path.name,
        "pattern_figure": patterns_fig_path.name,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {
        "features": features_path,
        "filters": filters_path,
        "patterns": patterns_path,
        "pipeline": pipeline_path,
        "metadata": metadata_path,
        "pattern_figure": patterns_fig_path,
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
    epochs, X, y, event_id, ep_path = load_epochs(args.subject)
    print(f"Loaded epochs from: {ep_path}")  # 提示使用了哪个被试的数据
    print(f"Data shape: {X.shape} (trials, channels, samples)")  # 数据尺寸核对
    print(f"Event ID mapping: {event_id}")  # 事件代码与标签的对应关系

    metrics, pipeline = run_ovr_csp(
        X,
        y,
        n_components=args.components,
        n_splits=args.splits,
    )
    print("Cross-validation scores:", metrics["scores"])  # 打印每折准确率
    print(f"Mean accuracy: {metrics['mean_score']:.3f} ± {metrics['std_score']:.3f}")  # 平均值 ± 标准差
    print("Confusion matrix:")  # 四类混淆矩阵
    print(np.array(metrics["confusion_matrix"]))

    artifacts = export_csp_artifacts(args.subject, pipeline, epochs, X, y, event_id)
    metrics["artifacts"] = {name: path.as_posix() for name, path in artifacts.items()}
    print("Exported CSP artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")

    metrics_path = save_results(args.subject, metrics)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
