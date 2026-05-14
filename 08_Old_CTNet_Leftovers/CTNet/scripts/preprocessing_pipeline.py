"""基于 MNE 的 EEG 预处理脚本入口。

本脚本包含我们当前工作重点的整个流水线，涵盖：
    1. 带通滤波（8-30 Hz，Motor Imagery 常用频段）
    2. ICA 伪迹识别与移除（主要针对眼电 EOG）
    3. Epoch 切片（按注释标签划窗）
    4. 输出保存与可视化（后续可接 CSP / 分类器）

所有步骤均依赖 MNE-Python 库。会议展示时可以按 main() 函数顺序逐步讲解。
"""

# === 讲解建议 ===
# 说明：启用 __future__ 中的 annotations 语法，让类型标注支持延迟解析；
# 这有助于下面的函数都能写出更清晰的类型提示，便于团队协作与 IDE 提示。
from __future__ import annotations

# === 讲解建议 ===
# 说明：引入所有后续处理所需的标准库与第三方库。
# 可以按顺序向教授说明每个库负责的功能：json 负责记录指标，Path 处理路径，
# typing 提供类型检查，matplotlib 画图，mne 处理脑电，ICA/Montage 用于伪迹处理与定位。
import json  # 标准库，用于写入 ICA/EOG 统计摘要
from pathlib import Path  # 统一管理文件路径
from typing import Iterable, Literal  # 类型标注，帮助阅读/IDE 补全

import matplotlib.pyplot as plt  # matplotlib 绘图库，用于保存 ICA 可视化图
import mne  # MNE-Python，处理 EEG/MEG 数据的核心库
from mne.channels import make_standard_montage  # 用于载入标准 10-20 蒙太奇坐标
from mne.preprocessing import ICA  # MNE 的独立成分分析实现

# === 讲解建议 ===
# 说明：这里集中定义全局常量，用于描述数据路径和通道映射策略。
# 强调这一部分帮助其他函数保持简洁，也方便后续切换数据源或修改通道命名。
# Default to the BCICIV 2a dataset that is already present in the repo.
DATA_ROOT = Path(__file__).resolve().parents[1] / "BCICIV_2a_gdf"  # 原始 GDF 数据目录
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "preprocessing"  # 预处理输出目录
# CHANNEL_RENAME_MAP 用于把 BCIC 原始命名（EEG-XX）映射到标准 10-20 名称，方便设定 montage 和展示拓扑图
CHANNEL_RENAME_MAP = {
    "EEG-Fz": "Fz",
    "EEG-0": "FC3",
    "EEG-1": "FC1",
    "EEG-2": "FCz",
    "EEG-3": "FC2",
    "EEG-4": "FC4",
    "EEG-5": "C5",
    "EEG-C3": "C3",
    "EEG-6": "C1",
    "EEG-Cz": "Cz",
    "EEG-7": "C2",
    "EEG-C4": "C4",
    "EEG-8": "C6",
    "EEG-9": "CP3",
    "EEG-10": "CP1",
    "EEG-11": "CPz",
    "EEG-12": "CP2",
    "EEG-13": "CP4",
    "EEG-14": "P1",
    "EEG-Pz": "Pz",
    "EEG-15": "P2",
    "EEG-16": "POz",
}
EOG_CHANNELS = ["EOG-left", "EOG-central", "EOG-right"]  # 记录 EOG 通道名称，用于 ICA 伪迹检测


# === 讲解建议 ===
# 向教授说明：`load_raw` 封装了对单个 GDF 文件的读取。
# 重点强调参数 `preload` 和 `verbose` 能控制内存占用与输出噪声，
# 是整个流程的入口，所有后续函数都基于这里产生的 Raw 对象。
def load_raw(
    subject: str = "A01T.gdf",
    *,
    preload: bool = False,
    verbose: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | str = "ERROR",
) -> mne.io.BaseRaw:
    """读取单个 GDF 文件并返回 MNE Raw 对象。

    mne.io.read_raw_gdf 会解析 GDF 原始数据及注释，preload=False 表示延迟加载节省内存。
    """
    gdf_path = DATA_ROOT / subject
    if not gdf_path.exists():
        raise FileNotFoundError(f"GDF file not found: {gdf_path}")
    raw = mne.io.read_raw_gdf(gdf_path.as_posix(), preload=preload, verbose=verbose)
    return raw


# === 讲解建议 ===
# 说明：`prepare_channels` 负责将原始原始通道转换成标准格式，并且给 EOG 打标签。
# 强调这是获得可视化效果和正确 ICA 处理的关键步骤，
# 它在 main() 中紧随读取数据之后调用，为后续滤波、ICA 奠定基础。
def prepare_channels(
    raw: mne.io.BaseRaw,
    *,
    rename_map: dict[str, str] | None = None,
    eog_channels: Iterable[str] | None = None,
    montage_name: str | None = "standard_1020",
    verbose: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | str = "ERROR",
) -> mne.io.BaseRaw:
    """对 Raw 进行通道预处理：重命名、设置 EOG 类型并指定标准蒙太奇。

    - rename_channels：将 BCIC 的 EEG-XX 改为标准名称（如 C3/C4），便于后续可视化。
    - set_channel_types：把 EOG 通道标记成 'eog' 类型，MNE 会在滤波/ICA 中自动处理。
    - make_standard_montage：加载 10-20 体系坐标，set_montage 后即可绘制拓扑图。
    """
    rename_map = rename_map or CHANNEL_RENAME_MAP
    eog_channels = list(eog_channels or EOG_CHANNELS)
    raw = raw.copy()  # copy 确保不修改传入的 Raw 原对象
    available_map = {old: new for old, new in rename_map.items() if old in raw.ch_names}
    if available_map:
        # 重命名 EEG 通道，保证后续 montage 中能匹配标准名称
        raw.rename_channels(available_map)
    channel_types = {ch: "eog" for ch in eog_channels if ch in raw.ch_names}
    if channel_types:
        # 将眼动通道显式声明为 eog 类型，MNE 的 ICA/滤波会自动正确处理
        raw.set_channel_types(channel_types, verbose=verbose)
    if montage_name:
        try:
            montage = make_standard_montage(montage_name)  # 加载标准 10-20 坐标
            raw.set_montage(montage, match_case=False, on_missing="warn", verbose=verbose)
        except Exception as err:
            print(f"Warning: setting montage failed ({err}). Continuing without montage.")
    return raw


# === 讲解建议 ===
# 说明：`bandpass_filter` 执行 8-30Hz 的带通滤波，用于保留运动想象相关频段。
# 可以提醒教授这是经典的 EEG 预处理步骤，输出给后续的 ICA 使用，
# 同时解释参数为什么选零相位 FIR，以避免动作想象信号的相位被扭曲。
def bandpass_filter(
    raw: mne.io.BaseRaw,
    *,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    picks: Literal["eeg", "meg", "eog", "misc"] | list[str] | None = "eeg",
    filter_length: str | float = "auto",
    l_trans_bandwidth: float = 2.0,
    h_trans_bandwidth: float = 2.0,
    method: str = "fir",
    phase: str = "zero-double",
    fir_window: str = "hamming",
    verbose: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | str = "ERROR",
) -> mne.io.BaseRaw:
    """对 Raw 复制后进行 8-30Hz 带通滤波。

    参数说明：
        raw: 输入的 Raw 对象；
        l_freq/h_freq: 截止频率（Hz）；
        picks: 指定只过滤 EEG 通道；
        method/phase: 选用零相位 FIR 滤波器，避免信号相位偏移；
        l_trans_bandwidth/h_trans_bandwidth: 过渡带宽，用于控制 FIR 阶数。
    MNE 内部调用 scipy.signal 生成 FIR 滤波器，并在内存中原位滤波。
    """
    raw_filt = raw.copy().load_data()  # MNE 过滤需先加载到内存中
    raw_filt.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks=picks,
        filter_length=filter_length,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
        method=method,
        phase=phase,
        fir_window=fir_window,
        verbose=verbose,
    )
    return raw_filt


# === 讲解建议 ===
# 说明：`run_ica` 负责拟合 FastICA 模型，提取独立成分。
# 可强调我们只对 EEG 通道做 ICA（排除 EOG），并解释 n_components 的自适应处理，
# 这个函数的输出被后续伪迹检测、清洗与保存步骤广泛引用。
def run_ica(
    raw: mne.io.BaseRaw,
    *,
    n_components: float | int | None = 0.99,
    method: str = "fastica",
    random_state: int = 97,
    max_iter: str | int = "auto",
    decim: int = 3,
    reject_by_annotation: bool = True,
    verbose: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | str = "ERROR",
) -> ICA:
    """在滤波后的数据上拟合 ICA 并返回 ICA 对象。

    - raw.copy().pick_types：仅取 EEG 通道，确保 ICA 不处理 EOG。
    - MNE.preprocessing.ICA 使用 sklearn 的 FastICA/FastICA 实现。
    - decim=3 表示每隔 3 个采样点取一次，提高速度。
    """
    eeg_channel_count = len(raw.copy().pick_types(eeg=True, exclude=[]).ch_names)  # 统计可用于 ICA 的 EEG 通道数
    if isinstance(n_components, float):
        # Convert variance ratio to an explicit component count with sane bounds.
        suggested = int(eeg_channel_count * n_components)  # 根据方差比例估算组件数
        n_components = max(5, suggested)  # 至少保留 5 个成分以避免过度压缩

    if n_components is None or n_components > eeg_channel_count:
        n_components = eeg_channel_count  # 不允许成分数超过通道数（FastICA 限制）

    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter=max_iter,
        verbose=verbose,
    )
    ica.fit(
        raw,
        decim=decim,
        reject_by_annotation=reject_by_annotation,
        verbose=verbose,
    )
    return ica


# === 讲解建议 ===
# 说明：`detect_eog_artifacts` 使用 EOG 通道与 ICA 成分的相关性来找伪迹。
# 可以告诉教授这是自动化判断眼动成分的关键函数，返回的组件编号会传入下一步的 apply_ica，
# 也会写入 JSON 供报告引用。
def detect_eog_artifacts(
    ica: ICA,
    raw: mne.io.BaseRaw,
    *,
    eog_channels: Iterable[str] | None = None,
    threshold: float = 3.0,
    verbose: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | str = "ERROR",
) -> tuple[list[int], dict[str, list[float]]]:
    """检测与 EOG 通道高度相关的 ICA 成分。

    调用 ica.find_bads_eog（MNE 内置方法），返回：
        - inds: 被判定为伪迹的成分编号；
        - scores: 每个成分与指定 EOG 通道的相关分数列表。
    """
    if eog_channels is None:
        eog_channels = [ch for ch in raw.ch_names if "EOG" in ch.upper()]  # 自动收集 Raw 中的 EOG 通道
    bad_inds: set[int] = set()  # 汇总需要排除的成分编号
    scores: dict[str, list[float]] = {}  # 记录每个 EOG 通道的相关系数向量
    for ch in eog_channels:
        inds, ch_scores = ica.find_bads_eog(
            raw,
            ch_name=ch,
            threshold=threshold,
            verbose=verbose,
        )
        bad_inds.update(int(i) for i in inds)  # 兼容 numpy.int 类型
        scores[ch] = ch_scores.tolist() if hasattr(ch_scores, "tolist") else list(ch_scores)  # 转成列表便于序列化
    return sorted(bad_inds), scores


# === 讲解建议 ===
# 说明：`apply_ica` 根据上一部检测出的成分实际移除伪迹，得到干净脑电。
# 重点说明我们使用复制以保护原数据，同时生成清洗后的 Raw 供 Epoch 和保存使用。
def apply_ica(
    ica: ICA,
    raw: mne.io.BaseRaw,
    *,
    exclude: Iterable[int],
    verbose: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | str = "ERROR",
) -> mne.io.BaseRaw:
    """根据排除的 ICA 成分将伪迹从 Raw 中剔除，返回清洗后的信号。"""
    raw_clean = raw.copy()  # 保留原始数据，避免覆盖
    ica_copy = ica.copy()  # 复制 ICA 模型，防止修改原对象的 exclude
    ica_copy.exclude = list(exclude)  # 指定要剔除的成分编号
    ica_copy.apply(raw_clean, verbose=verbose)  # 在 Raw 上应用逆变换，移除伪迹
    return raw_clean


# === 讲解建议 ===
# 说明：`save_pipeline_outputs` 将核心中间结果落盘，并输出图表。
# 可以向教授展示保存的文件如何被后续的 CSP、报告或可视化引用，
# 同时说明为什么要生成 ICA 拓扑图和 EOG 相关性图，方便会议上展示伪迹处理效果。
def save_pipeline_outputs(
    raw_clean: mne.io.BaseRaw,
    epochs: mne.Epochs,
    ica: ICA,
    excluded_components: Iterable[int],
    eog_scores: dict[str, list[float]],
) -> dict[str, Path]:
    """持久化预处理结果：保存 fif/JSON 文件并输出图表路径。

    - raw_clean.save / epochs.save / ica.save：MNE 内置写入函数，便于复用；
    - json.dump：记录 ICA 排除成分与 EOG 分数，方便会议汇报；
    - ica.plot_components / ica.plot_scores：matplotlib Figure，可保存拓扑图与评分图。
    返回值为保存路径字典，main() 会逐行打印。
    """
    subject_name = Path(raw_clean.filenames[0]).stem
    subject_dir = OUTPUT_ROOT / subject_name  # 当前被试输出目录
    plots_dir = subject_dir / "plots"  # 图表子目录
    subject_dir.mkdir(parents=True, exist_ok=True)  # 创建目录（存在则跳过）
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_path = subject_dir / f"{subject_name}_clean_raw.fif"
    epochs_path = subject_dir / f"{subject_name}_epochs-epo.fif"
    ica_path = subject_dir / f"{subject_name}_ica.fif"
    summary_path = subject_dir / "ica_eog_summary.json"

    raw_clean.save(raw_path.as_posix(), overwrite=True)  # 保存清洗后的连续数据
    epochs.save(epochs_path.as_posix(), overwrite=True)  # 保存 epoch 数据
    ica.save(ica_path.as_posix(), overwrite=True)  # 记录 ICA 权重，可复现伪迹剔除过程

    summary = {
        "subject": subject_name,
        "excluded_components": list(excluded_components),
        "eog_scores": eog_scores,
        "n_epochs": len(epochs),
        "epoch_event_id": epochs.event_id,
        "sfreq": raw_clean.info["sfreq"],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)  # 输出 JSON 摘要，供报告引用

    plot_paths: dict[str, Path] = {}
    # ICA component overview
    plot_paths: dict[str, Path] = {}
    inst_for_plot: mne.io.BaseRaw | None = raw_clean.copy()  # 用复制品绘制拓扑图，避免修改原数据
    try:
        montage = make_standard_montage("standard_1020")
        rename_map = {}
        for ch in montage.ch_names:
            prefixed = f"EEG-{ch}"
            if prefixed in inst_for_plot.ch_names:
                rename_map[ch] = prefixed  # 当 Raw 中仍含有 EEG- 前缀时做映射
        if rename_map:
            montage = montage.copy()
            montage.rename_channels(rename_map)
        inst_for_plot.set_montage(montage, on_missing="warn", match_case=False)
    except Exception as err:
        print(f"Warning: could not set montage for ICA plots ({err}).")
        inst_for_plot = None

    if inst_for_plot is not None:
        try:
            fig_components = ica.plot_components(inst=inst_for_plot, show=False)
        except RuntimeError as err:
            print(f"Skipping ICA component topomap plot ({err}).")
        else:
            figures = fig_components if isinstance(fig_components, (list, tuple)) else [fig_components]  # MNE 可能返回列表
            for idx, fig in enumerate(figures):
                components_path = plots_dir / f"ica_components_{idx:02d}.png"
                fig.savefig(components_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
            plot_paths["components"] = plots_dir / "ica_components_00.png"
    else:
        print("Skipping ICA component topomap plot due to missing montage information.")

    for ch_name, scores in eog_scores.items():
        fig_scores = ica.plot_scores(
            scores,
            exclude=list(excluded_components),
            title=f"EOG correlation ({ch_name})",
            show=False,
        )
        score_path = plots_dir / f"ica_scores_{ch_name.replace('/', '_')}.png"
        fig_scores.savefig(score_path, dpi=150, bbox_inches="tight")
        plt.close(fig_scores)
        plot_paths[f"scores_{ch_name}"] = score_path

    return {
        "raw_clean": raw_path,
        "epochs": epochs_path,
        "ica": ica_path,
        "summary": summary_path,
        **plot_paths,
    }


# === 讲解建议 ===
# 说明：`epoch_data` 根据注释切片出固定长度的窗口（Epoch），
# 强调它把连续脑电转换为试次样本，后续 CSP、分类器都依赖这些 Epoch 数据。
# 也可解释默认使用 BCI-IV-2a 的四个事件码，并支持自定义 event_id。
def epoch_data(
    raw: mne.io.BaseRaw,
    *,
    event_id: dict[str, int] | None = None,
    tmin: float = 0.0,
    tmax: float = 4.0,
    baseline: tuple[float | None, float | None] | None = None,
    picks: Literal["eeg", "meg", "eog", "misc"] | list[str] | None = "eeg",
    preload: bool = True,
    decim: int = 1,
    reject: dict[str, float] | None = None,
    event_repeated: Literal["drop", "merge", "repeat"] = "drop",
    verbose: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | str = "ERROR",
) -> mne.Epochs:
    """根据注释生成 Epochs（窗口化数据）。

    - mne.events_from_annotations：把注释字符串映射成事件码；
    - mne.Epochs：从 Raw 中截取 [tmin, tmax] 的时间窗，preload=True 便于后续处理；
    - event_id 默认为 BCI Competition IV 2a 的 4 类动作编码。
    """
    events, events_map = mne.events_from_annotations(raw, verbose=verbose)  # 从注释中提取事件矩阵
    if event_id is None:
        # Standard cue codes for BCI Competition IV 2a dataset.
        cue_map = {
            "left_hand": "769",
            "right_hand": "770",
            "foot": "771",
            "tongue": "772",
        }
        event_id = {
            label: events_map[code]
            for label, code in cue_map.items()
            if code in events_map
        }
    if not event_id:
        raise RuntimeError("No matching event codes found in annotations.")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks=picks,
        preload=preload,
        reject=reject,
        decim=decim,
        event_repeated=event_repeated,
        verbose=verbose,
    )
    return epochs


# === 讲解建议 ===
# 说明：`main` 函数串联前面的所有步骤，便于直接运行脚本演示完整流程。
# 向教授逐行解释输出信息与下游步骤的关系：读取→通道预处理→滤波→ICA→伪迹移除→Epoch→保存。
def main() -> None:
    """Quick diagnostic run to ensure data access works."""
    raw = load_raw(preload=False)  # 读取原始数据，延迟加载节省内存
    raw = prepare_channels(raw, verbose="WARNING")  # 通道重命名、EOG 标记、蒙太奇设置
    info = {
        "subject": raw.filenames[0],  # 原始文件路径
        "channels": raw.info["nchan"],  # 通道数量
        "sampling_rate": raw.info["sfreq"],  # 采样率 (Hz)
        "duration_sec": raw.n_times / raw.info["sfreq"],  # 记录时长 (秒)
    }
    print("Loaded recording:", info["subject"])
    print(f"Channels: {info['channels']} | sfreq: {info['sampling_rate']} Hz")
    print(f"Duration: {info['duration_sec']:.1f} s")
    print("First 5 annotation codes:", raw.annotations.description[:5])

    raw_filtered = bandpass_filter(raw, verbose="WARNING")  # 8-30Hz 带通滤波
    sample_data = raw_filtered.get_data(picks="eeg")[:1, :10]
    print("Bandpass 8-30 Hz applied (first channel, 10 samples):")
    print(sample_data)

    ica = run_ica(raw_filtered, verbose="WARNING")  # 拟合 ICA
    print(
        "ICA fitted: components=%s, explained=%0.2f"
        % (ica.n_components_, ica.pca_explained_variance_.sum())
    )

    eog_inds, eog_scores = detect_eog_artifacts(ica, raw_filtered, verbose="WARNING")  # 通过 EOG 找伪迹成分
    print("Detected EOG-related components:", eog_inds)
    if eog_inds:
        raw_clean = apply_ica(ica, raw_filtered, exclude=eog_inds, verbose="WARNING")  # 移除伪迹
        print("ICA applied. Data shape (cleaned):", raw_clean.get_data().shape)
    else:
        raw_clean = raw_filtered
        print("No ICA components marked for exclusion.")

    epochs = epoch_data(
        raw_clean,
        tmin=0.0,
        tmax=4.0,
        baseline=None,
        picks="eeg",
        preload=True,
        verbose="WARNING",
    )
    print("Epochs created:", epochs)
    counts = {label: len(epochs[label]) for label in epochs.event_id}
    print("Epoch counts per class:", counts)
    print("Epoch data shape:", epochs.get_data().shape)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    saved = save_pipeline_outputs(raw_clean, epochs, ica, eog_inds, eog_scores)  # 保存结果并生成图表
    print("Artifacts saved to:", saved["raw_clean"].parent)
    for key, path in saved.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
