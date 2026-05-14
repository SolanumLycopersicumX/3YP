#!/usr/bin/env python3
"""
OpenBCI/BrainFlow 实时 EEG 数据可视化测试

测试内容:
1. 启动 BrainFlow SYNTHETIC_BOARD 数据流
2. 采集 4 秒 EEG epoch
3. 可视化:
   - 原始 EEG 波形 (16通道)
   - 频谱分析 (FFT)
   - CTNet 格式数据 (22通道, z-score归一化后)
   - CTNet 实时分类结果

用法:
    python scripts/test_openbci_visual.py

创建时间: 2026-03-10
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import os
os.environ['MPLBACKEND'] = 'Agg'  # 必须在 matplotlib 导入前设置
os.environ['DISPLAY'] = ''  # 阻止 TkAgg 初始化

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.openbci_stream import OpenBCIStream


def test_and_visualize():
    print("=" * 70)
    print("OpenBCI/BrainFlow 实时 EEG 数据可视化测试")
    print("=" * 70)

    output_dir = _ROOT / "outputs" / "openbci_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Step 1: 初始化并启动数据流
    # ================================================================
    print("\n[Step 1] 初始化 OpenBCI 数据流 (SYNTHETIC_BOARD)...")
    stream = OpenBCIStream(board_type="synthetic")
    stream.start()

    # 等待缓冲区预热
    time.sleep(1.0)

    # ================================================================
    # Step 2: 采集原始 EEG 数据
    # ================================================================
    print("\n[Step 2] 采集原始 EEG 数据 (4 秒)...")
    raw_epoch = stream.get_eeg_epoch(duration_sec=4.0, apply_filter=False)
    print(f"  原始数据 shape: {raw_epoch.shape}")
    print(f"  通道数: {raw_epoch.shape[0]}")
    print(f"  样本数: {raw_epoch.shape[1]}")
    print(f"  数据范围: [{raw_epoch.min():.2f}, {raw_epoch.max():.2f}] μV")

    # 采集滤波后的数据
    print("\n[Step 3] 采集滤波后 EEG 数据 (4 秒, 8-30 Hz bandpass)...")
    filtered_epoch = stream.get_eeg_epoch(duration_sec=4.0, apply_filter=True)
    print(f"  滤波后数据 shape: {filtered_epoch.shape}")
    print(f"  数据范围: [{filtered_epoch.min():.2f}, {filtered_epoch.max():.2f}] μV")

    # 采集 CTNet 格式数据
    print("\n[Step 4] 获取 CTNet 格式数据 (22ch × 1000samples)...")
    ctnet_epoch = stream.get_eeg_epoch_for_ctnet(duration_sec=4.0)
    print(f"  CTNet 数据 shape: {ctnet_epoch.shape}")
    print(f"  期望 shape: (1, 1, 22, 1000)")
    print(f"  数据范围: [{ctnet_epoch.min():.4f}, {ctnet_epoch.max():.4f}]")

    # ================================================================
    # Step 3: CTNet 分类
    # ================================================================
    print("\n[Step 5] CTNet 实时分类...")
    import torch
    import CTNet_model
    import __main__
    for attr_name in dir(CTNet_model):
        if not attr_name.startswith('_'):
            setattr(__main__, attr_name, getattr(CTNet_model, attr_name))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = _ROOT / "A_heads_2_depth_6" / "model_1.pth"

    if model_path.exists():
        model = torch.load(str(model_path), map_location=device, weights_only=False)
        model.eval()
        print(f"  ✓ CTNet 模型已加载: {model_path}")

        with torch.no_grad():
            x = torch.from_numpy(ctnet_epoch).float().to(device)
            _, logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))

        class_names = ["Left Hand", "Right Hand", "Both Feet", "Tongue"]
        print(f"\n  🧠 分类结果:")
        for i, (name, prob) in enumerate(zip(class_names, probs)):
            marker = " ← 预测" if i == pred_class else ""
            bar = "█" * int(prob * 30)
            print(f"    {name:12s}: {prob:5.1%} {bar}{marker}")
    else:
        print(f"  ⚠ CTNet 模型不存在: {model_path}")
        probs = None
        pred_class = None
        class_names = ["Left Hand", "Right Hand", "Both Feet", "Tongue"]

    stream.stop()

    # ================================================================
    # Step 4: 可视化
    # ================================================================
    print("\n[Step 6] 生成可视化...")

    sfreq = stream.sampling_rate
    n_ch = raw_epoch.shape[0]
    n_samp = raw_epoch.shape[1]
    t = np.arange(n_samp) / sfreq

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("OpenBCI/BrainFlow Real-time EEG Data Test\n"
                 f"Board: SYNTHETIC_BOARD | {n_ch} channels | {sfreq} Hz | 4s epoch",
                 fontsize=14, fontweight='bold')

    # ---- Panel 1: 原始 EEG 波形 ----
    ax1 = fig.add_subplot(2, 2, 1)
    offsets = np.arange(n_ch) * 100  # 通道间距
    for ch in range(n_ch):
        ax1.plot(t, raw_epoch[ch] + offsets[ch], linewidth=0.5, alpha=0.8)
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Channel", fontsize=11)
    ax1.set_title("① Raw EEG Waveforms (16 channels)", fontweight='bold')
    ax1.set_yticks(offsets[::2])
    ax1.set_yticklabels([f"Ch{i+1}" for i in range(0, n_ch, 2)])
    ax1.grid(True, alpha=0.2)

    # ---- Panel 2: 滤波后 EEG 波形 ----
    ax2 = fig.add_subplot(2, 2, 2)
    offsets_f = np.arange(filtered_epoch.shape[0]) * 50
    for ch in range(filtered_epoch.shape[0]):
        ax2.plot(t, filtered_epoch[ch] + offsets_f[ch], linewidth=0.5, alpha=0.8)
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Channel", fontsize=11)
    ax2.set_title("② Filtered EEG (8-30 Hz bandpass)", fontweight='bold')
    ax2.set_yticks(offsets_f[::2])
    ax2.set_yticklabels([f"Ch{i+1}" for i in range(0, filtered_epoch.shape[0], 2)])
    ax2.grid(True, alpha=0.2)

    # ---- Panel 3: 频谱分析 (FFT) ----
    ax3 = fig.add_subplot(2, 2, 3)
    for ch in range(min(4, n_ch)):
        freqs = np.fft.rfftfreq(n_samp, d=1.0/sfreq)
        fft_vals = np.abs(np.fft.rfft(raw_epoch[ch]))
        fft_db = 20 * np.log10(fft_vals + 1e-10)
        ax3.plot(freqs, fft_db, alpha=0.7, linewidth=1.2, label=f"Ch{ch+1}")

    ax3.axvspan(8, 30, alpha=0.15, color='green', label='8-30 Hz (MI band)')
    ax3.set_xlabel("Frequency (Hz)", fontsize=11)
    ax3.set_ylabel("Power (dB)", fontsize=11)
    ax3.set_title("③ Power Spectrum (Raw)", fontweight='bold')
    ax3.set_xlim(0, 60)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ---- Panel 4: CTNet 分类概率 ----
    ax4 = fig.add_subplot(2, 2, 4)
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

    if probs is not None:
        bars = ax4.barh(class_names, probs, color=colors, alpha=0.85, height=0.5)
        for bar, prob in zip(bars, probs):
            ax4.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1%}', va='center', fontsize=12, fontweight='bold')
        ax4.set_xlim(0, 1.15)
        ax4.set_xlabel("Probability", fontsize=11)
        ax4.set_title(f"④ CTNet Classification → {class_names[pred_class]}",
                     fontweight='bold')
        ax4.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (25%)')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "CTNet model not found", ha='center', va='center',
                fontsize=14, transform=ax4.transAxes)
        ax4.set_title("④ CTNet Classification", fontweight='bold')

    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # 保存
    save_path = output_dir / "openbci_eeg_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ 可视化已保存: {save_path}")

    # 显示
    try:
        plt.show()
        print("  ✓ 可视化窗口已打开")
    except Exception as e:
        print(f"  ⚠ 无法显示窗口: {e}")
        plt.close()

    # ================================================================
    # 总结
    # ================================================================
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"  ✓ BrainFlow 数据流: 正常 (SYNTHETIC_BOARD)")
    print(f"  ✓ 原始 EEG 采集: {raw_epoch.shape}")
    print(f"  ✓ 带通滤波 (8-30Hz): {filtered_epoch.shape}")
    print(f"  ✓ CTNet 格式适配: {ctnet_epoch.shape}")
    if probs is not None:
        print(f"  ✓ CTNet 分类: {class_names[pred_class]} ({probs[pred_class]:.1%})")
    print(f"  ✓ 可视化: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    test_and_visualize()
