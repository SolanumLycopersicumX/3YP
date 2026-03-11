#!/usr/bin/env python3
"""
OpenBCI EEG 数据流模块 (BrainFlow)

通过 BrainFlow SDK 从 OpenBCI 硬件或合成板读取实时 EEG 数据。
提供简洁接口供 CTNet 分类器和控制流程使用。

支持的板卡:
    - SYNTHETIC_BOARD: 无需硬件，生成合成信号用于开发测试
    - CYTON_BOARD: OpenBCI Cyton 8通道
    - CYTON_DAISY_BOARD: OpenBCI Cyton+Daisy 16通道
    - GANGLION_BOARD: OpenBCI Ganglion 4通道

用法:
    from scripts.openbci_stream import OpenBCIStream

    stream = OpenBCIStream(board_type="synthetic")
    stream.start()
    epoch = stream.get_eeg_epoch(duration_sec=4.0)  # (1, n_channels, n_samples)
    stream.stop()

创建时间: 2026-03-10
"""

from __future__ import annotations

import time
from typing import Optional, Literal

import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes


# ============================================================================
# 板卡类型映射
# ============================================================================

BOARD_TYPE_MAP = {
    "synthetic": BoardIds.SYNTHETIC_BOARD.value,
    "cyton": BoardIds.CYTON_BOARD.value,
    "cyton_daisy": BoardIds.CYTON_DAISY_BOARD.value,
    "ganglion": BoardIds.GANGLION_BOARD.value,
}


# ============================================================================
# OpenBCI 数据流
# ============================================================================

class OpenBCIStream:
    """
    BrainFlow EEG 数据流封装

    提供:
    - start() / stop(): 启动/停止数据采集
    - get_eeg_epoch(): 获取一段 EEG epoch 数据
    - get_eeg_epoch_for_ctnet(): 获取适配 CTNet 输入格式的数据

    Parameters
    ----------
    board_type : str
        板卡类型: "synthetic", "cyton", "cyton_daisy", "ganglion"
    serial_port : str or None
        串口路径 (real hardware 时必需, e.g. "/dev/ttyUSB0")
    target_channels : int
        目标通道数 (CTNet 需要 22 通道, 不足时 zero-pad)
    target_samples : int
        目标样本数 (CTNet 需要 1000 个样本点, 即 250 Hz × 4s)
    bandpass : tuple or None
        带通滤波频率范围, None 表示不滤波. 默认 (8.0, 30.0) Hz
    """

    def __init__(
        self,
        board_type: Literal["synthetic", "cyton", "cyton_daisy", "ganglion"] = "synthetic",
        serial_port: Optional[str] = None,
        target_channels: int = 22,
        target_samples: int = 1000,
        bandpass: Optional[tuple] = (8.0, 30.0),
    ):
        self.board_type = board_type
        self.serial_port = serial_port
        self.target_channels = target_channels
        self.target_samples = target_samples
        self.bandpass = bandpass

        # BrainFlow 配置
        self._params = BrainFlowInputParams()
        if serial_port is not None:
            self._params.serial_port = serial_port

        # 获取板卡 ID
        self._board_id = BOARD_TYPE_MAP.get(board_type)
        if self._board_id is None:
            raise ValueError(
                f"Unknown board type: '{board_type}'. "
                f"Supported: {list(BOARD_TYPE_MAP.keys())}"
            )

        # 创建 BoardShim
        self._board = BoardShim(self._board_id, self._params)

        # 板卡信息 (prepare_session 前也能查)
        self._eeg_channels = BoardShim.get_eeg_channels(self._board_id)
        self._sfreq = BoardShim.get_sampling_rate(self._board_id)
        self._n_eeg = len(self._eeg_channels)

        self._is_streaming = False

        print(f"[OpenBCIStream] 初始化完成")
        print(f"  板卡类型: {board_type} (ID={self._board_id})")
        print(f"  EEG 通道数: {self._n_eeg} ({self._eeg_channels})")
        print(f"  采样率: {self._sfreq} Hz")
        print(f"  目标格式: ({target_channels} ch, {target_samples} samples)")
        if bandpass:
            print(f"  带通滤波: {bandpass[0]}-{bandpass[1]} Hz")

    @property
    def sampling_rate(self) -> int:
        """采样率"""
        return self._sfreq

    @property
    def n_eeg_channels(self) -> int:
        """EEG 通道数"""
        return self._n_eeg

    @property
    def is_streaming(self) -> bool:
        """是否正在采集"""
        return self._is_streaming

    def start(self) -> None:
        """启动数据采集"""
        if self._is_streaming:
            print("[OpenBCIStream] 已在采集中")
            return

        try:
            self._board.prepare_session()
            self._board.start_stream()
            self._is_streaming = True
            print("[OpenBCIStream] ✓ 数据流已启动")
        except Exception as e:
            raise RuntimeError(f"启动数据流失败: {e}")

    def stop(self) -> None:
        """停止数据采集"""
        if not self._is_streaming:
            return

        try:
            self._board.stop_stream()
            self._board.release_session()
            self._is_streaming = False
            print("[OpenBCIStream] ✓ 数据流已停止")
        except Exception as e:
            print(f"[OpenBCIStream] 停止数据流时出错: {e}")

    def get_eeg_epoch(
        self,
        duration_sec: float = 4.0,
        apply_filter: bool = True,
    ) -> np.ndarray:
        """
        获取一段原始 EEG epoch

        从 BrainFlow 环形缓冲区中读取指定时长的数据。
        如果缓冲区数据不足,会等待直到足够。

        Parameters
        ----------
        duration_sec : float
            epoch 时长 (秒)
        apply_filter : bool
            是否应用带通滤波

        Returns
        -------
        epoch : ndarray, shape (n_eeg_channels, n_samples)
            原始 EEG 数据 (微伏)
        """
        if not self._is_streaming:
            raise RuntimeError("数据流未启动，请先调用 start()")

        needed_samples = int(duration_sec * self._sfreq)

        # 等待足够的数据
        wait_time = duration_sec + 0.5  # 多等 0.5s 确保数据足够
        time.sleep(wait_time)

        # 读取数据
        data = self._board.get_current_board_data(needed_samples)

        if data.shape[1] < needed_samples:
            print(f"[OpenBCIStream] 警告: 数据不足 ({data.shape[1]}/{needed_samples}), "
                  f"尝试等待更多数据...")
            time.sleep(1.0)
            data = self._board.get_current_board_data(needed_samples)

        # 提取 EEG 通道
        eeg_data = data[self._eeg_channels, :]

        # 裁剪到目标长度
        if eeg_data.shape[1] > needed_samples:
            eeg_data = eeg_data[:, -needed_samples:]
        elif eeg_data.shape[1] < needed_samples:
            # 零填充
            pad_width = needed_samples - eeg_data.shape[1]
            eeg_data = np.pad(eeg_data, ((0, 0), (0, pad_width)), mode='constant')

        # 带通滤波
        if apply_filter and self.bandpass is not None:
            lo, hi = self.bandpass
            for ch_idx in range(eeg_data.shape[0]):
                DataFilter.perform_bandpass(
                    eeg_data[ch_idx],
                    self._sfreq,
                    lo + (hi - lo) / 2,  # center frequency
                    hi - lo,              # bandwidth
                    order=4,
                    filter_type=FilterTypes.BUTTERWORTH.value,
                    ripple=0,
                )

        return eeg_data

    def get_eeg_epoch_for_ctnet(
        self,
        duration_sec: float = 4.0,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        获取适配 CTNet 输入格式的 EEG epoch

        CTNet (BCI IV-2a, dataset A) 期望输入:
            shape = (batch, 1, 22, 1000)
            - 22 EEG 通道
            - 1000 样本点 (250 Hz × 4s)

        如果硬件通道数 < 22, 用零填充到 22 通道
        如果采样率不同, 重采样到 250 Hz

        Parameters
        ----------
        duration_sec : float
            epoch 时长 (秒), 默认 4.0s
        normalize : bool
            是否 z-score 归一化

        Returns
        -------
        epoch : ndarray, shape (1, 1, target_channels, target_samples)
            CTNet 兼容的输入张量
        """
        # 获取原始 EEG 数据
        raw_epoch = self.get_eeg_epoch(
            duration_sec=duration_sec,
            apply_filter=True,
        )

        n_ch, n_samp = raw_epoch.shape

        # 重采样到目标采样率 (如果需要)
        target_sfreq = self.target_samples / duration_sec  # 1000 / 4.0 = 250 Hz
        if abs(self._sfreq - target_sfreq) > 1:
            # 简单线性插值重采样
            x_old = np.linspace(0, 1, n_samp)
            x_new = np.linspace(0, 1, self.target_samples)
            resampled = np.zeros((n_ch, self.target_samples))
            for ch in range(n_ch):
                resampled[ch] = np.interp(x_new, x_old, raw_epoch[ch])
            raw_epoch = resampled
            n_samp = self.target_samples

        # 裁剪/填充到目标样本数
        if n_samp > self.target_samples:
            raw_epoch = raw_epoch[:, :self.target_samples]
        elif n_samp < self.target_samples:
            pad = self.target_samples - n_samp
            raw_epoch = np.pad(raw_epoch, ((0, 0), (0, pad)), mode='constant')

        # 通道填充到目标通道数
        if n_ch < self.target_channels:
            pad_ch = self.target_channels - n_ch
            raw_epoch = np.pad(raw_epoch, ((0, pad_ch), (0, 0)), mode='constant')
        elif n_ch > self.target_channels:
            raw_epoch = raw_epoch[:self.target_channels, :]

        # Z-score 归一化 (与 CTNet 训练数据处理一致)
        if normalize:
            mean = np.mean(raw_epoch)
            std = np.std(raw_epoch)
            if std > 1e-8:
                raw_epoch = (raw_epoch - mean) / std

        # 扩展维度: (1, 1, channels, samples)
        epoch = raw_epoch[np.newaxis, np.newaxis, :, :]

        return epoch.astype(np.float32)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# ============================================================================
# 快速测试
# ============================================================================

def _test_synthetic():
    """快速测试合成板卡"""
    print("\n" + "=" * 60)
    print("OpenBCI Stream 快速测试 (SYNTHETIC_BOARD)")
    print("=" * 60)

    with OpenBCIStream(board_type="synthetic") as stream:
        print(f"\n采样率: {stream.sampling_rate} Hz")
        print(f"EEG 通道数: {stream.n_eeg_channels}")

        # 获取原始 epoch
        print("\n--- 获取原始 EEG epoch (2s) ---")
        raw = stream.get_eeg_epoch(duration_sec=2.0)
        print(f"  原始 epoch shape: {raw.shape}")
        print(f"  数据范围: [{raw.min():.4f}, {raw.max():.4f}]")

        # 获取 CTNet 格式 epoch
        print("\n--- 获取 CTNet 格式 epoch (4s) ---")
        ctnet_epoch = stream.get_eeg_epoch_for_ctnet(duration_sec=4.0)
        print(f"  CTNet epoch shape: {ctnet_epoch.shape}")
        print(f"  期望 shape: (1, 1, 22, 1000)")
        assert ctnet_epoch.shape == (1, 1, 22, 1000), \
            f"Shape mismatch! Got {ctnet_epoch.shape}"
        print(f"  数据范围: [{ctnet_epoch.min():.4f}, {ctnet_epoch.max():.4f}]")

    print("\n✅ 测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    _test_synthetic()
