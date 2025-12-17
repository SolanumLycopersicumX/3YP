import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from drivers.so101_serial import So101Bus, So101Map, Limits


@dataclass
class SerialConfig:
    port: str
    baud: int = 1_000_000
    timeout: float = 0.02
    # 动作映射（必须 joint-drive 模式）
    joint_lr_name: str = "shoulder_pan"
    joint_ud_name: str = "wrist_flex"
    joint_step_rad: float = 0.05  # 每步弧度
    invert_lr: bool = False
    invert_ud: bool = False
    # 运行参数
    max_steps: int = 50
    move_time_ms: Optional[int] = 250  # 位置模式到达时间（更平滑）
    step_time_scale: float = 2.0       # 每步移动时间缩放（>1 更慢，抖动更少）
    # 标定映射（ID与限位）
    mapping: So101Map = So101Map.default()


class SerialArmEnv:
    """极简串口机械臂环境：将上下左右映射到两个关节的增量位移。

    - 动作: 0=left, 1=right, 2=up, 3=down
    - 观测: 占位 (y,z) = (nan,nan)
    - 仅支持 joint-drive 风格
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, cfg: SerialConfig, render_mode: Optional[str] = None):
        self.cfg = cfg
        self.render_mode = render_mode
        self._bus = So101Bus(cfg.port, cfg.baud, timeout=cfg.timeout)
        self._bus.open()
        # 初始化两个关节
        self._id_lr = cfg.mapping.name_to_id.get(cfg.joint_lr_name)
        self._id_ud = cfg.mapping.name_to_id.get(cfg.joint_ud_name)
        self._joint_limits: dict[int, Limits] = {}
        # 进入位置模式，打开扭矩，减少返回延迟
        for jid in (self._id_lr, self._id_ud):
            if jid is None:
                continue
            try:
                self._bus.set_operating_mode(jid, 0)
                self._bus.set_return_delay(jid, 0)
                self._bus.torque_enable(jid, True)
                # 读取各关节软限位，供夹紧
                try:
                    bmin = self._bus.read(jid, self._bus.MIN_POSITION_LIMIT, 2)
                    bmax = self._bus.read(jid, self._bus.MAX_POSITION_LIMIT, 2)
                    mn = int(bmin[0]) | (int(bmin[1]) << 8)
                    mx = int(bmax[0]) | (int(bmax[1]) << 8)
                    if mn > mx:
                        mn, mx = mx, mn
                    self._joint_limits[jid] = Limits(min_ticks=mn, max_ticks=mx)
                except Exception:
                    pass
            except Exception:
                pass
        self._step_count = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._step_count = 0
        # 返回占位观测
        obs = np.array([np.nan, np.nan], dtype=np.float32)
        return obs, {}

    def close(self):
        try:
            for jid in (self._id_lr, self._id_ud):
                if jid is not None:
                    self._bus.torque_enable(jid, False)
        except Exception:
            pass
        self._bus.close()

    def _read_ticks(self, jid: Optional[int]) -> Optional[int]:
        if jid is None:
            return None
        try:
            return int(self._bus.read_position(jid))
        except Exception:
            return None

    def _apply_ticks(self, jid: Optional[int], ticks: int):
        if jid is None:
            return
        # 限位夹紧（如提供）
        name = None
        for k, v in self.cfg.mapping.name_to_id.items():
            if v == jid:
                name = k
                break
        # 优先使用配置中的限位，否则用从设备读到的限位
        lim = None
        if name is not None:
            lim = self.cfg.mapping.limits.get(name)
        if lim is None:
            lim = self._joint_limits.get(jid)
        if lim is not None:
            ticks = lim.clamp(ticks)
        # 每步的到达时间：按比例放大以减少启动/停止抖动
        eff_time = None
        if self.cfg.move_time_ms is not None:
            try:
                eff_time = max(1, int(round(float(self.cfg.move_time_ms) * float(self.cfg.step_time_scale))))
            except Exception:
                eff_time = self.cfg.move_time_ms
        try:
            self._bus.write_position(jid, int(ticks), time_ms=eff_time)
        except Exception:
            pass

    def step(self, action: int):
        d_lr = 0.0
        d_ud = 0.0
        if action == 0:      # left
            d_lr = self.cfg.joint_step_rad * (-1.0 if self.cfg.invert_lr else 1.0)
        elif action == 1:    # right
            d_lr = -self.cfg.joint_step_rad * (-1.0 if self.cfg.invert_lr else 1.0)
        elif action == 2:    # up
            d_ud = self.cfg.joint_step_rad * (-1.0 if self.cfg.invert_ud else 1.0)
        elif action == 3:    # down
            d_ud = -self.cfg.joint_step_rad * (-1.0 if self.cfg.invert_ud else 1.0)

        # 读当前值，计算目标 ticks
        if self._id_lr is not None and abs(d_lr) > 0:
            cur = self._read_ticks(self._id_lr)
            if cur is not None:
                dticks = So101Bus.rad_to_ticks(d_lr)
                self._apply_ticks(self._id_lr, cur + dticks)
        if self._id_ud is not None and abs(d_ud) > 0:
            cur = self._read_ticks(self._id_ud)
            if cur is not None:
                dticks = So101Bus.rad_to_ticks(d_ud)
                self._apply_ticks(self._id_ud, cur + dticks)

        self._step_count += 1
        # 观测占位
        obs = np.array([np.nan, np.nan], dtype=np.float32)
        terminated = self._step_count >= self.cfg.max_steps
        truncated = False
        reward = 0.0
        info = {}
        # 轻微节流让动作更可见（上层也可 --sleep）
        if self.render_mode == "human":
            time.sleep(0.01)
        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def get_status(self):
        # 尽量读两个关节的当前位置（ticks 与 deg）
        lr = self._read_ticks(self._id_lr)
        ud = self._read_ticks(self._id_ud)
        def fmt(x):
            if x is None:
                return (np.nan, np.nan)
            d = So101Bus.ticks_to_deg(x)
            return (float(x), float(d))
        lr_t, lr_d = fmt(lr)
        ud_t, ud_d = fmt(ud)
        return {
            "ee_xyz": (np.nan, np.nan, np.nan),
            "target_yz": (np.nan, np.nan),
            "step": int(self._step_count),
            "max_steps": int(self.cfg.max_steps),
            "lr_ticks": lr_t,
            "lr_deg": lr_d,
            "ud_ticks": ud_t,
            "ud_deg": ud_d,
        }
