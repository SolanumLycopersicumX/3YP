"""
优化版串口机械臂环境 (V2)

改进内容:
1. 平滑运动: 增大 move_time_ms, 加入运动插值
2. 限位保护: 软限位检测 + 自动回中 + 限位感知
3. 步长优化: 增大 joint_step_rad 使运动更明显

创建时间: 2026-02-17
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import numpy as np

from drivers.so101_serial import So101Bus, So101Map, Limits


@dataclass
class SerialConfigV2:
    """优化版串口配置"""
    port: str
    baud: int = 1_000_000
    timeout: float = 0.02
    
    # 动作映射
    joint_lr_name: str = "shoulder_pan"
    joint_ud_name: str = "wrist_flex"
    
    # ========== 优化参数 ==========
    
    # 步长 (弧度) - 增大使运动更明显
    joint_step_rad: float = 0.12  # ~7度, 原先 0.05 (~3度)
    
    # 平滑运动参数
    move_time_ms: int = 500       # 每步运动时间 (传给伺服)
    action_delay_ms: int = 600    # 动作间延时
    move_velocity: Optional[int] = None  # 运动速度 (ticks/s), None=使用默认, 越小越慢
    
    # 限位保护参数
    soft_limit_margin: float = 0.10  # 软限位边距 (占总行程的百分比)
    auto_recenter_interval: int = 0  # 自动回中间隔 (0=禁用, >0=每N步回中)
    limit_warning_enabled: bool = True  # 限位预警
    
    # 方向反转
    invert_lr: bool = False
    invert_ud: bool = False
    
    # 运行参数
    max_steps: int = 50
    
    # 标定映射
    mapping: So101Map = field(default_factory=So101Map.default)


class SerialArmEnvV2:
    """
    优化版串口机械臂环境
    
    改进:
    1. 平滑运动 - 增大运动时间 + 插值
    2. 限位保护 - 软限位 + 自动回中
    3. 状态反馈 - 返回关节位置用于感知
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(self, cfg: SerialConfigV2, render_mode: Optional[str] = None):
        self.cfg = cfg
        self.render_mode = render_mode
        self._bus = So101Bus(cfg.port, cfg.baud, timeout=cfg.timeout)
        self._bus.open()
        
        # 关节 ID
        self._id_lr = cfg.mapping.name_to_id.get(cfg.joint_lr_name)
        self._id_ud = cfg.mapping.name_to_id.get(cfg.joint_ud_name)
        
        print(f"[SerialArmEnvV2] 初始化...")
        print(f"  LR 关节: {cfg.joint_lr_name} -> ID {self._id_lr}")
        print(f"  UD 关节: {cfg.joint_ud_name} -> ID {self._id_ud}")
        
        # 关节限位 (ticks)
        self._joint_limits: Dict[int, Limits] = {}
        self._joint_centers: Dict[int, int] = {}  # 中心位置
        self._joint_ranges: Dict[int, int] = {}   # 行程范围
        
        # 初始化关节
        joint_names = {self._id_lr: "LR (shoulder_pan)", self._id_ud: "UD (wrist_flex)"}
        for jid in (self._id_lr, self._id_ud):
            if jid is None:
                print(f"[SerialArmEnvV2] 警告: 关节 ID 为 None")
                continue
            try:
                self._bus.set_operating_mode(jid, 0)
                self._bus.set_return_delay(jid, 0)
                self._bus.torque_enable(jid, True)
                
                # ★★★ 关键：初始化时一次性设置速度和时间 ★★★
                # 这样之后只需写目标位置，运动就会匀速平滑
                self._set_joint_speed(jid)
                
                # 读取限位
                try:
                    bmin = self._bus.read(jid, self._bus.MIN_POSITION_LIMIT, 2)
                    bmax = self._bus.read(jid, self._bus.MAX_POSITION_LIMIT, 2)
                    mn = int(bmin[0]) | (int(bmin[1]) << 8)
                    mx = int(bmax[0]) | (int(bmax[1]) << 8)
                    if mn > mx:
                        mn, mx = mx, mn
                    
                    self._joint_limits[jid] = Limits(min_ticks=mn, max_ticks=mx)
                    self._joint_centers[jid] = (mn + mx) // 2
                    self._joint_ranges[jid] = mx - mn
                    print(f"[SerialArmEnvV2] {joint_names.get(jid, jid)}: "
                          f"限位=[{mn}, {mx}], 中心={self._joint_centers[jid]}")
                except Exception as e:
                    print(f"[SerialArmEnvV2] 读取关节 {jid} 限位失败: {e}")
            except Exception as e:
                print(f"[SerialArmEnvV2] 初始化关节 {jid} 失败: {e}")
        
        self._step_count = 0
        self._last_action = -1
        self._consecutive_limit_hits = 0
    
    def _set_joint_speed(self, jid: int, velocity: Optional[int] = None, time_ms: Optional[int] = None):
        """
        设置关节的运动速度和时间
        
        Args:
            jid: 关节 ID
            velocity: 速度 (ticks/s), None=使用配置值
            time_ms: 运动时间 (ms), None=使用配置值
        """
        try:
            # 使用传入值或配置值
            t = time_ms if time_ms is not None else self.cfg.move_time_ms
            v = velocity if velocity is not None else self.cfg.move_velocity
            
            # 设置运动时间
            if t is not None:
                t = max(0, min(0xFFFF, int(t)))
                self._bus.write(jid, self._bus.GOAL_TIME, bytes([t & 0xFF, (t >> 8) & 0xFF]))
            
            # 设置运动速度
            if v is not None:
                v = max(0, min(0xFFFF, int(v)))
                self._bus.write(jid, self._bus.GOAL_VELOCITY, bytes([v & 0xFF, (v >> 8) & 0xFF]))
        except Exception as e:
            print(f"[SerialArmEnvV2] 设置关节 {jid} 速度失败: {e}")
    
    def _set_fast_mode(self):
        """切换到快速模式（用于回中/归位）"""
        joint_names = {self._id_lr: "LR", self._id_ud: "UD"}
        for jid in (self._id_lr, self._id_ud):
            if jid is not None:
                print(f"  [快速模式] {joint_names.get(jid, jid)} (ID={jid}): velocity=500, time_ms=300")
                self._set_joint_speed(jid, velocity=500, time_ms=300)
            else:
                print(f"  [警告] 关节 ID 为 None, 跳过")
    
    def _set_slow_mode(self):
        """切换到慢速模式（用于正常控制）"""
        joint_names = {self._id_lr: "LR", self._id_ud: "UD"}
        for jid in (self._id_lr, self._id_ud):
            if jid is not None:
                v = self.cfg.move_velocity
                t = self.cfg.move_time_ms
                print(f"  [慢速模式] {joint_names.get(jid, jid)} (ID={jid}): velocity={v}, time_ms={t}")
                self._set_joint_speed(jid)
            else:
                print(f"  [警告] 关节 ID 为 None, 跳过")
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境"""
        self._step_count = 0
        self._last_action = -1
        self._consecutive_limit_hits = 0
        
        # 可选: 重置时回中
        if options and options.get("recenter", False):
            self._recenter_joints()
        
        obs = self._get_obs()
        return obs, {}
    
    def close(self):
        """关闭环境"""
        try:
            # 关闭前切换回快速模式，这样后续归位脚本能快速运行
            print("[SerialArmEnvV2] 关闭前切换到快速模式...")
            self._set_fast_mode()
            
            for jid in (self._id_lr, self._id_ud):
                if jid is not None:
                    self._bus.torque_enable(jid, False)
        except Exception:
            pass
        self._bus.close()
    
    def _read_ticks(self, jid: Optional[int]) -> Optional[int]:
        """读取关节当前位置"""
        if jid is None:
            return None
        try:
            return int(self._bus.read_position(jid))
        except Exception:
            return None
    
    def _get_soft_limits(self, jid: int) -> Tuple[int, int]:
        """
        获取软限位 (留有边距)
        
        Returns:
            (soft_min, soft_max): 软限位范围
        """
        lim = self._joint_limits.get(jid)
        if lim is None or lim.min_ticks is None or lim.max_ticks is None:
            return (0, 4095)  # 默认全范围
        
        margin = int(self._joint_ranges.get(jid, 0) * self.cfg.soft_limit_margin)
        soft_min = lim.min_ticks + margin
        soft_max = lim.max_ticks - margin
        return (soft_min, soft_max)
    
    def _is_near_limit(self, jid: int, ticks: int) -> Tuple[bool, str]:
        """
        检查是否接近限位
        
        Returns:
            (is_near, direction): 是否接近限位, 哪个方向
        """
        soft_min, soft_max = self._get_soft_limits(jid)
        if ticks <= soft_min:
            return True, "min"
        if ticks >= soft_max:
            return True, "max"
        return False, ""
    
    def _would_hit_limit(self, jid: int, d_rad: float) -> bool:
        """
        预判动作是否会碰限位
        """
        cur = self._read_ticks(jid)
        if cur is None:
            return False
        
        d_ticks = So101Bus.rad_to_ticks(d_rad)
        target = cur + d_ticks
        soft_min, soft_max = self._get_soft_limits(jid)
        
        return target < soft_min or target > soft_max
    
    def _recenter_joints(self):
        """回到中心位置（快速模式）"""
        print("[SerialArmEnvV2] 回中 (快速)...")
        
        # 切换到快速模式
        self._set_fast_mode()
        
        joint_names = {self._id_lr: "LR", self._id_ud: "UD"}
        moved = 0
        
        for jid in (self._id_lr, self._id_ud):
            if jid is None:
                print(f"  [警告] 关节 ID 为 None, 跳过")
                continue
            center = self._joint_centers.get(jid)
            if center is not None:
                cur = self._read_ticks(jid)
                print(f"  {joint_names.get(jid, jid)}: {cur} -> {center}")
                self._smooth_move(jid, center)
                moved += 1
            else:
                print(f"  [警告] {joint_names.get(jid, jid)} 没有中心值, 跳过")
        
        if moved == 0:
            print("  [警告] 没有关节被移动!")
        
        # 等待快速运动完成
        time.sleep(1.5)
        
        # 切换回慢速模式
        self._set_slow_mode()
        print("[SerialArmEnvV2] 已切换回慢速模式")
    
    def _smooth_move(self, jid: int, target_ticks: int):
        """
        平滑移动到目标位置
        
        ★ 关键改进：只写目标位置，不重复设置速度 ★
        速度已在初始化时设置，这样运动会匀速连续
        """
        # 应用硬限位
        lim = self._joint_limits.get(jid)
        if lim is not None:
            target_ticks = lim.clamp(target_ticks)
        
        try:
            # 只写目标位置，不设置 time_ms 和 velocity
            # 这样伺服会使用初始化时设置的速度匀速运动
            self._bus.write_position(jid, int(target_ticks))
        except Exception as e:
            print(f"[SerialArmEnvV2] 移动失败: {e}")
    
    def _apply_action(self, jid: Optional[int], d_rad: float) -> Tuple[bool, str]:
        """
        应用动作到关节
        
        Returns:
            (success, message): 是否成功, 消息
        """
        if jid is None:
            return False, "关节不存在"
        
        cur = self._read_ticks(jid)
        if cur is None:
            return False, "读取位置失败"
        
        d_ticks = So101Bus.rad_to_ticks(d_rad)
        target = cur + d_ticks
        
        # 检查软限位
        soft_min, soft_max = self._get_soft_limits(jid)
        
        # 如果会超出软限位, 限制到边缘
        limited = False
        if target < soft_min:
            target = soft_min
            limited = True
        elif target > soft_max:
            target = soft_max
            limited = True
        
        # 执行平滑移动
        self._smooth_move(jid, target)
        
        if limited:
            self._consecutive_limit_hits += 1
            if self.cfg.limit_warning_enabled:
                return True, f"⚠️ 限位保护 (连续{self._consecutive_limit_hits}次)"
        else:
            self._consecutive_limit_hits = 0
        
        return True, "OK"
    
    def step(self, action: int):
        """
        执行一步动作
        
        Args:
            action: 0=left, 1=right, 2=up, 3=down
        """
        # 计算方向
        d_lr = 0.0
        d_ud = 0.0
        action_name = ""
        
        if action == 0:  # left
            d_lr = self.cfg.joint_step_rad * (-1.0 if self.cfg.invert_lr else 1.0)
            action_name = "left"
        elif action == 1:  # right
            d_lr = -self.cfg.joint_step_rad * (-1.0 if self.cfg.invert_lr else 1.0)
            action_name = "right"
        elif action == 2:  # up
            d_ud = self.cfg.joint_step_rad * (-1.0 if self.cfg.invert_ud else 1.0)
            action_name = "up"
        elif action == 3:  # down
            d_ud = -self.cfg.joint_step_rad * (-1.0 if self.cfg.invert_ud else 1.0)
            action_name = "down"
        
        # 应用动作
        msg_lr = ""
        msg_ud = ""
        
        if self._id_lr is not None and abs(d_lr) > 0:
            _, msg_lr = self._apply_action(self._id_lr, d_lr)
        if self._id_ud is not None and abs(d_ud) > 0:
            _, msg_ud = self._apply_action(self._id_ud, d_ud)
        
        self._step_count += 1
        self._last_action = action
        
        # 自动回中检查
        if (self.cfg.auto_recenter_interval > 0 and 
            self._step_count > 0 and 
            self._step_count % self.cfg.auto_recenter_interval == 0):
            print(f"[Step {self._step_count}] 自动回中...")
            self._recenter_joints()
        
        # 动作间延时 (核心: 让运动完成, 减少抖动)
        delay_sec = self.cfg.action_delay_ms / 1000.0
        time.sleep(delay_sec)
        
        # 观测
        obs = self._get_obs()
        terminated = self._step_count >= self.cfg.max_steps
        truncated = False
        reward = 0.0
        
        info = {
            "action": action_name,
            "step": self._step_count,
            "msg_lr": msg_lr,
            "msg_ud": msg_ud,
            "limit_hits": self._consecutive_limit_hits,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """
        获取观测
        
        返回: [lr_normalized, ud_normalized] in [-1, 1]
        """
        lr_norm = 0.0
        ud_norm = 0.0
        
        # LR 归一化
        lr_ticks = self._read_ticks(self._id_lr)
        if lr_ticks is not None and self._id_lr in self._joint_limits:
            lim = self._joint_limits[self._id_lr]
            if lim.min_ticks is not None and lim.max_ticks is not None:
                center = self._joint_centers.get(self._id_lr, (lim.min_ticks + lim.max_ticks) // 2)
                half_range = (lim.max_ticks - lim.min_ticks) / 2
                if half_range > 0:
                    lr_norm = (lr_ticks - center) / half_range
        
        # UD 归一化
        ud_ticks = self._read_ticks(self._id_ud)
        if ud_ticks is not None and self._id_ud in self._joint_limits:
            lim = self._joint_limits[self._id_ud]
            if lim.min_ticks is not None and lim.max_ticks is not None:
                center = self._joint_centers.get(self._id_ud, (lim.min_ticks + lim.max_ticks) // 2)
                half_range = (lim.max_ticks - lim.min_ticks) / 2
                if half_range > 0:
                    ud_norm = (ud_ticks - center) / half_range
        
        return np.array([lr_norm, ud_norm], dtype=np.float32)
    
    def get_status(self) -> Dict:
        """获取详细状态"""
        lr = self._read_ticks(self._id_lr)
        ud = self._read_ticks(self._id_ud)
        
        def fmt(x):
            if x is None:
                return (np.nan, np.nan)
            d = So101Bus.ticks_to_deg(x)
            return (float(x), float(d))
        
        lr_t, lr_d = fmt(lr)
        ud_t, ud_d = fmt(ud)
        
        # 限位状态
        lr_near_limit = False
        ud_near_limit = False
        if lr is not None and self._id_lr is not None:
            lr_near_limit, _ = self._is_near_limit(self._id_lr, lr)
        if ud is not None and self._id_ud is not None:
            ud_near_limit, _ = self._is_near_limit(self._id_ud, ud)
        
        return {
            "step": int(self._step_count),
            "max_steps": int(self.cfg.max_steps),
            "lr_ticks": lr_t,
            "lr_deg": lr_d,
            "lr_near_limit": lr_near_limit,
            "ud_ticks": ud_t,
            "ud_deg": ud_d,
            "ud_near_limit": ud_near_limit,
            "consecutive_limit_hits": self._consecutive_limit_hits,
        }
    
    def render(self):
        return None


# ============================================================================
# 便捷函数
# ============================================================================

def create_smooth_config(
    port: str,
    smoothness: str = "medium",  # "low", "medium", "high", "segment"
    auto_recenter: int = 0,  # 自动回中间隔 (0=禁用)
) -> SerialConfigV2:
    """
    创建不同平滑度的配置
    
    Args:
        port: 串口路径
        smoothness: 平滑度级别
            - "low": 快速但抖动较大
            - "medium": 平衡
            - "high": 慢速但非常平滑
            - "segment": 分段模式 - 大步长 + 自动回中
        auto_recenter: 自动回中间隔 (每 N 步回中一次, 0=禁用)
    """
    if smoothness == "low":
        return SerialConfigV2(
            port=port,
            joint_step_rad=0.10,
            move_time_ms=300,
            action_delay_ms=400,
            auto_recenter_interval=auto_recenter,
        )
    elif smoothness == "high":
        return SerialConfigV2(
            port=port,
            joint_step_rad=0.15,
            move_time_ms=800,
            action_delay_ms=1000,
            soft_limit_margin=0.15,
            auto_recenter_interval=auto_recenter,
        )
    elif smoothness == "segment":
        # 分段模式: 大步长 + 每 10 步回中 + 慢速运动
        # 0.20 rad ≈ 130 ticks, 在 2 秒内完成 → 速度 ≈ 65 ticks/s
        # 设置较低速度值 (50-100) 实现真正的慢速运动
        return SerialConfigV2(
            port=port,
            joint_step_rad=0.20,       # 大步长 (~11.5°)
            move_time_ms=2000,         # 2秒运动时间
            move_velocity=80,          # 低速度 (ticks/s) - 关键！
            action_delay_ms=2500,      # 等待运动完成 + 稳定
            soft_limit_margin=0.12,    # 适当边距
            auto_recenter_interval=auto_recenter if auto_recenter > 0 else 10,  # 默认每 10 步回中
        )
    else:  # medium (default)
        return SerialConfigV2(
            port=port,
            joint_step_rad=0.12,
            move_time_ms=500,
            action_delay_ms=600,
            soft_limit_margin=0.10,
            auto_recenter_interval=auto_recenter,
        )

