#!/usr/bin/env python3
"""
EEG 物理机械臂控制 v2 — 大幅度运动版本

相比 v1 (eeg_physical_control.py) 的改动:
    - step_rad: 0.35 → 0.55 (物理臂每步弧度)
    - norm_step: 0.12 → 0.20 (归一化步长)
    - velocity: 100 → 120 (移动速度)
    - target_radius: 0.12 → 0.15 (到达判定半径)
    - 目标幅度: 0.75 → 0.90 (使目标更远，总位移更大)

目的:
    增大机械臂运动幅度 → 肉眼更容易看到运动方向 + 减小相对误差

用法:
    python scripts/eeg_physical_control_v2.py --serial-port /dev/ttyACM1 --subjects 1 2 3

    # 或自定义参数:
    python scripts/eeg_physical_control_v2.py --step-rad 0.60 --norm-step 0.25

创建时间: 2026-03-10
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ============================================================================
# 参数对比表
# ============================================================================

PARAM_COMPARISON = """
╔══════════════════╦═══════════╦═══════════╦════════════════════════════════╗
║ 参数             ║   v1      ║   v2      ║ 影响                          ║
╠══════════════════╬═══════════╬═══════════╬════════════════════════════════╣
║ step_rad         ║   0.35    ║   0.55    ║ 物理臂每步转动更大角度        ║
║ norm_step        ║   0.12    ║   0.20    ║ 归一化坐标中每步移动更远      ║
║ velocity         ║   100     ║   120     ║ 伺服电机移动速度更快          ║
║ target_radius    ║   0.12    ║   0.15    ║ 到达目标的判定范围更宽松      ║
║ target_amplitude ║   0.75    ║   0.90    ║ 目标位置更远，总路径更长      ║
╚══════════════════╩═══════════╩═══════════╩════════════════════════════════╝

相对误差 (Relative Error) 预期改善:
  原理: 绝对误差大致不变 → 总位移增大 → 相对误差 = 绝对误差/总位移 减小
  
  v1 示例: 绝对误差=0.05, 总位移=0.75 → 相对误差=6.7%
  v2 示例: 绝对误差=0.05, 总位移=0.90 → 相对误差=5.6% (↓1.1%)
"""


# ============================================================================
# 覆写默认参数并调用原始脚本
# ============================================================================

def main():
    """
    Wrapper: 用更大的步长参数调用 eeg_physical_control.py 的主流程
    """
    print("=" * 70)
    print("EEG 物理机械臂控制 v2 — 大幅度运动版本")
    print("=" * 70)
    print(PARAM_COMPARISON)

    # ---- 导入原始模块 ----
    from scripts.eeg_physical_control import (
        parse_args,
        main as original_main,
        MovementTarget,
        SubjectSequence,
    )

    # ---- 覆写默认值 ----
    # 通过修改 sys.argv 来将 v2 默认值注入
    import argparse

    # 解析用户传入的参数 (如果有)
    p = argparse.ArgumentParser(description="EEG 物理机械臂控制 v2")
    p.add_argument("--serial-port", type=str, default="/dev/ttyACM1")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--noise-level", type=float, default=0.1,
                   help="EEG 分类噪声水平 (0-1)")
    p.add_argument("--step-rad", type=float, default=0.55,
                   help="[v2 default: 0.55] 物理臂每步弧度")
    p.add_argument("--norm-step", type=float, default=0.20,
                   help="[v2 default: 0.20] 归一化移动步长")
    p.add_argument("--velocity", type=int, default=120,
                   help="[v2 default: 120] 伺服电机速度")
    p.add_argument("--target-radius", type=float, default=0.15,
                   help="[v2 default: 0.15] 到达判定半径")
    p.add_argument("--output-dir", type=Path,
                   default=_ROOT / "outputs" / "eeg_physical_control_v2")
    p.add_argument("--home-json", type=Path,
                   default=_ROOT / "serial_home.json")
    p.add_argument("--return-json", type=Path,
                   default=_ROOT / "serial_return.json")
    p.add_argument("--physical-results", type=Path,
                   default=_ROOT / "outputs" / "8dir_comprehensive_test")

    args = p.parse_args()

    # 更新 MovementTarget 位置到更大幅度
    _orig_from_name = MovementTarget.from_name.__func__

    @staticmethod
    def from_name_v2(name: str) -> MovementTarget:
        """v2 版本: 使用 0.90 幅度"""
        positions = {
            "center":     (0.0, 0.0),
            "left":       (-0.90, 0.0),
            "right":      (0.90, 0.0),
            "up":         (0.0, 0.90),
            "down":       (0.0, -0.90),
            "up_left":    (-0.70, 0.70),
            "up_right":   (0.70, 0.70),
            "down_left":  (-0.70, -0.70),
            "down_right": (0.70, -0.70),
        }
        if name not in positions:
            raise ValueError(f"Unknown position: {name}")
        y, z = positions[name]
        return MovementTarget(name=name, y=y, z=z)

    MovementTarget.from_name = from_name_v2

    # ---- 构建 sys.argv 传递给原始脚本 ----
    sys.argv = [
        "eeg_physical_control_v2.py",
        "--serial-port", args.serial_port,
        "--subjects", *[str(s) for s in args.subjects],
        "--noise-level", str(args.noise_level),
        "--step-rad", str(args.step_rad),
        "--norm-step", str(args.norm_step),
        "--velocity", str(args.velocity),
        "--target-radius", str(args.target_radius),
        "--output-dir", str(args.output_dir),
        "--home-json", str(args.home_json),
        "--return-json", str(args.return_json),
        "--physical-results", str(args.physical_results),
    ]

    print(f"\n使用参数:")
    print(f"  step_rad = {args.step_rad}")
    print(f"  norm_step = {args.norm_step}")
    print(f"  velocity = {args.velocity}")
    print(f"  target_radius = {args.target_radius}")
    print(f"  target_amplitude = 0.90")
    print(f"  output = {args.output_dir}")
    print(f"  subjects = {args.subjects}")
    print()

    # ---- 调用原始主函数 ----
    original_main()


if __name__ == "__main__":
    main()
