#!/usr/bin/env python3
"""
测试平滑控制参数

用法:
    # 测试不同平滑度 (含预归位和后归位)
    python scripts/test_smooth_control.py --port /dev/ttyACM1 --smoothness medium --pre-home --post-home
    
    # 手动指定参数
    python scripts/test_smooth_control.py --port /dev/ttyACM1 \
        --move-time 500 --action-delay 600 --step-rad 0.12 --pre-home --post-home

创建时间: 2026-02-17
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 项目路径
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from serial_arm_env_v2 import SerialArmEnvV2, SerialConfigV2, create_smooth_config


def parse_args():
    p = argparse.ArgumentParser(description="测试平滑控制参数")
    
    p.add_argument("--port", type=str, required=True, help="串口端口")
    
    # 预设平滑度
    p.add_argument("--smoothness", choices=["low", "medium", "high", "segment"], 
                   default="medium", 
                   help="平滑度级别: low/medium/high/segment (segment=大步长+自动回中)")
    
    # 手动参数 (覆盖预设)
    p.add_argument("--move-time", type=int, default=None, 
                   help="每步运动时间 (ms)")
    p.add_argument("--action-delay", type=int, default=None, 
                   help="动作间延时 (ms)")
    p.add_argument("--step-rad", type=float, default=None, 
                   help="每步弧度")
    p.add_argument("--velocity", type=int, default=None,
                   help="运动速度 (ticks/s), 越小越慢, 推荐: 50-200")
    p.add_argument("--soft-margin", type=float, default=None,
                   help="软限位边距 (0.0-0.3)")
    
    # 测试模式
    p.add_argument("--pattern", choices=["square", "cross", "random", "manual"],
                   default="square", help="测试图案")
    p.add_argument("--loops", type=int, default=2, help="重复次数")
    p.add_argument("--recenter", action="store_true", help="开始前软件回中 (使用环境内置)")
    p.add_argument("--auto-recenter", type=int, default=0,
                   help="自动回中间隔 (每 N 步回中, 0=禁用, segment模式默认=10)")
    
    # 归位控制 (使用外部脚本)
    p.add_argument("--pre-home", action="store_true", help="测试前回到中位")
    p.add_argument("--post-home", action="store_true", help="测试后回到归位")
    p.add_argument("--home-json", type=Path, default=_ROOT / "serial_home.json",
                   help="中位配置文件")
    p.add_argument("--return-json", type=Path, default=_ROOT / "serial_return.json",
                   help="归位配置文件")
    
    return p.parse_args()


def run_pattern(env: SerialArmEnvV2, pattern: str, loops: int):
    """运行测试图案"""
    
    if pattern == "square":
        # 正方形: L L U U R R D D
        actions = [0, 0, 2, 2, 1, 1, 3, 3] * loops
    elif pattern == "cross":
        # 十字: L R U D
        actions = [0, 1, 2, 3] * loops * 2
    elif pattern == "random":
        import random
        actions = [random.randint(0, 3) for _ in range(16 * loops)]
    elif pattern == "manual":
        print("手动模式: 输入 L/R/U/D 控制, Q 退出")
        actions = []  # 会在下面处理
    else:
        actions = []
    
    action_names = {0: "left", 1: "right", 2: "up", 3: "down"}
    recenter_interval = env.cfg.auto_recenter_interval
    
    print(f"\n{'='*60}")
    print(f"开始测试: {pattern} 图案, {loops} 次循环")
    if recenter_interval > 0:
        n_segments = (len(actions) + recenter_interval - 1) // recenter_interval
        print(f"分段模式: 每 {recenter_interval} 步回中, 共 {n_segments} 段")
    print(f"{'='*60}")
    
    if pattern == "manual":
        # 手动模式
        step = 0
        while True:
            status = env.get_status()
            print(f"\nStep {step} | LR: {status['lr_deg']:.1f}° | UD: {status['ud_deg']:.1f}°")
            if status.get('lr_near_limit'):
                print("  ⚠️ LR 接近限位!")
            if status.get('ud_near_limit'):
                print("  ⚠️ UD 接近限位!")
            
            cmd = input("输入 (L/R/U/D/C=回中/Q=退出): ").strip().upper()
            
            if cmd == "Q":
                break
            elif cmd == "L":
                env.step(0)
            elif cmd == "R":
                env.step(1)
            elif cmd == "U":
                env.step(2)
            elif cmd == "D":
                env.step(3)
            elif cmd == "C":
                env._recenter_joints()
            else:
                print("无效输入")
                continue
            
            step += 1
    else:
        # 自动模式
        segment = 1
        for i, action in enumerate(actions):
            # 分段提示
            if recenter_interval > 0 and i > 0 and i % recenter_interval == 0:
                segment += 1
                print(f"\n{'─'*60}")
                print(f"📍 段 {segment}: 已回中, 继续执行...")
                print(f"{'─'*60}")
            
            status = env.get_status()
            seg_step = (i % recenter_interval) + 1 if recenter_interval > 0 else i + 1
            seg_info = f"[段{segment} 步{seg_step:2d}]" if recenter_interval > 0 else f"[{i+1:3d}/{len(actions)}]"
            
            print(f"{seg_info} {action_names[action]:5s} | "
                  f"LR: {status['lr_deg']:6.1f}° | UD: {status['ud_deg']:6.1f}° | "
                  f"Limit: {'⚠️' if status.get('consecutive_limit_hits', 0) > 0 else '✓'}")
            
            obs, reward, done, truncated, info = env.step(action)
            
            if info.get("msg_lr") and "限位" in info.get("msg_lr", ""):
                print(f"       {info['msg_lr']}")
            if info.get("msg_ud") and "限位" in info.get("msg_ud", ""):
                print(f"       {info['msg_ud']}")


def go_home(port: str, home_json: Path):
    """调用外部脚本回到中位"""
    if not home_json.exists():
        print(f"[警告] 中位配置文件不存在: {home_json}")
        return
    
    print(f"[Pre-home] 回到中位...")
    cmd = f"python {_HERE / 'serial_go_home.py'} --port {port} --home-json {home_json}"
    os.system(cmd)
    time.sleep(1.0)


def go_return(port: str, return_json: Path):
    """调用外部脚本回到归位"""
    if not return_json.exists():
        print(f"[警告] 归位配置文件不存在: {return_json}")
        return
    
    print(f"\n[Post-home] 回到归位...")
    cmd = f"python {_HERE / 'serial_go_return.py'} --port {port} --return-json {return_json}"
    os.system(cmd)


def main():
    args = parse_args()
    
    # ==================== 预归位 ====================
    if args.pre_home:
        go_home(args.port, args.home_json)
    
    # 创建配置
    cfg = create_smooth_config(args.port, args.smoothness, auto_recenter=args.auto_recenter)
    
    # 覆盖手动参数
    if args.move_time is not None:
        cfg.move_time_ms = args.move_time
    if args.action_delay is not None:
        cfg.action_delay_ms = args.action_delay
    if args.step_rad is not None:
        cfg.joint_step_rad = args.step_rad
    if args.velocity is not None:
        cfg.move_velocity = args.velocity
    if args.soft_margin is not None:
        cfg.soft_limit_margin = args.soft_margin
    
    # 打印配置
    print("="*60)
    print("平滑控制参数测试")
    print("="*60)
    print(f"  串口: {cfg.port}")
    print(f"  平滑度: {args.smoothness}")
    print(f"  步长: {cfg.joint_step_rad:.3f} rad ({cfg.joint_step_rad * 180/3.14159:.1f}°)")
    print(f"  运动时间: {cfg.move_time_ms} ms")
    if cfg.move_velocity is not None:
        print(f"  🐢 运动速度: {cfg.move_velocity} ticks/s (越小越慢)")
    else:
        print(f"  运动速度: 默认")
    print(f"  动作延时: {cfg.action_delay_ms} ms")
    print(f"  软限位边距: {cfg.soft_limit_margin*100:.0f}%")
    if cfg.auto_recenter_interval > 0:
        print(f"  🔄 自动回中: 每 {cfg.auto_recenter_interval} 步")
    else:
        print(f"  自动回中: 禁用")
    print("="*60)
    
    # 创建环境
    env = None
    try:
        env = SerialArmEnvV2(cfg, render_mode="human")
        
        # ★★★ 重要：环境创建后先执行一次回中 ★★★
        # 这确保两个控制关节都在中位开始
        print("\n[测试前] 确保两个控制关节回中...")
        env._recenter_joints()
        
        # 重置计数器
        env.reset()
        
        # 运行测试
        run_pattern(env, args.pattern, args.loops)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except:
                pass
    
    # ==================== 后归位 ====================
    if args.post_home:
        go_return(args.port, args.return_json)
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()

