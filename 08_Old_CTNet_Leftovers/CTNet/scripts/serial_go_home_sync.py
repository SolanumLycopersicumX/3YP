#!/usr/bin/env python3
"""
同步回中位 - 所有关节在相同时间内到达

原理：
- 计算每个关节需要移动的距离
- 根据距离计算每个关节的速度
- 使所有关节在指定时间内同时到达目标

用法：
    python scripts/serial_go_home_sync.py --port /dev/ttyACM1 --duration 2.0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drivers.so101_serial import So101Bus, So101Map


def load_home_positions(path: Path) -> dict:
    """加载中位位置"""
    with open(path, 'r') as f:
        data = json.load(f)
    
    mids = data.get('mids', data)
    result = {}
    for k, v in mids.items():
        try:
            result[int(k)] = int(v)
        except:
            pass
    return result


def calculate_sync_velocities(
    bus: So101Bus,
    targets: dict,
    duration_s: float = 2.0,
    min_velocity: int = 30,
    max_velocity: int = 600,  # 允许补偿后的速度 (增大以支持快速同步)
    uniform_velocity: bool = False,  # 是否使用统一速度
) -> tuple:
    """
    计算同步速度 - 确保所有关节在相同时间内到达
    
    1. 读取每个关节当前位置
    2. 计算到目标的距离
    3. 找出需要最大速度的关节
    4. 如果最大速度超过上限，延长总时间
    5. 重新计算所有关节的速度
    
    如果 uniform_velocity=True，所有关节使用相同速度（max_velocity），
    这会导致各关节到达时间不同，但视觉上运动速度一致。
    
    Returns:
        (velocities, failed_joints, actual_duration): 速度字典、失败的关节列表、实际时间
    """
    distances = {}
    failed_joints = set()
    
    # 第一步：读取所有关节的当前位置和计算距离
    for jid, target in targets.items():
        # 跳过 gripper
        if jid == 6:
            continue
            
        current = None
        for attempt in range(2):
            try:
                cur_bytes = bus.read(jid, bus.PRESENT_POSITION, 2)
                current = int(cur_bytes[0]) | (int(cur_bytes[1]) << 8)
                break
            except Exception as e:
                if attempt == 1:
                    print(f"  [警告] 关节 {jid} 读取位置失败: {e}")
                    failed_joints.add(jid)
                time.sleep(0.05)
        
        if current is not None:
            distances[jid] = abs(target - current)
    
    # 第二步：找出需要移动最远的关节
    if not distances:
        return {}, failed_joints, duration_s
    
    max_distance = max(distances.values())
    max_joint = [jid for jid, d in distances.items() if d == max_distance][0]
    
    print(f"  [计算] 最大距离: {max_distance} ticks (关节 {max_joint})")
    print(f"  [计算] 目标时间: {duration_s}s, 最大速度限制: {max_velocity}")
    
    # 统一速度模式：所有关节使用相同速度
    if uniform_velocity:
        print(f"  [模式] 统一速度: 所有关节使用 vel={max_velocity}")
        actual_duration = max_distance / max_velocity if max_distance > 0 else duration_s
        print(f"  [计算] 最长关节用时: {actual_duration:.2f}s")
        
        velocities = {}
        for jid, distance in distances.items():
            if distance < 10:
                velocities[jid] = min_velocity
            else:
                velocities[jid] = max_velocity
        
        return velocities, failed_joints, actual_duration
    
    # 同步到达模式：计算实际需要的时间（基于最大速度限制）
    if max_distance > 0:
        required_velocity = max_distance / duration_s
        print(f"  [计算] 需要速度: {required_velocity:.1f}")
        if required_velocity > max_velocity:
            actual_duration = max_distance / max_velocity
            print(f"  [调整] 速度超限! 延长时间到 {actual_duration:.2f}s")
        else:
            actual_duration = duration_s
            print(f"  [计算] 速度OK, 使用 {duration_s}s")
    else:
        actual_duration = duration_s
    
    # 计算每个关节的速度（基于实际时间）
    # 由于负载和物理特性，某些关节实际速度比理论值慢，需要补偿
    VELOCITY_BOOST = {
        2: 1.5,  # Shoulder Lift - 负载大，需更多补偿
        3: 1.3,  # Elbow Flex - 负载中等
        4: 1.6,  # Wrist Flex - 负载大，需最多补偿
    }
    
    velocities = {}
    for jid, distance in distances.items():
        if distance < 10:
            velocities[jid] = min_velocity
        else:
            velocity = int(distance / actual_duration)
            
            # 速度补偿
            if jid in VELOCITY_BOOST:
                boost = VELOCITY_BOOST[jid]
                velocity = int(velocity * boost)
                joint_names = {2: "Shoulder Lift", 3: "Elbow Flex", 4: "Wrist Flex"}
                print(f"  [补偿] {joint_names[jid]} 速度 x{boost} = {velocity}")
            
            velocity = max(min_velocity, min(max_velocity, velocity))
            velocities[jid] = velocity
    
    return velocities, failed_joints, actual_duration


def go_home_sync(
    port: str,
    home_json: Path,
    duration_s: float = 2.0,
    wait_after: float = 0.5,
    uniform_velocity: bool = False,
):
    """
    同步回中位
    
    所有关节在相同时间内到达中位
    
    参数:
        uniform_velocity: 如果为 True，所有关节使用相同速度移动，
                         看起来更协调但到达时间不同
    """
    if not home_json.exists():
        print(f"[错误] 找不到: {home_json}")
        return False
    
    targets = load_home_positions(home_json)
    mp = So101Map.default()
    id_to_name = {v: k for k, v in mp.name_to_id.items()}
    
    print(f"\n同步回中位 (目标时间: {duration_s}s)")
    print("-" * 50)
    
    # 打开串口
    bus = So101Bus(port, 1_000_000, timeout=0.1, debug=False)  # 增加超时
    bus.open()
    time.sleep(0.1)  # 等待串口稳定
    
    try:
        # 计算每个关节的速度
        result = calculate_sync_velocities(bus, targets, duration_s, uniform_velocity=uniform_velocity)
        velocities, failed_joints, actual_duration = result
        
        # 设置每个关节
        time_ms = int(actual_duration * 1000)
        success_count = 0
        
        if actual_duration != duration_s:
            print(f"  [注意] 实际运动时间: {actual_duration:.2f}s (原计划: {duration_s}s)")
        
        for jid, target in targets.items():
            name = id_to_name.get(jid, f"joint_{jid}")
            
            # 跳过 gripper
            if name == 'gripper' or jid == 6:
                print(f"  [{jid}] {name:<16} 跳过")
                continue
            
            # 跳过通信失败的关节
            if jid in failed_joints:
                print(f"  [{jid}] {name:<16} 跳过 (通信失败)")
                continue
            
            vel = velocities.get(jid, 100)
            
            try:
                # 设置模式
                bus.set_operating_mode(jid, 0)
                bus.set_return_delay(jid, 0)
                bus.torque_enable(jid, True)
                
                # 设置速度和时间
                v_bytes = bytes([vel & 0xFF, (vel >> 8) & 0xFF])
                t_bytes = bytes([time_ms & 0xFF, (time_ms >> 8) & 0xFF])
                
                bus.write(jid, bus.GOAL_VELOCITY, v_bytes)
                bus.write(jid, bus.GOAL_TIME, t_bytes)
                
                # 写入目标位置
                bus.write_position(jid, target, time_ms=time_ms, velocity=vel)
                
                print(f"  [{jid}] {name:<16} -> {target} (vel={vel}, time={time_ms}ms)")
                success_count += 1
            
            except Exception as e:
                print(f"  [{jid}] {name:<16} 失败: {e}")
        
        if failed_joints:
            print(f"\n[注意] {len(failed_joints)} 个关节通信失败，请检查连接")
        
        # 等待运动完成
        print(f"\n等待 {actual_duration + wait_after}s...")
        time.sleep(actual_duration + wait_after)
        print(f"完成 ({success_count} 个关节)")
        
        return True
    
    finally:
        bus.close()


def main():
    p = argparse.ArgumentParser(description="同步回中位")
    p.add_argument("--port", type=str, default="/dev/ttyACM1")
    p.add_argument("--home-json", type=Path, default=_ROOT / "serial_home.json")
    p.add_argument("--duration", type=float, default=2.0,
                   help="所有关节到达时间 (秒)")
    p.add_argument("--wait", type=float, default=0.5,
                   help="完成后等待时间")
    p.add_argument("--uniform-velocity", action="store_true",
                   help="使用统一速度模式 (所有关节相同速度，视觉更协调)")
    
    args = p.parse_args()
    
    go_home_sync(
        port=args.port,
        home_json=args.home_json,
        duration_s=args.duration,
        wait_after=args.wait,
        uniform_velocity=args.uniform_velocity,
    )


if __name__ == "__main__":
    main()
