#!/usr/bin/env python3
import argparse
import os
import sys
import json
import time

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from drivers.so101_serial import So101Bus, So101Map


def load_mids(path: str) -> dict[int, int]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    src = data.get('mids', data)
    mids: dict[int, int] = {}
    for k, v in src.items():
        try:
            mids[int(k)] = int(v)
        except Exception:
            pass
    gripper_closed = None
    if 'gripper_closed_ticks' in data:
        try:
            gripper_closed = int(data['gripper_closed_ticks'])
        except Exception:
            gripper_closed = None
    return mids, gripper_closed


def main():
    ap = argparse.ArgumentParser(description="将机械臂回到'归位'姿态（使用 JSON 提供的位置）")
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--timeout", type=float, default=0.02)
    ap.add_argument("--return-json", required=True, help="包含各关节归位位置的 JSON 文件")
    ap.add_argument("--time-ms", type=int, default=600, help="到达时间（ms）")
    ap.add_argument("--velocity", type=int, default=500, help="运动速度（ticks/s），越大越快")
    ap.add_argument("--wait", type=float, default=0.8, help="完成后等待（s）")
    ap.add_argument("--gripper", choices=["keep", "mid", "closed"], default="mid")
    ap.add_argument("--gripper-closed-ticks", type=int, default=None)
    args = ap.parse_args()

    mids, closed_from_json = load_mids(args.return_json)
    mp = So101Map.default()

    bus = So101Bus(args.port, args.baud, timeout=args.timeout, debug=False)
    bus.open()

    # ★★★ 先为所有关节设置快速速度 ★★★
    print(f"设置所有关节速度: velocity={args.velocity}, time_ms={args.time_ms}")
    for name, jid in mp.name_to_id.items():
        try:
            bus.set_operating_mode(jid, 0)
            bus.set_return_delay(jid, 0)
            bus.torque_enable(jid, True)
            # 设置速度和时间
            v = max(0, min(0xFFFF, args.velocity))
            t = max(0, min(0xFFFF, args.time_ms))
            bus.write(jid, bus.GOAL_VELOCITY, bytes([v & 0xFF, (v >> 8) & 0xFF]))
            bus.write(jid, bus.GOAL_TIME, bytes([t & 0xFF, (t >> 8) & 0xFF]))
        except Exception as e:
            print(f"  [警告] 设置关节 {jid} ({name}) 速度失败: {e}")
    
    time.sleep(0.1)  # 等待设置生效
    
    print("回'归位'：")
    for name, jid in mp.name_to_id.items():
        try:
            if name == 'gripper' or jid == 6:
                mode = args.gripper
                if mode == 'keep':
                    print(f"[{jid}] gripper 保持当前")
                    continue
                elif mode == 'mid':
                    tgt = mids.get(jid, None)
                else:
                    tgt = args.gripper_closed_ticks
                    if tgt is None:
                        tgt = closed_from_json
                    if tgt is None:
                        try:
                            bmin = bus.read(jid, bus.MIN_POSITION_LIMIT, 2)
                            tgt = int(bmin[0]) | (int(bmin[1]) << 8)
                        except Exception:
                            tgt = None
            else:
                tgt = mids.get(jid, None)
            if tgt is None:
                print(f"[{jid:02d}] {name:<16} 未找到目标，跳过")
                continue
            print(f"[{jid:02d}] {name:<16} -> {tgt} (time={args.time_ms}ms, velocity={args.velocity})")
            bus.write_position(jid, int(tgt), time_ms=args.time_ms, velocity=args.velocity)
        except Exception as e:
            print(f"[{jid:02d}] {name:<16} 回归位失败: {e}")
    time.sleep(max(0.0, args.time_ms / 1000.0 + args.wait))
    print("完成")
    bus.close()


if __name__ == "__main__":
    main()

