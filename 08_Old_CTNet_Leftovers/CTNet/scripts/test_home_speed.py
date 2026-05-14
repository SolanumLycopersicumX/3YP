#!/usr/bin/env python3
"""
测试中位和归位速度

用法：
    # 测试回中位速度
    python scripts/test_home_speed.py --action home --velocity 100
    
    # 测试归位速度
    python scripts/test_home_speed.py --action return --velocity 200
    
    # 测试两个速度
    python scripts/test_home_speed.py --action both --home-vel 100 --return-vel 200
"""

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def test_home(port: str, velocity: int, time_ms: int = 1200):
    """测试回中位"""
    home_json = _ROOT / "serial_home.json"
    if not home_json.exists():
        print(f"[错误] 找不到: {home_json}")
        return
    
    print(f"\n{'='*60}")
    print(f"测试回中位: velocity={velocity}, time_ms={time_ms}")
    print(f"{'='*60}")
    
    cmd = f"python {_HERE}/serial_go_home.py --port {port} --home-json {home_json} --time-ms {time_ms} --velocity {velocity} --wait 1.0"
    os.system(cmd)
    time.sleep(1.5)
    
    print("\n回中位完成。观察是否平滑？")


def test_return(port: str, velocity: int, time_ms: int = 1000):
    """测试归位"""
    return_json = _ROOT / "serial_return.json"
    if not return_json.exists():
        print(f"[错误] 找不到: {return_json}")
        return
    
    print(f"\n{'='*60}")
    print(f"测试归位: velocity={velocity}, time_ms={time_ms}")
    print(f"{'='*60}")
    
    cmd = f"python {_HERE}/serial_go_return.py --port {port} --return-json {return_json} --time-ms {time_ms} --velocity {velocity} --wait 1.0"
    os.system(cmd)
    time.sleep(1.5)
    
    print("\n归位完成。观察是否平滑？")


def main():
    p = argparse.ArgumentParser(description="测试中位/归位速度")
    p.add_argument("--port", type=str, default="/dev/ttyACM1")
    p.add_argument("--action", type=str, default="home",
                   choices=["home", "return", "both"],
                   help="测试动作: home=回中位, return=归位, both=两者")
    p.add_argument("--velocity", type=int, default=100,
                   help="速度 (用于 home 或 return)")
    p.add_argument("--home-vel", type=int, default=100,
                   help="回中位速度 (用于 both)")
    p.add_argument("--return-vel", type=int, default=200,
                   help="归位速度 (用于 both)")
    p.add_argument("--time-ms", type=int, default=1200,
                   help="运动时间 (ms)")
    
    args = p.parse_args()
    
    if args.action == "home":
        test_home(args.port, args.velocity, args.time_ms)
    elif args.action == "return":
        test_return(args.port, args.velocity, args.time_ms)
    else:  # both
        test_home(args.port, args.home_vel, args.time_ms)
        input("\n按 Enter 继续测试归位...")
        test_return(args.port, args.return_vel, args.time_ms)


if __name__ == "__main__":
    main()
