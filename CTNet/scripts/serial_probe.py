#!/usr/bin/env python3
import argparse
import time
import os
import sys

# 允许直接用 `python scripts/serial_probe.py` 运行时找到项目根目录下的 drivers
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from drivers.so101_serial import So101Bus


def main():
    p = argparse.ArgumentParser(description="SO101 串口探测：Ping/读位置/小幅移动(可选)")
    p.add_argument("--port", required=True, help="串口端口，如 /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=1_000_000, help="波特率，默认 1Mbps")
    p.add_argument("--ids", type=str, default="1,2,3,4,5,6", help="扫描的ID列表，逗号分隔")
    p.add_argument("--move", action="store_true", help="对检测到的ID做±30 ticks 小幅位移验证")
    p.add_argument("--timeout", type=float, default=0.02, help="串口超时秒")
    args = p.parse_args()

    ids = [int(x.strip()) for x in args.ids.split(',') if x.strip()]
    bus = So101Bus(args.port, args.baud, timeout=args.timeout, debug=False)
    bus.open()
    print(f"打开串口: {args.port} @ {args.baud}")

    found = []
    for i in ids:
        ok = bus.ping(i)
        print(f"ID={i}: {'在线' if ok else '无响应'}")
        if ok:
            found.append(i)

    for i in found:
        try:
            pos = bus.read_position(i)
            print(f"ID={i}: 当前位置 ticks={pos} 角度={So101Bus.ticks_to_deg(pos):.2f}°")
        except Exception as e:
            print(f"ID={i}: 读位置失败: {e}")

    if args.move:
        for i in found:
            try:
                cur = bus.read_position(i)
                print(f"ID={i}: 准备小幅移动… 当前={cur}")
                bus.set_operating_mode(i, 0)
                bus.set_return_delay(i, 0)
                bus.torque_enable(i, True)
                bus.write_position(i, cur + 30, time_ms=300)
                time.sleep(0.4)
                bus.write_position(i, cur - 30, time_ms=300)
                time.sleep(0.4)
                bus.write_position(i, cur, time_ms=300)
                time.sleep(0.4)
            except Exception as e:
                print(f"ID={i}: 小幅移动失败: {e}")

    bus.close()
    print("完成")


if __name__ == "__main__":
    main()
