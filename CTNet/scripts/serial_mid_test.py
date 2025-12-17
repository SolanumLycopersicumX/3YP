#!/usr/bin/env python3
import argparse
import os
import sys
import time

# 允许直接运行时找到项目根目录的 drivers 包
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from drivers.so101_serial import So101Bus


def _read_limits(bus: So101Bus, dev_id: int):
    """读取 Min/Max_Position_Limit（返回 mn, mx）。失败则返回 (None, None)。"""
    try:
        bmin = bus.read(dev_id, bus.MIN_POSITION_LIMIT, 2)
        bmax = bus.read(dev_id, bus.MAX_POSITION_LIMIT, 2)
        mn = int(bmin[0]) | (int(bmin[1]) << 8)
        mx = int(bmax[0]) | (int(bmax[1]) << 8)
        if mn > mx:
            mn, mx = mx, mn
        return mn, mx
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser(description="测试单个电机是否能回到中位（Feetech STS 总线）")
    ap.add_argument("--port", required=True, help="串口端口，如 /dev/ttyACM2 /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1_000_000, help="波特率，默认 1Mbps")
    ap.add_argument("--id", type=int, required=True, help="电机 ID")
    ap.add_argument("--timeout", type=float, default=0.02, help="串口超时（秒）")
    ap.add_argument("--time-ms", type=int, default=500, help="回中位目标到达时间（毫秒）")
    ap.add_argument("--wait", type=float, default=0.6, help="到达后额外等待（秒）")
    ap.add_argument("--tol-ticks", type=int, default=20, help="到位判定公差（ticks）")
    ap.add_argument("--only-print", action="store_true", help="只读取并打印，不执行回中位")
    # 覆盖参数（如无法读到限位）
    ap.add_argument("--min-ticks", type=int, default=None, help="覆盖最小 ticks")
    ap.add_argument("--max-ticks", type=int, default=None, help="覆盖最大 ticks")
    ap.add_argument("--mid-ticks", type=int, default=None, help="直接指定中位 ticks（优先级最高）")
    ap.add_argument("--torque-off", action="store_true", help="测试结束后关闭扭矩")
    args = ap.parse_args()

    bus = So101Bus(args.port, args.baud, timeout=args.timeout, debug=False)
    bus.open()
    dev_id = int(args.id)

    print(f"串口: {args.port} @ {args.baud}, 设备ID: {dev_id}")
    ok = bus.ping(dev_id)
    if not ok:
        print("警告：设备无响应（PING 失败），尝试继续…")

    # 基本设置
    try:
        bus.set_operating_mode(dev_id, 0)
        bus.set_return_delay(dev_id, 0)
        bus.torque_enable(dev_id, True)
    except Exception:
        pass

    # 读取当前位置
    try:
        pos = bus.read_position(dev_id)
        print(f"当前位置：ticks={pos:4d} 角度={bus.ticks_to_deg(pos):6.1f}°")
    except Exception as e:
        print(f"读取当前位置失败：{e}")
        pos = None

    # 读取限位
    mn, mx = _read_limits(bus, dev_id)
    if args.min_ticks is not None:
        mn = int(args.min_ticks)
    if args.max_ticks is not None:
        mx = int(args.max_ticks)

    if mn is not None and mx is not None:
        print(f"软限位：min={mn:4d}({bus.ticks_to_deg(mn):6.1f}°)  max={mx:4d}({bus.ticks_to_deg(mx):6.1f}°)")
        mid = (mn + mx) // 2
    else:
        print("未能读取软限位，将仅基于 --mid-ticks（若提供）或当前位置测试")
        mid = pos if pos is not None else None

    if args.mid_ticks is not None:
        mid = int(args.mid_ticks)

    if args.only_print:
        print("仅打印信息（--only-print），不执行回中位。")
        bus.close()
        return

    if mid is None:
        print("无法确定中位（mid），请提供 --mid-ticks 或 --min-ticks/--max-ticks。")
        bus.close()
        return

    # 下发回中位
    try:
        print(f"回中位：mid={mid} 角度={bus.ticks_to_deg(mid):.1f}°  用时={args.time_ms}ms…")
        bus.write_position(dev_id, mid, time_ms=args.time_ms)
    except Exception as e:
        print(f"写入回中位失败：{e}")
        bus.close()
        return

    time.sleep(max(0.0, args.time_ms / 1000.0 + args.wait))

    # 回读并判定
    try:
        pos2 = bus.read_position(dev_id)
        err = abs(int(pos2) - int(mid))
        ok = err <= int(args.tol_ticks)
        print(f"到位检查：当前位置={pos2} 误差={err} ticks  判定={'PASS' if ok else 'FAIL'}")
    except Exception as e:
        print(f"回读位置失败：{e}")

    if args.torque_off:
        try:
            bus.torque_enable(dev_id, False)
        except Exception:
            pass

    bus.close()


if __name__ == "__main__":
    main()

