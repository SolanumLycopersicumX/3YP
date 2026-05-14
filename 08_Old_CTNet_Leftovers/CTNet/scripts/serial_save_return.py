#!/usr/bin/env python3
import argparse
import os
import sys
import json
import time

# 允许直接运行时找到项目根目录的 drivers 包
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from drivers.so101_serial import So101Bus, So101Map


def main():
    ap = argparse.ArgumentParser(description="记录当前各关节位置为‘归位’姿态（与 phy_control 的 --home-json 兼容）")
    ap.add_argument("--port", required=True, help="串口端口，如 /dev/ttyACM2 /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1_000_000, help="波特率，默认 1Mbps")
    ap.add_argument("--timeout", type=float, default=0.02, help="串口超时（秒）")
    ap.add_argument("--ids", type=str, default="1,2,3,4,5,6", help="要读取的ID列表，逗号分隔")
    ap.add_argument("--out", type=str, default="serial_return.json", help="保存路径（默认 serial_return.json）")
    # 夹爪闭合位选项（可用于演示前/后闭合）
    ap.add_argument("--gripper-id", type=int, default=6, help="夹爪电机ID，默认 6")
    ap.add_argument("--record-gripper-closed", action="store_true", help="将当前夹爪位置保存为闭合位")
    ap.add_argument("--gripper-closed-ticks", type=int, default=None, help="手动指定夹爪闭合位 ticks（优先于当前值）")
    args = ap.parse_args()

    ids = [int(x.strip()) for x in args.ids.split(',') if x.strip()]
    bus = So101Bus(args.port, args.baud, timeout=args.timeout, debug=False)
    bus.open()
    mp = So101Map.default()

    data = {
        "label": "return",  # 标注：这是‘归位’姿态
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "port": args.port,
        "baud": args.baud,
        "mids": {},   # id -> ticks (当前pos，供 phy_control --home-json 使用)
        "names": {},  # id -> name（便于查看）
    }

    id_to_name = {v: k for k, v in mp.name_to_id.items()}

    for i in ids:
        try:
            ok = bus.ping(i)
            if not ok:
                print(f"ID={i}: 无响应，跳过")
                continue
            pos = bus.read_position(i)
            data["mids"][str(i)] = int(pos)
            data["names"][str(i)] = id_to_name.get(i, f"id_{i}")
            print(f"ID={i}: 当前 pos={pos}，将保存为归位位置")
        except Exception as e:
            print(f"ID={i}: 读取失败：{e}")

    # 夹爪闭合位（可选）
    if args.record_gripper_closed or (args.gripper_closed_ticks is not None):
        gid = int(args.gripper_id)
        closed = args.gripper_closed_ticks
        if closed is None:
            try:
                if bus.ping(gid):
                    closed = int(bus.read_position(gid))
                    print(f"夹爪(ID={gid}) 当前 pos={closed}，保存为闭合位")
                else:
                    print(f"夹爪(ID={gid}) 无响应，无法保存闭合位")
            except Exception as e:
                print(f"夹爪闭合位读取失败：{e}")
        if closed is not None:
            data["gripper_id"] = gid
            data["gripper_closed_ticks"] = int(closed)
            data.setdefault("names", {})[str(gid)] = id_to_name.get(gid, f"id_{gid}")

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存：{args.out}")

    bus.close()


if __name__ == "__main__":
    main()

