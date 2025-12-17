import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import pybullet as pb
    import pybullet_data
except Exception:
    pb = None
    pybullet_data = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "交互式关节调参器：自由调整每个关节的位置，并在 PyBullet 中实时可视化。\n"
            "在提示符内输入 'help' 查看可用命令。"
        )
    )
    p.add_argument("--urdf", type=Path, required=True, help="机器人 URDF 文件路径（例如 SO101）")
    p.add_argument("--gui", action="store_true", help="使用 PyBullet GUI 窗口")
    p.add_argument("--fixed-base", action="store_true", help="以固定底座方式加载机器人")
    p.add_argument("--sleep", type=float, default=1.0 / 240.0, help="GUI 模式下每步模拟的暂停时间（秒）")
    p.add_argument("--steps", type=int, default=240, help="每次 set/inc 后默认步进的模拟步数")
    p.add_argument("--include", type=str, default=None, help="仅控制的关节索引（逗号分隔，可选）")
    p.add_argument("--exclude", type=str, default=None, help="排除控制的关节索引（逗号分隔，可选）")
    p.add_argument("--so101-calib", action="store_true", help="启用 SO101 编码计数校准（用于 setc 命令）")
    # 终端界面（不覆盖输入的自动刷新）
    p.add_argument("--tui", action="store_true", help="启用终端界面（指令栏与状态栏固定显示并自动刷新）")
    p.add_argument("--tui-interval", type=float, default=0.5, help="TUI 自动刷新间隔（秒）")
    # 缓动/慢动作设置（goto 分段插值）
    p.add_argument("--goto-segments", type=int, default=20, help="goto 的插值段数（越大越慢更平滑）")
    p.add_argument("--goto-wait", type=float, default=0.0, help="goto 每段后的等待秒数（可选）")
    p.add_argument("--goto-tol", type=float, default=0.01, help="goto 判定到位公差（米），默认 0.01")
    return p.parse_args()


def _so101_calib() -> Dict[str, Dict[str, float]]:
    # Encoder counts calibration provided by user
    return {
        "shoulder_pan":   {"min": 818,  "pos": 2059, "max": 3441},
        "shoulder_lift":  {"min": 809,  "pos": 2006, "max": 3222},
        "elbow_flex":     {"min": 868,  "pos": 2062, "max": 3089},
        "wrist_flex":     {"min": 858,  "pos": 2072, "max": 3239},
        "wrist_roll":     {"min": 105,  "pos": 2036, "max": 3956},
        "gripper":        {"min": 1375, "pos": 1424, "max": 2941},
    }


def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def fmt_joint_line(j: int, name: str, jtype: int, lo: float, hi: float, pos: float, xyz: Tuple[float, float, float] | None = None) -> str:
    types = {
        pb.JOINT_REVOLUTE: "旋转",
        pb.JOINT_PRISMATIC: "滑移",
        pb.JOINT_FIXED: "固定",
        pb.JOINT_SPHERICAL: "球形",
        pb.JOINT_PLANAR: "平面",
    }
    tname = types.get(jtype, str(jtype))
    rng = (hi - lo) if np.isfinite(hi - lo) and (hi > lo) else 0.0
    pct = clamp((pos - lo) / rng, 0.0, 1.0) * 100.0 if rng > 0 else float('nan')
    if xyz is None:
        return f"[{j:02d}] {name:<24s} 类型={tname:<4s} 限位=({lo:.4f},{hi:.4f}) 位置={pos:.4f} | {pct:5.1f}%"
    else:
        x, y, z = xyz
        return (
            f"[{j:02d}] {name:<24s} 类型={tname:<4s} 限位=({lo:.4f},{hi:.4f}) "
            f"位置={pos:.4f} | {pct:5.1f}% | x={x:.4f}, y={y:.4f}, z={z:.4f}"
        )


def main():
    if pb is None:
        print("未安装 pybullet。请先运行：pip install pybullet")
        sys.exit(1)

    args = parse_args()

    mode = pb.GUI if args.gui else pb.DIRECT
    cid = pb.connect(mode)

    # Search paths
    if pybullet_data is not None:
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    urdf_dir = os.path.dirname(os.path.abspath(str(args.urdf)))
    if urdf_dir:
        pb.setAdditionalSearchPath(urdf_dir)

    pb.resetSimulation()
    pb.setGravity(0, 0, -9.81)
    try:
        pb.loadURDF("plane.urdf")
    except Exception:
        pass

    bid = pb.loadURDF(str(args.urdf), useFixedBase=args.fixed_base)
    n = pb.getNumJoints(bid)

    # Build joint info
    infos = []
    movable = []
    for j in range(n):
        info = pb.getJointInfo(bid, j)
        name = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
        jtype = info[2]
        lo, hi = float(info[8]), float(info[9])
        pos = pb.getJointState(bid, j)[0]
        infos.append((j, name, jtype, lo, hi, pos))
        if jtype in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC):
            movable.append(j)

    # Apply include/exclude
    control = list(movable)
    if args.include:
        control = [int(s.strip()) for s in args.include.split(',') if s.strip()]
    if args.exclude:
        ex = {int(s.strip()) for s in args.exclude.split(',') if s.strip()}
        control = [j for j in control if j not in ex]

    name_to_index = {name: idx for idx, name, *_ in infos}
    if args.gui and not args.tui:
        print("已打开 PyBullet GUI。可通过命令调整关节；输入 quit 退出或直接关闭窗口。")

    # Quick help
    def print_help():
        print(
            "\n命令列表：\n"
            "  help                         显示帮助\n"
            "  list                         列出关节（索引、名称、限位、位置与百分位）\n"
            "  status                       显示关节状态与末端 EE(x,y,z)\n"
            "  set   <j> <pos>              将第 j 个关节设置到绝对位置（弧度/米）\n"
            "  setr  <j> <ratio>            将第 j 个关节按比例设置，ratio∈[0,1]（映射到[min,max]）\n"
            "  inc   <j> <dpos>             将第 j 个关节在当前基础上增加 dpos（弧度/米）\n"
            "  incr  <j> <dratio>           将第 j 个关节在当前基础上按比例增加 dratio*range\n"
            "  setc  <name> <count>         （SO101）按编码计数设置关节（需 --so101-calib）\n"
            "  reset pos                    一键回到中位（所有受控关节设为 (min+max)/2）\n"
            "  ee    <idx>                  设置用于显示 EE 的末端链接索引（默认=最后一个可动关节）\n"
            "  speed <steps>                设置每次改动后的默认模拟步数（当前=%d）\n"
            "  hold  <seconds>              持续模拟给定秒数（GUI 下可视化等待）\n"
            "  save  <path.json>            保存当前关节信息到 JSON\n"
            "  quit                         退出并关闭模拟\n" % (args.steps)
        )

    if not args.tui:
        print_help()

    calib = _so101_calib() if args.so101_calib else None

    def refresh_infos():
        nonlocal infos
        new_infos = []
        for j in range(n):
            info = pb.getJointInfo(bid, j)
            name = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
            jtype = info[2]
            lo, hi = float(info[8]), float(info[9])
            pos = pb.getJointState(bid, j)[0]
            new_infos.append((j, name, jtype, lo, hi, pos))
        infos = new_infos

    def step_sim(steps: int):
        for _ in range(max(1, steps)):
            pb.stepSimulation()
            if args.gui:
                time.sleep(max(0.0, args.sleep))

    # 默认末端为最后一个可动关节
    ee_idx = control[-1] if control else (movable[-1] if movable else -1)
    # helper: move EE to xyz using IK, then report per-axis reachability
    def ee_goto(x: float, y: float, z: float, steps: int = None, tol: float | None = None) -> Tuple[bool, List[str]]:
        if ee_idx < 0:
            return False, ["未设置末端链接索引"]
        steps = steps if steps is not None else args.steps
        tol = args.goto_tol if tol is None else tol
        # keep current orientation
        cur = pb.getLinkState(bid, ee_idx)
        orn = cur[5] if isinstance(cur, (list, tuple)) and len(cur) > 5 else pb.getQuaternionFromEuler([0, 0, 0])
        try:
            q = pb.calculateInverseKinematics(bid, ee_idx, [float(x), float(y), float(z)], orn)
        except Exception:
            return False, ["IK 计算失败"]
        # apply to movable joints
        for i, j in enumerate(movable):
            if i < len(q):
                pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=float(q[i]), force=300)
        step_sim(steps)
        # check achieved pose
        try:
            achieved = pb.getLinkState(bid, ee_idx)[0]
        except Exception:
            achieved = (np.nan, np.nan, np.nan)
        msgs = []
        okx = abs(achieved[0] - x) <= tol
        oky = abs(achieved[1] - y) <= tol
        okz = abs(achieved[2] - z) <= tol
        if not okx:
            msgs.append("x轴到达限位")
        if not oky:
            msgs.append("y轴到达限位")
        if not okz:
            msgs.append("z轴到达限位")
        return okx and oky and okz, msgs

    def ee_goto_smooth(x: float, y: float, z: float, segments: int | None = None, steps_total: int | None = None, tol: float | None = None, wait: float = 0.0) -> Tuple[bool, List[str]]:
        """分段插值的缓动 goto：将当前 EE→目标分为 N 段，逐段 IK，形成慢动作。
        segments: 段数，默认取 --goto-segments
        steps_total: 总步数，均分到每段
        wait: 每段后额外等待秒数
        返回：整体是否到达、以及累积的限位提示（不重复）
        """
        segments = segments if segments is not None else max(1, args.goto_segments)
        steps_total = steps_total if steps_total is not None else args.steps
        per_seg_steps = max(1, steps_total // max(1, segments))
        tol = args.goto_tol if tol is None else tol
        try:
            cur = pb.getLinkState(bid, ee_idx)[0]
        except Exception:
            cur = (0.3, 0.0, 0.5)
        x0, y0, z0 = float(cur[0]), float(cur[1]), float(cur[2])
        ok_all = True
        msgs_all: List[str] = []
        for k in range(1, segments + 1):
            alpha = k / float(segments)
            tx = x0 + (x - x0) * alpha
            ty = y0 + (y - y0) * alpha
            tz = z0 + (z - z0) * alpha
            ok, msgs = ee_goto(tx, ty, tz, steps=per_seg_steps, tol=tol)
            if not ok:
                ok_all = False
                for m in msgs:
                    if m not in msgs_all:
                        msgs_all.append(m)
            if wait > 0:
                time.sleep(wait)
        return ok_all, msgs_all

    # helper: move EE to xyz; only检查某一轴是否到达（axis in {0:x,1:y,2:z}）
    def ee_goto_axis(x: float, y: float, z: float, axis: int, steps: int = None, tol: float = 0.01) -> Tuple[bool, Tuple[float, float, float]]:
        if ee_idx < 0:
            return False, (np.nan, np.nan, np.nan)
        steps = steps if steps is not None else args.steps
        cur = pb.getLinkState(bid, ee_idx)
        orn = cur[5] if isinstance(cur, (list, tuple)) and len(cur) > 5 else pb.getQuaternionFromEuler([0, 0, 0])
        try:
            q = pb.calculateInverseKinematics(bid, ee_idx, [float(x), float(y), float(z)], orn)
        except Exception:
            return False, (np.nan, np.nan, np.nan)
        for i, j in enumerate(movable):
            if i < len(q):
                pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=float(q[i]), force=300)
        step_sim(steps)
        try:
            achieved = pb.getLinkState(bid, ee_idx)[0]
        except Exception:
            achieved = (np.nan, np.nan, np.nan)
        arr = [achieved[0], achieved[1], achieved[2]]
        target = [x, y, z]
        ok = abs(arr[axis] - target[axis]) <= tol
        return ok, (arr[0], arr[1], arr[2])

    # axis scan: 从当前EE出发，固定其它两轴，向正/负方向探测可达范围，并二分逼近边界
    def scan_axis(axis: int, init_step: float = 0.02, tol: float = 0.01, max_span: float = 1.5) -> Tuple[float, float]:
        # 读取当前EE
        try:
            cur = pb.getLinkState(bid, ee_idx)[0]
        except Exception:
            cur = (0.3, 0.0, 0.5)
        x0, y0, z0 = cur
        base = [x0, y0, z0]

        def dir_scan(sign: int) -> float:
            # 逐步扩展，直到该方向失败，再二分逼近
            step = init_step
            lo_val = base[axis]
            hi_val = lo_val + sign * step
            span = 0.0
            # 向外扩展上界
            while abs(hi_val - base[axis]) <= max_span:
                target = [base[0], base[1], base[2]]
                target[axis] = hi_val
                ok, _ = ee_goto_axis(target[0], target[1], target[2], axis, tol=tol)
                if ok:
                    lo_val = hi_val
                    step *= 1.5
                    hi_val = lo_val + sign * step
                    span = abs(lo_val - base[axis])
                else:
                    break
            # 二分逼近边界
            a = min(lo_val, hi_val)
            b = max(lo_val, hi_val)
            for _ in range(16):
                if b - a <= tol:
                    break
                mid = 0.5 * (a + b)
                target = [base[0], base[1], base[2]]
                target[axis] = mid
                ok, _ = ee_goto_axis(target[0], target[1], target[2], axis, tol=tol)
                if ok:
                    a = mid
                else:
                    b = mid
            # 返回该方向的最远可达值
            return a if sign > 0 else b

        # 正向/负向扫描
        pos_limit = dir_scan(+1)
        # 回到基准，避免漂移
        ee_goto(base[0], base[1], base[2], steps=max(1, args.steps // 2))
        neg_limit = dir_scan(-1)
        # 回到基准
        ee_goto(base[0], base[1], base[2], steps=max(1, args.steps // 2))
        return (neg_limit, pos_limit) if neg_limit <= pos_limit else (pos_limit, neg_limit)

    def get_current_ee() -> Tuple[float, float, float]:
        try:
            cur = pb.getLinkState(bid, ee_idx)[0]
            return (float(cur[0]), float(cur[1]), float(cur[2]))
        except Exception:
            return (0.3, 0.0, 0.5)

    def autoscan_save(save_path: Path | None = None, tol: float = 0.01) -> Dict[str, Dict[str, float]]:
        # 探边 X/Y/Z，可选保存为 JSON
        base = get_current_ee()
        rx = scan_axis(0, tol=tol)
        ry = scan_axis(1, tol=tol)
        rz = scan_axis(2, tol=tol)
        result = {
            "base": {"x": base[0], "y": base[1], "z": base[2]},
            "x": {"min": rx[0], "max": rx[1]},
            "y": {"min": ry[0], "max": ry[1]},
            "z": {"min": rz[0], "max": rz[1]},
        }
        if save_path is not None:
            try:
                import json
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"保存扫描结果失败: {e}")
        return result

    def do_status_print():
        refresh_infos()
        if ee_idx >= 0:
            link = pb.getLinkState(bid, ee_idx)
            xyz = link[0] if isinstance(link, (list, tuple)) else (np.nan, np.nan, np.nan)
            print(f"EE 位置: x={xyz[0]:.4f}, y={xyz[1]:.4f}, z={xyz[2]:.4f}")
        for (j, name, jtype, lo, hi, pos) in infos:
            try:
                link_state = pb.getLinkState(bid, j)
                jxyz = link_state[0] if link_state else (np.nan, np.nan, np.nan)
            except Exception:
                jxyz = (np.nan, np.nan, np.nan)
            print(fmt_joint_line(j, name, jtype, lo, hi, pos, jxyz))

    # 非 TUI：原有交互循环
    while not args.tui:
        try:
            line = input("tuner> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd in ("h", "help", "?"):
                print_help()
            elif cmd == "list":
                refresh_infos()
                for (j, name, jtype, lo, hi, pos) in infos:
                    # 关节 j 的子链接位置
                    try:
                        link_state = pb.getLinkState(bid, j)
                        xyz = link_state[0] if link_state else (float('nan'),) * 3
                    except Exception:
                        xyz = (float('nan'),) * 3
                    print(fmt_joint_line(j, name, jtype, lo, hi, pos, xyz))
            elif cmd == "status":
                do_status_print()
            elif cmd == "speed" and len(parts) == 2:
                args.steps = int(parts[1])
                print(f"默认步数={args.steps}")
            elif cmd == "hold" and len(parts) == 2:
                t = float(parts[1])
                steps = int(max(1, round(t / max(1e-6, args.sleep))))
                step_sim(steps)
            elif cmd == "ee" and len(parts) == 2:
                ee_idx = int(parts[1])
                print(f"已将末端链接索引设置为 {ee_idx}")
            elif cmd == "set" and len(parts) == 3:
                j = int(parts[1])
                val = float(parts[2])
                info = pb.getJointInfo(bid, j)
                lo, hi = float(info[8]), float(info[9])
                val = clamp(val, lo, hi)
                pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=val, force=300)
                step_sim(args.steps)
                print("已设置。")
            elif cmd == "setr" and len(parts) == 3:
                j = int(parts[1])
                r = clamp(float(parts[2]), 0.0, 1.0)
                info = pb.getJointInfo(bid, j)
                lo, hi = float(info[8]), float(info[9])
                val = lo + r * (hi - lo)
                pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=val, force=300)
                step_sim(args.steps)
                print("已设置（比例）。")
            elif cmd == "inc" and len(parts) == 3:
                j = int(parts[1])
                d = float(parts[2])
                cur = pb.getJointState(bid, j)[0]
                info = pb.getJointInfo(bid, j)
                lo, hi = float(info[8]), float(info[9])
                val = clamp(cur + d, lo, hi)
                pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=val, force=300)
                step_sim(args.steps)
                print("已增量设置。")
            elif cmd == "incr" and len(parts) == 3:
                j = int(parts[1])
                dr = float(parts[2])
                info = pb.getJointInfo(bid, j)
                lo, hi = float(info[8]), float(info[9])
                cur = pb.getJointState(bid, j)[0]
                val = clamp(cur + dr * (hi - lo), lo, hi)
                pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=val, force=300)
                step_sim(args.steps)
                print("已按比例增量设置。")
            elif cmd == "setc" and len(parts) == 3:
                if calib is None:
                    print("请使用 --so101-calib 启用 setc <name> <count>（SO101 编码计数）功能。")
                else:
                    nm = parts[1]
                    count = float(parts[2])
                    if nm not in calib:
                        print(f"未知的关节名称：{nm}")
                    else:
                        row = calib[nm]
                        lo_c, hi_c = float(row['min']), float(row['max'])
                        r = 0.5 if hi_c <= lo_c else clamp((count - lo_c) / (hi_c - lo_c), 0.0, 1.0)
                        if nm in name_to_index:
                            j = name_to_index[nm]
                            info = pb.getJointInfo(bid, j)
                            lo, hi = float(info[8]), float(info[9])
                            val = lo + r * (hi - lo)
                            pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=val, force=300)
                            step_sim(args.steps)
                            print("已按编码计数设置。")
                        else:
                            print(f"URDF 中未找到名为 '{nm}' 的关节")
            elif cmd == "reset" and len(parts) == 2 and parts[1].lower() == "pos":
                # 一键回中：对受控关节设置到 (min+max)/2
                for j in control:
                    info = pb.getJointInfo(bid, j)
                    lo, hi = float(info[8]), float(info[9])
                    mid = (lo + hi) * 0.5 if np.isfinite(lo) and np.isfinite(hi) else 0.0
                    pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=mid, force=300)
                step_sim(args.steps)
                print("已回到中位。")
            elif cmd == "save" and len(parts) == 2:
                out = Path(parts[1])
                out.parent.mkdir(parents=True, exist_ok=True)
                data = []
                refresh_infos()
                for (j, name, jtype, lo, hi, pos) in infos:
                    data.append({
                        "index": j,
                        "name": name,
                        "type": int(jtype),
                        "lower": float(lo),
                        "upper": float(hi),
                        "position": float(pos),
                    })
                import json
                with open(out, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                print(f"已保存 {len(data)} 个关节到 {out}")
            elif cmd == "goto" and (len(parts) == 4 or len(parts) == 5):
                try:
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    seg = int(parts[4]) if len(parts) == 5 else args.goto_segments
                except Exception:
                    print("用法：goto <x> <y> <z> [segments]")
                    continue
                ok, msgs = ee_goto_smooth(tx, ty, tz, segments=seg, steps_total=args.steps, wait=max(0.0, args.goto_wait))
                if ok or not msgs:
                    print("已移动至目标位置。")
                else:
                    print("限位提示：" + "/".join(msgs))
            elif cmd == "watch" and len(parts) == 2:
                # 自动刷新 status，直到 Ctrl+C
                try:
                    interval = max(0.05, float(parts[1]))
                except Exception:
                    interval = 0.5
                print("开始自动刷新，按 Ctrl+C 停止。")
                try:
                    while True:
                        refresh_infos()
                        if ee_idx >= 0:
                            link = pb.getLinkState(bid, ee_idx)
                            xyz = link[0] if isinstance(link, (list, tuple)) else (np.nan, np.nan, np.nan)
                            print(f"EE 位置: x={xyz[0]:.4f}, y={xyz[1]:.4f}, z={xyz[2]:.4f}")
                        for (j, name, jtype, lo, hi, pos) in infos:
                            try:
                                link_state = pb.getLinkState(bid, j)
                                jxyz = link_state[0] if link_state else (np.nan, np.nan, np.nan)
                            except Exception:
                                jxyz = (np.nan, np.nan, np.nan)
                            print(fmt_joint_line(j, name, jtype, lo, hi, pos, jxyz))
                        print("-" * 60)
                        time.sleep(interval)
                except KeyboardInterrupt:
                    print("已停止自动刷新。")
            elif cmd in ("scanx", "scany", "scanz"):
                ax = {"scanx": 0, "scany": 1, "scanz": 2}[cmd]
                rng = scan_axis(ax)
                axis_name = "XYZ"[ax]
                print(f"{axis_name} 轴可达范围（固定其余轴）：[{rng[0]:.4f}, {rng[1]:.4f}]")
            elif cmd in ("scanall", "autoscan"):
                # 可选保存路径
                save_path = Path(parts[1]) if len(parts) >= 2 else None
                res = autoscan_save(save_path)
                print(
                    "自动探边结果:\n"
                    f"  X: [{res['x']['min']:.4f}, {res['x']['max']:.4f}]\n"
                    f"  Y: [{res['y']['min']:.4f}, {res['y']['max']:.4f}]\n"
                    f"  Z: [{res['z']['min']:.4f}, {res['z']['max']:.4f}]\n"
                    f"  基准: (x={res['base']['x']:.4f}, y={res['base']['y']:.4f}, z={res['base']['z']:.4f})"
                )
                if save_path:
                    print(f"已保存到: {save_path}")
            elif cmd in ("q", "quit", "exit"):
                break
            else:
                print("未知命令。输入 'help' 查看用法。")
        except Exception as e:
            print(f"错误：{e}")

    # TUI 模式：使用 curses 固定显示指令与状态，并在底部输入命令
    if args.tui:
        try:
            import curses
            import locale
            locale.setlocale(locale.LC_ALL, '')

            def tui(stdscr):
                curses.curs_set(1)
                stdscr.nodelay(True)
                stdscr.timeout(int(args.tui_interval * 1000))

                def layout():
                    nonlocal status_h, input_h, height, width, status_win, input_win
                    height, width = stdscr.getmaxyx()
                    input_h = 2
                    # 合并命令提示与状态到同一窗口：status_win
                    status_h = max(3, height - input_h)
                    status_win = curses.newwin(status_h, width, 0, 0)
                    input_win = curses.newwin(input_h, width, status_h, 0)

                # 初始化布局
                status_h = input_h = height = width = 0
                status_win = input_win = None
                layout()

                # 限位提示当前状态（始终显示）。默认均未限位。
                last_limits: List[str] = ["x轴未限位", "y轴未限位", "z轴未限位"]
                last_scan_lines: List[str] = []

                def draw_status():
                    status_win.erase()
                    refresh_infos()
                    # 先画边框，避免覆盖首列文字
                    status_win.box()
                    avail_w = max(0, width - 2)  # 去除左右边框
                    # 顶部两列：中文与英文一一对应（每行一个指令）
                    help_pairs = [
                        ("帮助(无参)", "help"),
                        ("状态(无参)", "status"),
                        ("回中(无参)", "reset pos"),
                        ("末端(索引)", "ee (idx)"),
                        ("保存(JSON路径)", "save (path.json)"),
                        ("移动EE到(x,y,z)[段]", "goto (x) (y) (z) (segments可选)"),
                        ("退出(无参)", "quit"),
                        ("绝对设置(j, pos)", "set (j) (pos)"),
                        ("按比例设置(j, 0..1)", "setr (j) (ratio)"),
                        ("增量设置(j, Δpos)", "inc (j) (dpos)"),
                        ("比例增量(j, Δ比例)", "incr (j) (dratio)"),
                        (f"步数(steps=当前={args.steps})", "speed (steps)"),
                        ("等待(秒)", "hold (sec)"),
                        ("计数(名称, 计数)", "setc (name) (count)"),
                    ]

                    col_w = max(20, avail_w // 2)
                    row = 1  # 边框内起始行
                    for cn, en in help_pairs:
                        if row >= status_h - 2:
                            break
                        left = (cn or "")[: max(0, col_w - 1)]
                        right = (en or "")[: max(0, avail_w - col_w - 1)]
                        try:
                            status_win.addstr(row, 1, left)
                            if col_w < avail_w:
                                status_win.addstr(row, 1 + col_w, right)
                        except Exception:
                            pass
                        row += 1

                    # EE 位置
                    if row < status_h - 1 and ee_idx >= 0:
                        link = pb.getLinkState(bid, ee_idx)
                        xyz = link[0] if isinstance(link, (list, tuple)) else (np.nan, np.nan, np.nan)
                        head = f"EE 位置: x={xyz[0]:.4f}, y={xyz[1]:.4f}, z={xyz[2]:.4f}"
                        try:
                            status_win.addstr(row, 1, head[: max(0, avail_w - 1)])
                        except Exception:
                            pass
                        row += 1
                    # 限位提示行（始终显示；右列列出 x/y/z 的当前限位状态）
                    if row < status_h - 1:
                        col_w = max(20, avail_w // 2)
                        left = "限位提示："
                        right = "/".join(last_limits)
                        try:
                            status_win.addstr(row, 1, left[: max(0, col_w - 1)])
                            if col_w < avail_w:
                                status_win.addstr(row, 1 + col_w, right[: max(0, avail_w - col_w - 1)])
                        except Exception:
                            pass
                        row += 1
                    # 扫描结果（最近一次，多行显示）
                    if last_scan_lines:
                        for ln in last_scan_lines:
                            if row >= status_h - 1:
                                break
                            try:
                                status_win.addstr(row, 1, ln[: max(0, avail_w - 1)])
                            except Exception:
                                pass
                            row += 1
                    # joints
                    for (j, name, jtype, lo, hi, pos) in infos:
                        if row >= status_h - 1:
                            break
                        try:
                            link_state = pb.getLinkState(bid, j)
                            jxyz = link_state[0] if link_state else (np.nan, np.nan, np.nan)
                        except Exception:
                            jxyz = (np.nan, np.nan, np.nan)
                        line = fmt_joint_line(j, name, jtype, lo, hi, pos, jxyz)
                        try:
                            status_win.addstr(row, 1, line[: max(0, avail_w - 1)])
                        except Exception:
                            pass
                        row += 1
                    status_win.noutrefresh()

                def draw_input(buf: str):
                    input_win.erase()
                    prompt = "tuner> "
                    try:
                        input_win.addstr(0, 0, (prompt + buf)[: max(0, width - 1)])
                    except Exception:
                        pass
                    input_win.noutrefresh()

                buffer = ""
                draw_status()
                draw_input(buffer)
                curses.doupdate()

                while True:
                    ch = stdscr.getch()
                    if ch == curses.KEY_RESIZE:
                        layout()
                        draw_status()
                        draw_input(buffer)
                        curses.doupdate()
                        continue
                    if ch == -1:
                        # timeout -> refresh status and input without breaking input buffer
                        draw_status()
                        draw_input(buffer)
                        curses.doupdate()
                        continue
                    if ch in (curses.KEY_ENTER, 10, 13):
                        line = buffer.strip()
                        buffer = ""
                        # process line with existing handlers (non-TUI branch code)
                        if line:
                            parts = line.split()
                            cmd = parts[0].lower()
                            try:
                                if cmd in ("h", "help", "?"):
                                    pass  # 顶部已包含常用命令
                                elif cmd == "list":
                                    pass  # status窗已展示
                                elif cmd == "status":
                                    pass
                                elif cmd == "speed" and len(parts) == 2:
                                    args.steps = int(parts[1])
                                elif cmd == "hold" and len(parts) == 2:
                                    t = float(parts[1])
                                    steps = int(max(1, round(t / max(1e-6, args.sleep))))
                                    step_sim(steps)
                                elif cmd == "ee" and len(parts) == 2:
                                    nonlocal ee_idx
                                    ee_idx = int(parts[1])
                                elif cmd == "set" and len(parts) == 3:
                                    j = int(parts[1]); val = float(parts[2])
                                    info = pb.getJointInfo(bid, j); lo, hi = float(info[8]), float(info[9])
                                    val = clamp(val, lo, hi)
                                    pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=val, force=300)
                                    step_sim(args.steps)
                                elif cmd == "setr" and len(parts) == 3:
                                    j = int(parts[1]); r = clamp(float(parts[2]), 0.0, 1.0)
                                    info = pb.getJointInfo(bid, j); lo, hi = float(info[8]), float(info[9])
                                    val = lo + r * (hi - lo)
                                    pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=val, force=300)
                                    step_sim(args.steps)
                                elif cmd == "inc" and len(parts) == 3:
                                    j = int(parts[1]); d = float(parts[2])
                                    cur = pb.getJointState(bid, j)[0]
                                    info = pb.getJointInfo(bid, j); lo, hi = float(info[8]), float(info[9])
                                    val = clamp(cur + d, lo, hi)
                                    pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=val, force=300)
                                    step_sim(args.steps)
                                elif cmd == "incr" and len(parts) == 3:
                                    j = int(parts[1]); dr = float(parts[2])
                                    info = pb.getJointInfo(bid, j); lo, hi = float(info[8]), float(info[9])
                                    cur = pb.getJointState(bid, j)[0]
                                    val = clamp(cur + dr * (hi - lo), lo, hi)
                                    pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=val, force=300)
                                    step_sim(args.steps)
                                elif cmd == "setc" and len(parts) == 3 and calib is not None:
                                    nm = parts[1]; count = float(parts[2])
                                    if nm in calib and nm in name_to_index:
                                        j = name_to_index[nm]
                                        info = pb.getJointInfo(bid, j); lo, hi = float(info[8]), float(info[9])
                                        lo_c, hi_c = float(calib[nm]['min']), float(calib[nm]['max'])
                                        r = 0.5 if hi_c <= lo_c else clamp((count - lo_c) / (hi_c - lo_c), 0.0, 1.0)
                                        val = lo + r * (hi - lo)
                                        pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=val, force=300)
                                        step_sim(args.steps)
                                elif cmd == "reset" and len(parts) == 2 and parts[1].lower() == "pos":
                                    for j in control:
                                        info = pb.getJointInfo(bid, j)
                                        lo, hi = float(info[8]), float(info[9])
                                        mid = (lo + hi) * 0.5 if np.isfinite(lo) and np.isfinite(hi) else 0.0
                                        pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=mid, force=300)
                                    step_sim(args.steps)
                                elif cmd == "goto" and (len(parts) == 4 or len(parts) == 5):
                                    try:
                                        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                                        seg = int(parts[4]) if len(parts) == 5 else args.goto_segments
                                        ok, msgs = ee_goto_smooth(tx, ty, tz, segments=seg, steps_total=args.steps, wait=max(0.0, args.goto_wait))
                                        # 重置为未限位
                                        x_stat = "x轴未限位"; y_stat = "y轴未限位"; z_stat = "z轴未限位"
                                        if not ok and msgs:
                                            # 根据返回的命中轴设置对应为到达限位
                                            for m in msgs:
                                                if m.startswith("x轴"):
                                                    x_stat = "x轴到达限位"
                                                elif m.startswith("y轴"):
                                                    y_stat = "y轴到达限位"
                                                elif m.startswith("z轴"):
                                                    z_stat = "z轴到达限位"
                                        last_limits = [x_stat, y_stat, z_stat]
                                    except Exception:
                                        last_limits = ["用法：goto <x> <y> <z> [segments]", "", ""]
                                elif cmd in ("q", "quit", "exit"):
                                    return
                                elif cmd in ("scanx", "scany", "scanz"):
                                    ax = {"scanx": 0, "scany": 1, "scanz": 2}[cmd]
                                    rng = scan_axis(ax)
                                    axis_name = "XYZ"[ax]
                                    last_scan_lines = [f"{axis_name} 轴可达范围: [{rng[0]:.4f}, {rng[1]:.4f}]（固定其他轴）"]
                                elif cmd in ("scanall", "autoscan"):
                                    save_path = Path(parts[1]) if len(parts) >= 2 else None
                                    res = autoscan_save(save_path)
                                    last_scan_lines = [
                                        f"X 轴可达范围: [{res['x']['min']:.4f}, {res['x']['max']:.4f}]",
                                        f"Y 轴可达范围: [{res['y']['min']:.4f}, {res['y']['max']:.4f}]",
                                        f"Z 轴可达范围: [{res['z']['min']:.4f}, {res['z']['max']:.4f}]",
                                        f"基准: (x={res['base']['x']:.4f}, y={res['base']['y']:.4f}, z={res['base']['z']:.4f})",
                                        (f"已保存到: {save_path}" if save_path else ""),
                                    ]
                            except Exception:
                                pass
                        draw_status()
                        draw_input("")
                        curses.doupdate()
                        continue
                    elif ch in (curses.KEY_BACKSPACE, 127, 8):
                        buffer = buffer[:-1]
                    elif 32 <= ch <= 126:
                        buffer += chr(ch)
                    draw_input(buffer)
                    curses.doupdate()

            curses.wrapper(tui)
        except Exception as e:
            print(f"TUI 出错：{e}")

    pb.disconnect()


if __name__ == "__main__":
    main()
