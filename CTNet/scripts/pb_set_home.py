import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import pybullet as pb
    import pybullet_data
except Exception:
    pb = None
    pybullet_data = None


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Load a URDF in PyBullet and set 6 motors to positions inside their [min,max] limits "
            "to design a neutral (middle) arm pose."
        )
    )
    p.add_argument("--urdf", type=Path, required=True, help="Path to SO101 URDF file")
    p.add_argument("--gui", action="store_true", help="Open PyBullet GUI window")
    p.add_argument("--fixed-base", action="store_true", help="Load robot with fixed base")
    p.add_argument(
        "--ratios",
        type=str,
        default=None,
        help=(
            "Comma-separated ratios in [0,1] for each motor (6 values). "
            "0 -> joint lower limit, 0.5 -> middle, 1 -> upper limit. "
            "If not provided, all 0.5 (middle)."
        ),
    )
    p.add_argument(
        "--include",
        type=str,
        default=None,
        help=(
            "Comma-separated joint indices to include (overrides auto-detect 6 motors). "
            "Example: 0,1,2,3,4,6"
        ),
    )
    p.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated joint indices to exclude from control.",
    )
    p.add_argument("--hold", type=float, default=3.0, help="Seconds to hold the pose (GUI)")
    p.add_argument("--save", type=Path, default=None, help="Optional JSON path to save joint positions")
    # Built-in SO101 calibration support (encoder counts)
    p.add_argument(
        "--so101-calib",
        action="store_true",
        help=(
            "Use built-in SO101 calibration (encoder counts) to compute ratios per joint name and set the pose. "
            "Names supported: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper."
        ),
    )
    p.add_argument(
        "--so101-mode",
        choices=["pos", "mid"],
        default="pos",
        help=(
            "When --so101-calib is enabled: pos -> use the POS column as target; mid -> use (MIN+MAX)/2 as target."
        ),
    )
    p.add_argument(
        "--gripper-closed",
        action="store_true",
        help=(
            "When --so101-calib is enabled, force 'gripper' to its closed state (ratio=0.0 -> near lower limit), "
            "ignoring POS/MID for that joint."
        ),
    )
    return p.parse_args()


def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def main():
    if pb is None:
        raise ImportError("pybullet is not installed. Install with: pip install pybullet")

    args = parse_args()

    mode = pb.GUI if args.gui else pb.DIRECT
    cid = pb.connect(mode)

    # Search paths: pybullet_data then URDF directory
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
    n_joints = pb.getNumJoints(bid)

    # Build name->index map and collect movable
    name_to_index = {}
    movable: List[int] = []
    for j in range(n_joints):
        info = pb.getJointInfo(bid, j)
        jtype = info[2]
        jname = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
        name_to_index[jname] = j
        if jtype in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC):
            movable.append(j)

    # Choose motors (6) by include list, SO101 names, or first 6 movable
    if args.include:
        motors = [int(s.strip()) for s in args.include.split(",") if s.strip()]
    elif args.so101_calib:
        so101_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        motors = []
        missing = []
        for nm in so101_names:
            if nm in name_to_index:
                motors.append(name_to_index[nm])
            else:
                missing.append(nm)
        if missing:
            print(f"Warning: SO101 joints not found in URDF: {missing}")
    else:
        motors = movable[:6]
        if len(motors) < 6:
            print(f"Warning: only found {len(motors)} movable joints; proceeding with those: {motors}")
    if args.exclude:
        ex = {int(s.strip()) for s in args.exclude.split(",") if s.strip()}
        motors = [j for j in motors if j not in ex]

    # Parse ratios or use SO101 calibration (counts -> ratio)
    ratios: List[float] = []
    if args.so101_calib:
        calib = {
            "shoulder_pan":   {"min": 818,  "pos": 2059, "max": 3441},
            "shoulder_lift":  {"min": 809,  "pos": 2006, "max": 3222},
            "elbow_flex":     {"min": 868,  "pos": 2062, "max": 3089},
            "wrist_flex":     {"min": 858,  "pos": 2072, "max": 3239},
            "wrist_roll":     {"min": 105,  "pos": 2036, "max": 3956},
            "gripper":        {"min": 1375, "pos": 1424, "max": 2941},
        }
        # Build reverse map index->name
        # Reconstruct name_to_index to ensure availability
        tmp_map = {}
        for j in range(n_joints):
            info = pb.getJointInfo(bid, j)
            jname = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
            tmp_map[j] = jname
        for j in motors:
            nm = tmp_map.get(j, None)
            if nm in calib:
                row = calib[nm]
                lo_c, hi_c = float(row["min"]), float(row["max"])
                if args.gripper_closed and nm == "gripper":
                    # Force closed state (near lower limit)
                    r = 0.0
                else:
                    tgt_c = float(row["pos"]) if args.so101_mode == "pos" else (lo_c + hi_c) * 0.5
                    r = 0.5
                    if hi_c > lo_c:
                        r = clamp((tgt_c - lo_c) / (hi_c - lo_c), 0.0, 1.0)
            else:
                r = 0.5
            ratios.append(r)
    else:
        if args.ratios:
            parts = [float(s.strip()) for s in args.ratios.split(",") if s.strip()]
            if len(parts) == 1:
                ratios = parts * len(motors)
            else:
                assert len(parts) == len(motors), "Provide either 1 ratio or exactly 6 ratios"
                ratios = parts
        else:
            ratios = [0.5] * len(motors)  # default: middle

    # Compute target positions per motor
    targets: List[Tuple[int, float]] = []
    report = []
    for i, j in enumerate(motors):
        info = pb.getJointInfo(bid, j)
        jname = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
        lo, hi = float(info[8]), float(info[9])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            # fallback: small window around 0
            lo, hi = -1.0, 1.0
        r = clamp(ratios[i], 0.0, 1.0)
        pos = lo + r * (hi - lo)
        targets.append((j, pos))
        report.append({"index": j, "name": jname, "lower": lo, "upper": hi, "ratio": r, "position": pos})

    # Apply targets
    for j, pos in targets:
        pb.setJointMotorControl2(bid, j, controlMode=pb.POSITION_CONTROL, targetPosition=pos, force=300)

    # Step simulation to reach pose
    for _ in range(240):
        pb.stepSimulation()
        if args.gui:
            time.sleep(1.0 / 240.0)

    # Print summary
    print("Configured motor targets (index, name, lower, upper, ratio -> position):")
    for it in report:
        print(
            f"[{it['index']:02d}] {it['name']:<24s} limits=({it['lower']:.4f},{it['upper']:.4f}) "
            f"ratio={it['ratio']:.2f} -> pos={it['position']:.4f}"
        )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved motor configuration to {args.save}")

    if args.gui and args.hold > 0:
        print(f"Holding pose for {args.hold} seconds ...")
        time.sleep(args.hold)

    pb.disconnect()


if __name__ == "__main__":
    main()
