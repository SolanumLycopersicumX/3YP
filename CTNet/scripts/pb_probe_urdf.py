import argparse
import time
from pathlib import Path

import numpy as np

try:
    import pybullet as pb
    import pybullet_data
except Exception as e:
    pb = None


def parse_args():
    p = argparse.ArgumentParser(description="Inspect a URDF in PyBullet and print joint/link info.")
    p.add_argument("--urdf", type=Path, required=True, help="Path to URDF file.")
    p.add_argument("--gui", action="store_true", help="Open PyBullet GUI window.")
    p.add_argument("--fixed-base", action="store_true", help="Load robot with fixed base.")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep with GUI open.")
    return p.parse_args()


def main():
    if pb is None:
        raise ImportError("pybullet is not installed. Install with: pip install pybullet pybullet_data")

    args = parse_args()
    mode = pb.GUI if args.gui else pb.DIRECT
    cid = pb.connect(mode)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.81)
    pb.loadURDF("plane.urdf")

    urdf_path = str(args.urdf)
    bid = pb.loadURDF(urdf_path, useFixedBase=args.fixed_base)
    n_joints = pb.getNumJoints(bid)

    type_map = {
        pb.JOINT_REVOLUTE: "REVOLUTE",
        pb.JOINT_PRISMATIC: "PRISMATIC",
        pb.JOINT_SPHERICAL: "SPHERICAL",
        pb.JOINT_PLANAR: "PLANAR",
        pb.JOINT_FIXED: "FIXED",
    }

    print(f"URDF: {urdf_path}")
    print(f"Num joints: {n_joints}")
    candidate = None
    for j in range(n_joints):
        info = pb.getJointInfo(bid, j)
        jname = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
        jtype = type_map.get(info[2], str(info[2]))
        parent = info[16] if len(info) > 16 else info[8]
        link_name = info[12].decode("utf-8") if len(info) > 12 and isinstance(info[12], (bytes, bytearray)) else str(info[12]) if len(info) > 12 else ""
        ll = info[8] if len(info) > 8 else None
        ul = info[9] if len(info) > 9 else None
        print(f"[{j}] joint='{jname}' link='{link_name}' type={jtype} parent={parent}")
        if jtype in ("REVOLUTE", "PRISMATIC"):
            candidate = j

    if candidate is not None:
        print(f"\nSuggested end-effector link index (last movable): {candidate}")
    else:
        print("\nNo movable joints found; robot may be fixed-only.")

    if args.gui and args.sleep > 0:
        time.sleep(args.sleep)

    pb.disconnect()


if __name__ == "__main__":
    main()

