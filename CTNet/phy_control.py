import argparse
import os
import time
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from serial_arm_env import SerialArmEnv, SerialConfig
from drivers.so101_serial import So101Bus, So101Map


# 与 gym_control 保持一致：类别到方向的映射
# 0: left, 1: right, 2: up, 3: down
MOVE_MAP = {
    0: 0,  # left
    1: 1,  # right
    2: 2,  # up
    3: 3,  # down
}


def number_class_channel(dataset_type: str) -> Tuple[int, int]:
    if dataset_type == "A":
        return 4, 22
    if dataset_type == "B":
        return 2, 3
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def load_data(dir_path: Path, dataset_type: str, subject_id: int, mode: str):
    suffix = "T" if mode == "train" else "E"
    mat_path = dir_path / f"{dataset_type}{subject_id:02d}{suffix}.mat"
    data_mat = loadmat(mat_path)
    return data_mat["data"], data_mat["label"]


def load_data_evaluate(dir_path: Path, dataset_type: str, subject_id: int, evaluate_mode: str):
    if evaluate_mode == "LOSO":
        x_train, y_train = None, None
        for i in range(1, 10):
            x1, y1 = load_data(dir_path, dataset_type, i, mode="train")
            x2, y2 = load_data(dir_path, dataset_type, i, mode="test")
            x = np.concatenate([x1, x2], axis=0)
            y = np.concatenate([y1, y2], axis=0)
            if i == subject_id:
                x_test, y_test = x, y
            elif x_train is None:
                x_train, y_train = x, y
            else:
                x_train = np.concatenate([x_train, x], axis=0)
                y_train = np.concatenate([y_train, y], axis=0)
        return x_train, y_train, x_test, y_test
    else:
        x_train, y_train = load_data(dir_path, dataset_type, subject_id, mode="train")
        x_test, y_test = load_data(dir_path, dataset_type, subject_id, mode="test")
        return x_train, y_train, x_test, y_test


def expand_labels(labels: Sequence[int]) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        return labels
    if labels.min() == 1:
        labels = labels - 1
    return labels.astype(int)


def parse_args():
    p = argparse.ArgumentParser(description="物理 SO101 串口演示：用 CTNet 预测驱动关节（shoulder_pan/wrist_flex）")
    # 数据与模型
    p.add_argument("--subject", type=int, default=2)
    p.add_argument("--dataset", choices=["A", "B"], default="A")
    p.add_argument("--evaluate-mode", choices=["LOSO", "subject-dependent"], default="subject-dependent")
    p.add_argument("--data-dir", type=Path, default=Path("./mymat_raw/"))
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--demo-mode", choices=["pred", "ground_truth", "random"], default="pred")
    p.add_argument("--repeat", type=int, default=1, help="每个动作重复次数，用于放慢/拉长动作")
    p.add_argument("--sleep", type=float, default=0.2, help="每步之间的节流/延时（秒）")
    # 串口
    p.add_argument("--serial-port", type=str, required=True, help="串口端口，如 /dev/ttyUSB0")
    p.add_argument("--serial-baud", type=int, default=1_000_000, help="波特率，默认 1Mbps")
    p.add_argument("--serial-timeout", type=float, default=0.02, help="串口超时（秒）")
    p.add_argument("--serial-move-time", type=int, default=300, help="目标到达时间(ms)，位置模式更平滑")
    p.add_argument("--serial-step-time-scale", type=float, default=2.0, help="每步移动时间缩放系数，默认 2.0（翻倍）")
    # 预处理：先回中位
    p.add_argument("--pre-home-mid", dest="pre_home_mid", action="store_true", help="启动前读取当前各关节并回到中位")
    p.add_argument("--no-pre-home-mid", dest="pre_home_mid", action="store_false", help="跳过回中位步骤")
    p.set_defaults(pre_home_mid=True)
    p.add_argument("--home-time", type=int, default=500, help="回中位的到达时间(ms)")
    p.add_argument("--home-wait", type=float, default=0.6, help="回中位后等待时间(s)")
    p.add_argument("--home-json", type=Path, default=None, help="可选：从JSON加载各关节中位 ticks（优先于(min+max)/2）")
    p.add_argument("--pre-home-source", choices=["json", "limits"], default="json",
                   help="回中位目标的来源：json=使用 --home-json 的中位；limits=忽略 JSON，使用(min+max)/2")
    p.add_argument("--gripper-home", choices=["keep", "mid", "closed"], default="keep",
                   help="回中位阶段 6号夹爪的行为：keep=保持当前(默认)，mid=回中位，closed=闭合")
    p.add_argument("--gripper-closed-ticks", type=int, default=None,
                   help="当 --gripper-home closed 时的闭合目标ticks；未提供则用最小限位")
    # 演示结束后归位
    p.add_argument("--post-home", dest="post_home", action="store_true", help="演示结束后归位")
    p.add_argument("--no-post-home", dest="post_home", action="store_false", help="跳过演示结束后的归位")
    p.set_defaults(post_home=False)
    p.add_argument("--post-home-time", type=int, default=None, help="演示结束后归位时间(ms)，默认沿用 --home-time")
    p.add_argument("--post-home-wait", type=float, default=None, help="演示结束后归位等待(s)，默认沿用 --home-wait")
    p.add_argument("--post-home-json", type=Path, default=None, help="演示结束后归位使用的 JSON（若未提供则使用 --home-json）")
    p.add_argument("--post-home-source", choices=["json", "limits"], default="json",
                   help="归位目标的来源：json=使用 --post-home-json/--home-json；limits=忽略 JSON，使用(min+max)/2")
    p.add_argument("--post-gripper-home", choices=["keep", "mid", "closed"], default=None,
                   help="演示结束时夹爪行为（不设则沿用 --gripper-home）")
    p.add_argument("--post-gripper-closed-ticks", type=int, default=None, help="演示结束时闭合位 ticks（优先级：该参数 > JSON > 最小限位）")
    # 关节驱动配置
    p.add_argument("--joint-step", type=float, default=0.05, help="每步弧度，默认 0.05rad")
    p.add_argument("--joint-lr-name", type=str, default="shoulder_pan", help="左右映射的关节名")
    p.add_argument("--joint-ud-name", type=str, default="elbow_flex", help="上下映射的关节名")
    p.add_argument("--invert-lr", action="store_true", help="反转左右方向")
    p.add_argument("--invert-ud", action="store_true", help="反转上下方向")
    # 显示/输出
    p.add_argument("--tui", action="store_true", help="启用终端界面（顶部指令+状态）")
    p.add_argument("--tui-interval", type=float, default=0.25, help="TUI 刷新间隔（秒）")
    p.add_argument("--show-probs", action="store_true", help="显示模型输出的类别概率（pred 模式）")
    p.add_argument("--save-preds", type=Path, default=None, help="保存每步预测与概率到CSV")
    return p.parse_args()


def load_model(dataset_type: str, subject_id: int, device: torch.device, model_path: Optional[Path]):
    # 延迟导入，避免无 GPU 时初始化开销
    import sys
    from CTNet_model import (
        EEGTransformer,
        BranchEEGNetTransformer,
        PatchEmbeddingCNN,
        PositioinalEncoding,
        TransformerEncoder,
        TransformerEncoderBlock,
        MultiHeadAttention,
        FeedForwardBlock,
        ResidualAdd,
        ClassificationHead,
    )
    try:
        from torch.serialization import add_safe_globals, safe_globals  # torch>=2.6
    except Exception:
        add_safe_globals = None
        safe_globals = None

    if model_path is None:
        model_path = Path("./models/new") / f"model_{subject_id}.pth"

    safe_types = [
        EEGTransformer,
        BranchEEGNetTransformer,
        PatchEmbeddingCNN,
        PositioinalEncoding,
        TransformerEncoder,
        TransformerEncoderBlock,
        MultiHeadAttention,
        FeedForwardBlock,
        ResidualAdd,
        ClassificationHead,
    ]

    def _alias_main():
        main_mod = sys.modules.setdefault("__main__", sys.modules.get("__main__"))
        for cls in safe_types:
            setattr(main_mod, cls.__name__, cls)

    def _extract_state(state):
        if hasattr(state, "state_dict") and callable(getattr(state, "state_dict")):
            return state.state_dict()
        if isinstance(state, dict):
            for k in ["state_dict", "model_state", "model", "net", "weights"]:
                if k in state and isinstance(state[k], (dict,)):
                    inner = state[k]
                    if hasattr(inner, "state_dict"):
                        return inner.state_dict()
                    if isinstance(inner, dict):
                        return inner
            return state
        return state

    def _load_ckpt(path: Path):
        if safe_globals is not None:
            try:
                _alias_main()
                with safe_globals(safe_types):
                    obj = torch.load(path, map_location=device, weights_only=True)
                return _extract_state(obj)
            except Exception:
                pass
        try:
            _alias_main()
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)
        except Exception as e1:
            if add_safe_globals is not None:
                try:
                    add_safe_globals(safe_types)
                    _alias_main()
                    return torch.load(path, map_location=device)
                except Exception:
                    pass
            raise e1

    state = _load_ckpt(model_path)
    state = _extract_state(state)

    _, ch = number_class_channel(dataset_type)
    model = EEGTransformer(
        heads=2,
        emb_size=16,
        depth=6,
        database_type=dataset_type,
        eeg1_f1=8,
        eeg1_D=2,
        eeg1_kernel_size=64,
        eeg1_pooling_size1=8,
        eeg1_pooling_size2=8,
        eeg1_dropout_rate=0.5,
        eeg1_number_channel=ch,
        flatten_eeg1=240,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_actions(model, device, test_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    preds = []
    all_probs = []
    with torch.no_grad():
        for i in range(test_tensor.shape[0]):
            x = test_tensor[i].unsqueeze(0).to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                _, logits = logits
            prob = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
            cls = int(np.argmax(prob))
            preds.append(cls)
            all_probs.append(prob)
    return np.array(preds, dtype=int), np.array(all_probs)


def main():
    args = parse_args()

    # 线程库环境清理，避免 OMP 争用
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    os.environ.setdefault("KMP_AFFINITY", "none")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("KMP_SETTINGS", "0")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_CREATE_SHM", "0")

    device = torch.device(args.device)

    # 载入数据
    _, _, test_data, test_labels = load_data_evaluate(args.data_dir, args.dataset, args.subject, args.evaluate_mode)
    test_data = np.expand_dims(test_data, axis=1)  # [N,1,22,1000]
    mean, std = np.mean(test_data), np.std(test_data) or 1.0
    test_data = (test_data - mean) / std
    test_tensor = torch.tensor(test_data, dtype=torch.float32, device=device)
    labels = expand_labels(test_labels)

    # 动作来源
    actions = None
    probs_matrix = None
    if args.demo_mode == "ground_truth":
        actions = labels.copy()
    elif args.demo_mode == "random":
        rng = np.random.default_rng(2024)
        actions = rng.integers(low=0, high=4, size=min(args.num_trials, len(test_tensor)))
    else:
        model = load_model(args.dataset, args.subject, device, args.model_path)
        preds, probs = predict_actions(model, device, test_tensor)
        actions = preds
        probs_matrix = probs

    actions = np.asarray(actions, dtype=int)[: args.num_trials]
    if args.repeat and args.repeat > 1:
        actions = np.repeat(actions, args.repeat)

    # 解析并强制使用 JSON：回中位用 serial_home.json；归位用 serial_return.json
    from pathlib import Path as _P
    home_json_resolved = None
    post_json_resolved = None
    if args.pre_home_mid and args.pre_home_source == 'json':
        cand = args.home_json if args.home_json is not None else _P('serial_home.json')
        cand = _P(cand)
        if not cand.exists():
            raise FileNotFoundError(f"未找到中位JSON: {cand}. 请提供 --home-json 或把 serial_home.json 放在当前目录。")
        home_json_resolved = cand
    if args.post_home and args.post_home_source == 'json':
        cand2 = args.post_home_json if args.post_home_json is not None else _P('serial_return.json')
        cand2 = _P(cand2)
        if not cand2.exists():
            raise FileNotFoundError(f"未找到归位JSON: {cand2}. 请提供 --post-home-json 或把 serial_return.json 放在当前目录。")
        post_json_resolved = cand2

    # 可选：预读取状态并回中位
    if args.pre_home_mid:
        try:
            bus = So101Bus(args.serial_port, args.serial_baud, timeout=args.serial_timeout, debug=False)
            bus.open()
            mapping = So101Map.default()
            print(f"读取各关节当前位置/限位，并回中位… (home-json={str(home_json_resolved) if home_json_resolved else 'limits'})")
            rows = []
            mids: dict[int, int] = {}
            mins: dict[int, int] = {}
            maxs: dict[int, int] = {}
            # 如果提供了 home-json，先加载其中的 mid ticks
            home_overrides: dict[int, int] = {}
            gripper_closed_from_json: int | None = None
            if home_json_resolved is not None:
                try:
                    import json
                    with open(home_json_resolved, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # 支持 {"mids": {"1":2059, ...}} 或 直接 {"1":2059, ...}
                    src = data.get('mids', data)
                    for k, v in src.items():
                        try:
                            home_overrides[int(k)] = int(v)
                        except Exception:
                            pass
                    # 可选：JSON 中的夹爪闭合位
                    try:
                        if 'gripper_closed_ticks' in data:
                            gripper_closed_from_json = int(data['gripper_closed_ticks'])
                    except Exception:
                        gripper_closed_from_json = None
                    if len(home_overrides) > 0:
                        print(f"已加载 home-json 覆盖 {len(home_overrides)} 个关节的中位。")
                except Exception as e:
                    print(f"home-json 加载失败，忽略：{e}")
            for name, jid in mapping.name_to_id.items():
                try:
                    # 基本设置
                    try:
                        bus.set_operating_mode(jid, 0)
                        bus.set_return_delay(jid, 0)
                        bus.torque_enable(jid, True)
                    except Exception:
                        pass
                    # 读 pos/min/max
                    pos = bus.read_position(jid)
                    bmin = bus.read(jid, bus.MIN_POSITION_LIMIT, 2)
                    bmax = bus.read(jid, bus.MAX_POSITION_LIMIT, 2)
                    mn = int(bmin[0]) | (int(bmin[1]) << 8)
                    mx = int(bmax[0]) | (int(bmax[1]) << 8)
                    mins[jid], maxs[jid] = mn, mx
                    # 选择回中位目标来源
                    if args.pre_home_source == "json" and jid in home_overrides:
                        mid = int(home_overrides[jid])
                        src = "json"
                    else:
                        mid = int((mn + mx) // 2)
                        src = "limits"
                    mids[jid] = mid
                    rows.append((name, jid, pos, mn, mx, mid))
                except Exception as e:
                    rows.append((name, jid, None, None, None, None))
            # 打印表格
            for (name, jid, pos, mn, mx, mid) in rows:
                if pos is None:
                    print(f"[{jid:02d}] {name:<16} - 无响应")
                else:
                    def deg(t):
                        return So101Bus.ticks_to_deg(t)
                    print(f"[{jid:02d}] {name:<16} pos={pos:4d}({deg(pos):6.1f}°)  min={mn:4d}({deg(mn):6.1f}°)  max={mx:4d}({deg(mx):6.1f}°)  mid={mid:4d}({deg(mid):6.1f}°)")
            # 下发回中位
            moved = []
            for name, jid in mapping.name_to_id.items():
                if jid not in mids:
                    continue
                try:
                    if name == 'gripper' or jid == 6:
                        # 区分 6 号夹爪
                        mode = args.gripper_home
                        if mode == 'keep':
                            continue  # 不动
                        elif mode == 'mid':
                            bus.write_position(jid, mids[jid], time_ms=args.home_time)
                            moved.append((jid, name, mids[jid]))
                        elif mode == 'closed':
                            tgt = args.gripper_closed_ticks
                            if tgt is None:
                                tgt = gripper_closed_from_json
                            if tgt is None:
                                tgt = mins.get(jid, None)
                            if tgt is not None:
                                bus.write_position(jid, int(tgt), time_ms=args.home_time)
                                moved.append((jid, name, int(tgt)))
                            else:
                                # 无法确定闭合位，保持不动
                                pass
                        else:
                            # 未知模式，保持不动
                            pass
                    else:
                        # 其它 1..5 轴回中位
                        bus.write_position(jid, mids[jid], time_ms=args.home_time)
                        moved.append((jid, name, mids[jid]))
                except Exception:
                    pass
            if moved:
                print("回中位命令已下发：")
                for jid, name, tgt in moved:
                    print(f"  [{jid:02d}] {name:<16} -> {tgt}")
            time.sleep(max(0.0, args.home_time / 1000.0 + args.home_wait))
        except Exception as e:
            print(f"预回中位步骤失败：{e}")
        finally:
            try:
                bus.close()
            except Exception:
                pass

    # 构建串口环境
    scfg = SerialConfig(
        port=args.serial_port,
        baud=int(args.serial_baud),
        timeout=float(args.serial_timeout),
        joint_lr_name=args.joint_lr_name,
        joint_ud_name=args.joint_ud_name,
        joint_step_rad=float(args.joint_step),
        invert_lr=bool(args.invert_lr),
        invert_ud=bool(args.invert_ud),
        max_steps=len(actions),
        move_time_ms=int(args.serial_move_time) if args.serial_move_time is not None else None,
        step_time_scale=float(args.serial_step_time_scale),
    )
    env = SerialArmEnv(scfg, render_mode="human")
    obs, info = env.reset()

    label_names = ["left", "right", "up", "down"] if args.dataset == "A" else ["left", "right"]
    taken_actions: List[int] = []
    pred_rows: List[dict] = []
    last_probs = None

    def run_step(a: int):
        nonlocal last_probs
        taken_actions.append(int(a))
        obs, reward, terminated, truncated, step_info = env.step(int(a))
        if args.sleep > 0:
            time.sleep(args.sleep)
        # 记录概率（若有）
        if probs_matrix is not None and len(pred_rows) < len(actions):
            i = len(pred_rows)
            if i < len(probs_matrix):
                prob = probs_matrix[i]
                last_probs = prob
                row = {"step": i, "action": int(a), "action_name": label_names[int(a)] if int(a) < len(label_names) else str(int(a))}
                for j in range(len(prob)):
                    row[f"p_{label_names[j] if j < len(label_names) else j}"] = float(prob[j])
                pred_rows.append(row)
        return obs, reward, terminated, truncated

    # TUI（可选）
    if args.tui:
        try:
            def tui(stdscr):
                curses.curs_set(0)
                stdscr.nodelay(True)
                stdscr.timeout(int(args.tui_interval * 1000))

                paused = False
                i = 0

                def get_status_text():
                    st = env.get_status()
                    act = label_names[int(actions[i])] if i < len(actions) and int(actions[i]) < len(label_names) else "-"
                    lr_t = st.get("lr_ticks", float('nan'))
                    ud_t = st.get("ud_ticks", float('nan'))
                    return (i, len(actions), act, lr_t, st.get("lr_deg", float('nan')), ud_t, st.get("ud_deg", float('nan')))

                def draw_all(paused_flag: bool):
                    stdscr.erase()
                    h, w = stdscr.getmaxyx()
                    row = 1
                    # 指令（中/英）
                    cmds = [
                        ["空格：暂停/继续", "SPACE: pause/resume"],
                        ["n：单步前进", "n: next step"],
                        ["q：退出", "q: quit"],
                        ["方向含义：", "Mapping: L/R→shoulder_pan, U/D→wrist_flex"],
                    ]
                    col_w = max(20, w // 2)
                    for cn, en in cmds:
                        if row >= h - 5:
                            break
                        try:
                            stdscr.addstr(row, 1, cn[: max(0, col_w - 1)])
                            if col_w < w:
                                stdscr.addstr(row, 1 + col_w, en[: max(0, w - col_w - 1)])
                        except Exception:
                            pass
                        row += 1
                    # 状态
                    idx, total, act, lr_t, lr_d, ud_t, ud_d = get_status_text()
                    lines = [
                        f"状态: {'暂停' if paused_flag else '运行'} | 动作 {idx}/{total} -> {act}",
                        f"关节: LR ticks/deg = {lr_t} / {lr_d:.2f} | UD ticks/deg = {ud_t} / {ud_d:.2f}",
                    ]
                    for ln in lines:
                        if row >= h - 3:
                            break
                        try:
                            stdscr.addstr(row, 1, ln[: max(0, w - 1)])
                        except Exception:
                            pass
                        row += 1
                    # 概率
                    if args.show_probs and last_probs is not None and row < h - 2:
                        parts = []
                        for i2 in range(len(label_names)):
                            p = float(last_probs[i2]) if i2 < len(last_probs) else 0.0
                            parts.append(f"{label_names[i2]}={p:.2f}")
                        prob_line = "Probs: " + "  ".join(parts)
                        try:
                            stdscr.addstr(row, 1, prob_line[: max(0, w - 1)])
                        except Exception:
                            pass
                        row += 1
                    stdscr.box()
                    stdscr.refresh()
                
                draw_all(paused)
                i = 0
                while True:
                    ch = stdscr.getch()
                    if ch in (ord('q'), ord('Q')):
                        break
                    if ch in (ord(' '),):
                        paused = not paused
                        draw_all(paused)
                        continue
                    if ch in (ord('n'), ord('N')) and i < len(actions):
                        run_step(actions[i])
                        i += 1
                        draw_all(paused)
                        continue
                    if not paused and i < len(actions):
                        run_step(actions[i])
                        i += 1
                        draw_all(paused)
                        continue
                    if i >= len(actions):
                        draw_all(paused)
                        time.sleep(max(0.05, args.tui_interval))
                        continue
            import curses
            curses.wrapper(tui)
        except Exception as e:
            print(f"TUI 出错：{e}")
    else:
        # 非 TUI 简单运行
        for t, a in enumerate(actions):
            obs, reward, terminated, truncated = run_step(a)
            if terminated or truncated:
                break

    # 保存预测
    if args.save_preds is not None and len(pred_rows) > 0:
        try:
            import csv
            args.save_preds.parent.mkdir(parents=True, exist_ok=True)
            headers = []
            for r in pred_rows:
                for k in r.keys():
                    if k not in headers:
                        headers.append(k)
            with open(args.save_preds, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(pred_rows)
            print(f"Saved predictions to: {args.save_preds}")
        except Exception as e:
            print(f"Failed to save predictions: {e}")

    # 演示结束后归位（可选）
    if args.post_home:
        try:
            bus = So101Bus(args.serial_port, args.serial_baud, timeout=args.serial_timeout, debug=False)
            bus.open()
            mapping = So101Map.default()
            # 读取 JSON 覆盖与夹爪闭合位（与 pre-home 相同逻辑）
            home_overrides: dict[int, int] = {}
            gripper_closed_from_json: int | None = None
            print(f"演示结束归位… (json={str(post_json_resolved) if post_json_resolved else 'limits'})")
            if post_json_resolved is not None:
                try:
                    import json
                    with open(post_json_resolved, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    src = data.get('mids', data)
                    for k, v in src.items():
                        try:
                            home_overrides[int(k)] = int(v)
                        except Exception:
                            pass
                    if 'gripper_closed_ticks' in data:
                        try:
                            gripper_closed_from_json = int(data['gripper_closed_ticks'])
                        except Exception:
                            gripper_closed_from_json = None
                except Exception:
                    pass
            # 归位目标时间/等待
            post_time = int(args.post_home_time) if args.post_home_time is not None else int(args.home_time)
            post_wait = float(args.post_home_wait) if args.post_home_wait is not None else float(args.home_wait)
            # 夹爪行为
            post_grip_mode = args.post_gripper_home if args.post_gripper_home is not None else args.gripper_home
            post_grip_closed = args.post_gripper_closed_ticks
            # 逐关节归位
            mins: dict[int, int] = {}
            maxs: dict[int, int] = {}
            moved2 = []
            for name, jid in mapping.name_to_id.items():
                try:
                    # 基本设置
                    try:
                        bus.set_operating_mode(jid, 0)
                        bus.set_return_delay(jid, 0)
                        bus.torque_enable(jid, True)
                    except Exception:
                        pass
                    # 限位与中位
                    bmin = bus.read(jid, bus.MIN_POSITION_LIMIT, 2)
                    bmax = bus.read(jid, bus.MAX_POSITION_LIMIT, 2)
                    mn = int(bmin[0]) | (int(bmin[1]) << 8)
                    mx = int(bmax[0]) | (int(bmax[1]) << 8)
                    if mn > mx:
                        mn, mx = mx, mn
                    mins[jid], maxs[jid] = mn, mx
                    # 选择归位目标来源
                    if args.post_home_source == "json" and jid in home_overrides:
                        mid = int(home_overrides[jid])
                    else:
                        mid = int((mn + mx) // 2)
                    # 目标选择
                    if name == 'gripper' or jid == 6:
                        if post_grip_mode == 'keep':
                            continue
                        elif post_grip_mode == 'mid':
                            tgt = mid
                        elif post_grip_mode == 'closed':
                            tgt = post_grip_closed
                            if tgt is None:
                                tgt = gripper_closed_from_json
                            if tgt is None:
                                tgt = mn
                        else:
                            continue
                    else:
                        tgt = mid
                    bus.write_position(jid, int(tgt), time_ms=post_time)
                    moved2.append((jid, name, int(tgt)))
                except Exception:
                    pass
            time.sleep(max(0.0, post_time / 1000.0 + post_wait))
            if moved2:
                print("归位命令已下发：")
                for jid, name, tgt in moved2:
                    print(f"  [{jid:02d}] {name:<16} -> {tgt}")
        except Exception as e:
            print(f"演示结束后归位失败：{e}")
        finally:
            try:
                bus.close()
            except Exception:
                pass

    env.close()


if __name__ == "__main__":
    main()
    # 自动加载默认中位/归位 JSON（若未显式指定且文件存在）
    try:
        if args.home_json is None:
            from pathlib import Path as _P
            _h = _P("serial_home.json")
            if _h.exists():
                args.home_json = _h
                print(f"[info] 自动使用中位JSON: {args.home_json}")
        if args.post_home and args.post_home_json is None:
            from pathlib import Path as _P
            _r = _P("serial_return.json")
            if _r.exists():
                args.post_home_json = _r
                print(f"[info] 自动使用归位JSON: {args.post_home_json}")
    except Exception:
        pass
