#!/usr/bin/env python3
"""
CTNet + DQN End-to-End Offline Evaluation & EEG-aware RL Training

This script:
  1. Loads the trained CTNet model
  2. Classifies PhysioNet test subjects to produce EEG predictions
  3. Trains DQN agents WITH and WITHOUT EEG predictions in the state
  4. Evaluates both agents on control performance (reach rate, reward)
  5. Saves comprehensive results

Usage:
  python scripts/ctnet_dqn_e2e_eval.py
"""

from __future__ import annotations
import argparse, copy, json, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from scripts.dqn_transformer import create_dqn_network
from scripts.physionet_loader import load_subject_4class_mne, preprocess_data

TARGET_SAMPLES = 1000
CLASS_NAMES = ["Left", "Right", "Hands/Up", "Feet/Down"]
DIRECTION_MAP = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}


# ============================================================================
# Phase 1: CTNet classification → predictions
# ============================================================================

def resample_epochs(data: np.ndarray, target_len: int = TARGET_SAMPLES) -> np.ndarray:
    n_epochs, n_ch, n_times = data.shape
    if n_times == target_len:
        return data
    out = np.zeros((n_epochs, n_ch, target_len), dtype=np.float32)
    for i in range(n_epochs):
        out[i] = resample(data[i], target_len, axis=1)
    return out


def load_ctnet_model(model_path: Path, device: torch.device):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    meta_path = model_path.with_suffix(".json")
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return model, meta


def classify_subjects(
    model, meta: dict, subjects: List[int], device: torch.device
) -> Dict[str, np.ndarray]:
    norm_mean = meta.get("norm_mean", 0.0)
    norm_std = meta.get("norm_std", 1.0)

    all_preds, all_confs, all_labels = [], [], []

    for sub in subjects:
        print(f"  Classifying S{sub:03d}...", end=" ", flush=True)
        try:
            data, labels = load_subject_4class_mne(sub)
        except Exception as e:
            print(f"SKIP ({e})")
            continue
        data = preprocess_data(data)
        data = resample_epochs(data, TARGET_SAMPLES)
        data = (data - norm_mean) / (norm_std + 1e-12)

        X = torch.from_numpy(np.expand_dims(data, 1)).float().to(device)
        with torch.no_grad():
            _, logits = model(X)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1).cpu().numpy()
        confs = probs.max(dim=-1).values.cpu().numpy()

        acc = (preds == labels).mean() * 100
        print(f"acc={acc:.1f}%  ({len(labels)} trials)")

        all_preds.append(preds)
        all_confs.append(confs)
        all_labels.append(labels)

    return {
        "predictions": np.concatenate(all_preds),
        "confidences": np.concatenate(all_confs),
        "labels": np.concatenate(all_labels),
    }


# ============================================================================
# Phase 2: RL Environment
# ============================================================================

@dataclass
class RLArmConfig:
    step_size: float = 0.05
    target_radius: float = 0.1
    max_steps: int = 100
    reward_reach: float = 10.0
    reward_step: float = -0.01
    reward_distance_scale: float = 1.0
    reward_oscillation: float = -0.5
    reward_boundary: float = -0.5
    boundary: float = 1.0


class RLArm2DEnv:
    def __init__(self, config: Optional[RLArmConfig] = None):
        self.cfg = config or RLArmConfig()
        self.action_space_n = 4
        self.observation_dim = 5
        self._rng = np.random.RandomState(42)
        self._reset_state()

    def _reset_state(self):
        self.pos = np.array([0.0, 0.0])
        self.target = np.array([0.0, 0.0])
        self.steps = 0
        self.prev_dist = 0.0
        self.action_history = []

    def reset(self, seed=None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.pos = self._rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
        self.target = self._rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
        while np.linalg.norm(self.pos - self.target) < 0.3:
            self.target = self._rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
        self.steps = 0
        self.prev_dist = np.linalg.norm(self.pos - self.target)
        self.action_history = []
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        dist = np.linalg.norm(self.pos - self.target)
        return np.array(
            [self.pos[0], self.pos[1], self.target[0], self.target[1], dist],
            dtype=np.float32,
        )

    def step(self, action: int):
        directions = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}
        dy, dz = directions[action]
        self.pos[0] = np.clip(self.pos[0] + dy * self.cfg.step_size, -self.cfg.boundary, self.cfg.boundary)
        self.pos[1] = np.clip(self.pos[1] + dz * self.cfg.step_size, -self.cfg.boundary, self.cfg.boundary)
        self.steps += 1
        self.action_history.append(action)

        dist = np.linalg.norm(self.pos - self.target)
        reward = self.cfg.reward_step
        reward += (self.prev_dist - dist) * self.cfg.reward_distance_scale

        if self._detect_oscillation():
            reward += self.cfg.reward_oscillation
        at_boundary = np.any(np.abs(self.pos) >= self.cfg.boundary - 1e-6)
        if at_boundary:
            reward += self.cfg.reward_boundary

        reached = dist < self.cfg.target_radius
        done = reached or self.steps >= self.cfg.max_steps
        if reached:
            reward += self.cfg.reward_reach

        self.prev_dist = dist
        info = {"reached": reached, "distance": dist, "steps": self.steps}
        return self._get_obs(), reward, done, info

    def _detect_oscillation(self) -> bool:
        if len(self.action_history) < 3:
            return False
        a, b, c = self.action_history[-3:]
        return a == c and a != b


class EEGActionEnv:
    """Wraps RLArm2DEnv with EEG classification predictions in the state."""

    def __init__(self, base_env: RLArm2DEnv, eeg_predictions: np.ndarray,
                 eeg_confidences: Optional[np.ndarray] = None):
        self.base_env = base_env
        self.eeg_predictions = eeg_predictions
        self.eeg_confidences = eeg_confidences
        self._eeg_idx = 0
        self.observation_dim = base_env.observation_dim + 2
        self.action_space_n = base_env.action_space_n

    def reset(self, seed=None) -> np.ndarray:
        base_obs = self.base_env.reset(seed)
        self._eeg_idx = np.random.randint(0, len(self.eeg_predictions))
        return self._augment_obs(base_obs)

    def _augment_obs(self, base_obs: np.ndarray) -> np.ndarray:
        idx = self._eeg_idx % len(self.eeg_predictions)
        eeg_action = float(self.eeg_predictions[idx])
        confidence = float(self.eeg_confidences[idx]) if self.eeg_confidences is not None else 1.0
        return np.concatenate([base_obs, [eeg_action / 3.0, confidence]]).astype(np.float32)

    def step(self, action: int):
        obs, reward, done, info = self.base_env.step(action)
        self._eeg_idx = (self._eeg_idx + 1) % len(self.eeg_predictions)
        return self._augment_obs(obs), reward, done, info


# ============================================================================
# Phase 3: DQN Training (Double DQN with soft update)
# ============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.states = torch.zeros((capacity, 1, state_dim), device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.next_states = torch.zeros((capacity, 1, state_dim), device=device)
        self.dones = torch.zeros(capacity, device=device)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, ns, d):
        idx = self.ptr % self.capacity
        self.states[idx] = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        self.actions[idx] = int(a)
        self.rewards[idx] = float(r)
        self.next_states[idx] = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        self.dones[idx] = float(d)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        t = torch.as_tensor(idxs, device=self.device)
        return self.states[t], self.actions[t], self.rewards[t], self.next_states[t], self.dones[t]


def train_dqn(
    env,
    network_type: str,
    state_dim: int,
    action_dim: int = 4,
    episodes: int = 1000,
    device: torch.device = torch.device("cpu"),
    label: str = "",
) -> Dict:
    policy = create_dqn_network(network_type, state_dim=state_dim, action_dim=action_dim, seq_len=1).to(device)
    target = copy.deepcopy(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    buf = ReplayBuffer(100_000, state_dim, device)

    tau = 0.005
    gamma = 0.99
    eps = 1.0
    eps_end = 0.05
    eps_decay_steps = min(1500, int(episodes * 0.8))
    batch_size = 64
    min_buf = 500

    logs = {"reward": [], "reach": [], "eps": []}
    best_reach = 0.0
    best_weights = None

    t0 = time.time()
    for ep in range(episodes):
        obs = env.reset(seed=ep)
        state = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        total_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():
                q = policy(state)
            if np.random.rand() < eps:
                action = np.random.randint(action_dim)
            else:
                action = int(q.argmax().item())

            next_obs, reward, done, info = env.step(action)
            next_state = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            buf.add(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            if buf.size >= min_buf:
                s, a, r, ns, d = buf.sample(batch_size)
                with torch.no_grad():
                    best_a = policy(ns).argmax(dim=-1)
                    target_q = target(ns).gather(1, best_a.unsqueeze(1)).squeeze(1)
                    y = r + (1 - d) * gamma * target_q
                current_q = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(current_q, y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()

                for pp, tp in zip(policy.parameters(), target.parameters()):
                    tp.data.copy_(tau * pp.data + (1 - tau) * tp.data)

        eps = max(eps_end, eps - (1.0 - eps_end) / eps_decay_steps)
        logs["reward"].append(total_reward)
        logs["reach"].append(1 if info.get("reached", False) else 0)
        logs["eps"].append(eps)

        reach_window = np.mean(logs["reach"][-100:]) * 100
        if reach_window > best_reach:
            best_reach = reach_window
            best_weights = copy.deepcopy(policy.state_dict())

        if (ep + 1) % 200 == 0:
            avg_r = np.mean(logs["reward"][-100:])
            print(f"  [{label}] ep {ep+1}/{episodes}  reward={avg_r:.2f}  reach={reach_window:.0f}%  eps={eps:.3f}")

    if best_weights is not None:
        policy.load_state_dict(best_weights)
    elapsed = time.time() - t0
    return {"policy": policy, "logs": logs, "best_reach": best_reach, "time": elapsed}


# ============================================================================
# Phase 4: Evaluation
# ============================================================================

def evaluate_agent(policy, env, n_episodes: int = 200, device: torch.device = torch.device("cpu")) -> Dict:
    reaches = []
    rewards = []
    steps_list = []

    for ep in range(n_episodes):
        obs = env.reset(seed=10000 + ep)
        state = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        total_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():
                q = policy(state)
            action = int(q.argmax().item())
            next_obs, reward, done, info = env.step(action)
            state = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            total_reward += reward

        reaches.append(1 if info.get("reached", False) else 0)
        rewards.append(total_reward)
        steps_list.append(info.get("steps", 0))

    return {
        "reach_rate": np.mean(reaches) * 100,
        "mean_reward": np.mean(rewards),
        "mean_steps": np.mean(steps_list),
        "std_reward": np.std(rewards),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CTNet + DQN End-to-End Evaluation")
    parser.add_argument("--model", type=Path,
                        default=_ROOT / "outputs/physionet_ctnet_109sub_v2/pool/model_pool.pth")
    parser.add_argument("--subjects", type=int, nargs="+", default=[7, 48, 3, 9, 70, 50, 38])
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=_ROOT / "outputs/ctnet_dqn_e2e")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Step 1: CTNet Classification
    # ----------------------------------------------------------------
    print("=" * 60)
    print("Phase 1: CTNet Classification on PhysioNet test subjects")
    print("=" * 60)
    model, meta = load_ctnet_model(args.model, device)
    eeg_results = classify_subjects(model, meta, args.subjects, device)

    preds = eeg_results["predictions"]
    confs = eeg_results["confidences"]
    labels = eeg_results["labels"]
    overall_acc = (preds == labels).mean() * 100
    print(f"\nOverall classification accuracy: {overall_acc:.2f}%")
    print(f"Total trials: {len(preds)}")
    print(f"Mean confidence: {confs.mean():.3f}")

    np.save(args.output / "eeg_predictions.npy", preds)
    np.save(args.output / "eeg_confidences.npy", confs)
    np.save(args.output / "eeg_labels.npy", labels)

    # ----------------------------------------------------------------
    # Step 2: Train DQN agents (with vs. without EEG)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2: EEG-aware RL Training (with vs. without EEG)")
    print("=" * 60)

    base_env_no_eeg = RLArm2DEnv()
    base_env_with_eeg = RLArm2DEnv()
    eeg_env = EEGActionEnv(base_env_with_eeg, preds, confs)

    net_type = "transformer"
    results = {}

    print(f"\n--- Training: Transformer DQN WITHOUT EEG info ---")
    res_no_eeg = train_dqn(
        base_env_no_eeg, net_type, state_dim=5, episodes=args.episodes,
        device=device, label="No-EEG",
    )
    results["no_eeg"] = res_no_eeg

    print(f"\n--- Training: Transformer DQN WITH EEG predictions ---")
    res_with_eeg = train_dqn(
        eeg_env, net_type, state_dim=7, episodes=args.episodes,
        device=device, label="With-EEG",
    )
    results["with_eeg"] = res_with_eeg

    # ----------------------------------------------------------------
    # Step 3: Evaluate both agents
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 3: Control Evaluation (300 episodes)")
    print("=" * 60)

    eval_env_no_eeg = RLArm2DEnv()
    eval_env_with_eeg = EEGActionEnv(RLArm2DEnv(), preds, confs)

    eval_no = evaluate_agent(res_no_eeg["policy"], eval_env_no_eeg, args.eval_episodes, device)
    eval_with = evaluate_agent(res_with_eeg["policy"], eval_env_with_eeg, args.eval_episodes, device)

    print(f"\n{'Metric':<25} {'Without EEG':>15} {'With EEG':>15}")
    print("-" * 55)
    print(f"{'Reach Rate (%)':<25} {eval_no['reach_rate']:>15.1f} {eval_with['reach_rate']:>15.1f}")
    print(f"{'Mean Reward':<25} {eval_no['mean_reward']:>15.2f} {eval_with['mean_reward']:>15.2f}")
    print(f"{'Mean Steps':<25} {eval_no['mean_steps']:>15.1f} {eval_with['mean_steps']:>15.1f}")
    print(f"{'Training Time (s)':<25} {res_no_eeg['time']:>15.1f} {res_with_eeg['time']:>15.1f}")

    # ----------------------------------------------------------------
    # Step 4: Save results
    # ----------------------------------------------------------------
    summary = {
        "classification": {
            "model": str(args.model),
            "subjects": args.subjects,
            "overall_accuracy": round(overall_acc, 2),
            "total_trials": int(len(preds)),
            "mean_confidence": round(float(confs.mean()), 4),
            "per_class_accuracy": {},
        },
        "rl_comparison": {
            "training_episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "network_type": net_type,
            "without_eeg": {
                "state_dim": 5,
                "reach_rate": round(eval_no["reach_rate"], 2),
                "mean_reward": round(eval_no["mean_reward"], 2),
                "mean_steps": round(eval_no["mean_steps"], 1),
                "training_time_s": round(res_no_eeg["time"], 1),
                "best_train_reach": round(res_no_eeg["best_reach"], 1),
            },
            "with_eeg": {
                "state_dim": 7,
                "eeg_accuracy_pct": round(overall_acc, 2),
                "reach_rate": round(eval_with["reach_rate"], 2),
                "mean_reward": round(eval_with["mean_reward"], 2),
                "mean_steps": round(eval_with["mean_steps"], 1),
                "training_time_s": round(res_with_eeg["time"], 1),
                "best_train_reach": round(res_with_eeg["best_reach"], 1),
            },
        },
    }

    for c in range(4):
        mask = labels == c
        if mask.sum() > 0:
            acc_c = (preds[mask] == labels[mask]).mean() * 100
            summary["classification"]["per_class_accuracy"][CLASS_NAMES[c]] = round(acc_c, 2)

    def _to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        return obj

    with open(args.output / "e2e_summary.json", "w") as f:
        json.dump(_to_native(summary), f, indent=2)

    torch.save(res_no_eeg["policy"].state_dict(), args.output / "dqn_no_eeg.pth")
    torch.save(res_with_eeg["policy"].state_dict(), args.output / "dqn_with_eeg.pth")

    # Training curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        w = 50
        for label_name, logs in [("No EEG", res_no_eeg["logs"]), ("With EEG", res_with_eeg["logs"])]:
            r = np.convolve(logs["reward"], np.ones(w)/w, mode="valid")
            axes[0].plot(r, label=label_name)
            reach = np.convolve(logs["reach"], np.ones(w)/w, mode="valid") * 100
            axes[1].plot(reach, label=label_name)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward (moving avg)")
        axes[0].set_title("Training Reward")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Reach Rate %")
        axes[1].set_title("Training Reach Rate")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.output / "training_comparison.png", dpi=150)
        plt.close()
        print(f"\nPlot saved: {args.output / 'training_comparison.png'}")
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print(f"\nResults saved to: {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
