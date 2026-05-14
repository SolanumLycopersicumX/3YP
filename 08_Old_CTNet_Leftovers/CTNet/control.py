import os
import numpy as np
import pandas as pd
import random
import datetime
import time
import matplotlib.pyplot as plt  # ### NEW: VIS

from pandas import ExcelWriter
from torchsummary import summary
import torch
import math
import warnings
warnings.filterwarnings("ignore")

from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

from utils import calMetrics
from utils import calculatePerClass
from utils import numberClassChannel
from utils import load_data_evaluate

from torch.autograd import Variable

from CTNet_model import EEGTransformer, BranchEEGNetTransformer, PatchEmbeddingCNN, PositioinalEncoding, TransformerEncoder, TransformerEncoderBlock, MultiHeadAttention, FeedForwardBlock, ResidualAdd, ClassificationHead
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import robot_startup, robot_shutdown


# === Settings ===
subject_id = 2  # subject index 1–9
data_dir = './mymat_raw/'
model_path = f'./models/new/model_{subject_id}.pth'  # change if your path differs
dataset_type = 'A'
evaluate_mode = 'subject-dependent'
device = torch.device('cpu')  # Or 'cuda' if using GPU

# === Load and Normalize Test Data ===
_, _, test_data, test_labels = load_data_evaluate(data_dir, dataset_type, subject_id, mode_evaluate=evaluate_mode)
test_data = np.expand_dims(test_data, axis=1)
mean = np.mean(test_data)
std = np.std(test_data) if np.std(test_data) > 0 else 1.0
test_data = (test_data - mean) / std
test_tensor = torch.tensor(test_data, dtype=torch.float32)

# Make labels 1D vector; shift to 0..3 if needed
test_labels = np.asarray(test_labels).reshape(-1)
# Some pipelines store labels in shape (N,1) with values 1..4
if test_labels.min() == 1:
    test_labels = test_labels - 1  # now 0..3

# === Load Model ===
number_class, number_channel = numberClassChannel(dataset_type)
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
    eeg1_number_channel=number_channel,
    flatten_eeg1=240
).to(device)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict.state_dict() if hasattr(state_dict, "state_dict") else state_dict)
model.eval()

# === Initialize Robot ===
bot = InterbotixManipulatorXS(robot_model='px150', group_name='arm', gripper_name='gripper')
robot_startup()
bot.arm.go_to_home_pose()

# === Movement Step Parameters ===
step_size = 0.03
y = 0.0
z = 0.0

# ### NEW: VIS — paths (cumulative positions) & displacements
pred_y_path = [0.0]
pred_z_path = [0.0]
true_y_path = [0.0]
true_z_path = [0.0]

pred_steps = []
true_steps = []

# === Run Real-Time Simulation Trial-by-Trial ===
trial_times = []
num_trials = min(10, len(test_tensor))
for i in range(num_trials):
    x_input = test_tensor[i].unsqueeze(0).to(device)  # shape: [1, 1, 22, 1000]

    # Start timing
    start_time = time.time()

    # Predict class
    with torch.no_grad():
        # CTNet returns (features, logits); adjust if yours differs
        _, output = model(x_input)
        pred = torch.argmax(output, dim=1).item()

    # Map class -> displacement (predicted)
    dy_pred, dz_pred = 0.0, 0.0
    if pred == 0: dy_pred = -step_size   # left
    elif pred == 1: dy_pred = step_size  # right
    elif pred == 2: dz_pred = step_size  # up
    elif pred == 3: dz_pred = -step_size # down

    # Compute "true" displacement from label
    true_cls = int(test_labels[i])
    dy_true, dz_true = 0.0, 0.0
    if true_cls == 0: dy_true = -step_size
    elif true_cls == 1: dy_true = step_size
    elif true_cls == 2: dz_true = step_size
    elif true_cls == 3: dz_true = -step_size

    # Accumulate predicted path (this is what you actually send to the robot)
    y += dy_pred
    z += dz_pred
    pred_y_path.append(y)
    pred_z_path.append(z)

    # Accumulate the hypothetical true path (what it would be if you followed ground truth)
    true_y_path.append(true_y_path[-1] + dy_true)
    true_z_path.append(true_z_path[-1] + dz_true)

    # Stash step vectors for optional quiver/arrows
    pred_steps.append((dy_pred, dz_pred))
    true_steps.append((dy_true, dz_true))

    # Send command to robot (predicted motion)
    bot.arm.set_ee_cartesian_trajectory(z=dz_pred)
    # belt joint change: scaled by 10π/2 (~rad) — keep as you had
    bot.arm.set_single_joint_position(joint_name='waist', position=y * 10 * np.pi / 2.0)

    # End timing
    end_time = time.time()
    elapsed = end_time - start_time
    trial_times.append(elapsed)
    print(f"[Trial {i+1}] Pred Class: {pred} | True: {true_cls} | "
          f"Δy/Δz pred=({dy_pred:.2f},{dz_pred:.2f}) true=({dy_true:.2f},{dz_true:.2f}) | "
          f"y={y:.2f}, z={z:.2f} | Time: {elapsed:.3f}s")

    time.sleep(2)

# === Wrap Up Robot ===
bot.arm.go_to_sleep_pose()
robot_shutdown()

# === Performance Summary ===
print("\n--- Runtime per Trial ---")
for i, t in enumerate(trial_times):
    print(f"Trial {i+1}: {t:.3f} seconds")
print(f"\nAverage time per trial: {np.mean(trial_times):.3f} seconds")

# === SIMPLE LINE CHARTS ===
import numpy as np
import matplotlib.pyplot as plt

# Convert paths to arrays
pred_y_arr = np.array(pred_y_path)   # length = num_trials + 1 (starts at 0)
pred_z_arr = np.array(pred_z_path)
true_y_arr = np.array(true_y_path)
true_z_arr = np.array(true_z_path)

# Per-trial deltas (length = num_trials)
pred_dy = np.array([dy for dy, _ in pred_steps])
pred_dz = np.array([dz for _, dz in pred_steps])
true_dy = np.array([dy for dy, _ in true_steps])
true_dz = np.array([dz for _, dz in true_steps])

# X axes
pos_x = np.arange(len(pred_y_arr))          # 0..num_trials (positions)
diff_x = np.arange(1, len(pred_y_arr))      # 1..num_trials (per-trial deltas)

fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=False)

# 1) Y Movement (positions)
axs[0].plot(pos_x, pred_y_arr, label='Predicted Y', marker='o')
axs[0].plot(pos_x, true_y_arr, label='True Y', marker='x')
axs[0].set_title('Y Movement')
axs[0].set_ylabel('Y position')
axs[0].grid(True, alpha=0.3)
axs[0].legend()

# 2) Z Movement (positions)
axs[1].plot(pos_x, pred_z_arr, label='Predicted Z', marker='o')
axs[1].plot(pos_x, true_z_arr, label='True Z', marker='x')
axs[1].set_title('Z Movement')
axs[1].set_ylabel('Z position')
axs[1].grid(True, alpha=0.3)
axs[1].legend()

# 3) Y Movement Difference (per-trial ΔY)
axs[2].plot(diff_x, pred_dy, label='Predicted ΔY', marker='o')
axs[2].plot(diff_x, true_dy, label='True ΔY', marker='x')
axs[2].set_title('Y Movement Difference')
axs[2].set_ylabel('ΔY per trial')
axs[2].grid(True, alpha=0.3)
axs[2].legend()

# 4) Z Movement Difference (per-trial ΔZ)
axs[3].plot(diff_x, pred_dz, label='Predicted ΔZ', marker='o')
axs[3].plot(diff_x, true_dz, label='True ΔZ', marker='x')
axs[3].set_title('Z Movement Difference')
axs[3].set_xlabel('Trial')
axs[3].set_ylabel('ΔZ per trial')
axs[3].grid(True, alpha=0.3)
axs[3].legend()

plt.tight_layout()
# plt.savefig('movement_compare_lines.png', dpi=150)
plt.show()
