# 会议记录 (Meeting Notes)

---

## 2026-02-03 周会

### 教授问题与回答

#### Q1: 如何划分训练集、验证集和测试集？

**当前实现：**

| 评估模式 | 训练集 | 测试集 | 验证集 |
|----------|--------|--------|--------|
| **Subject-dependent** | 同一被试的 Session 1 (T 文件) | 同一被试的 Session 2 (E 文件) | ❌ 无 |
| **LOSO** | 8 个被试的所有数据 | 1 个被试的所有数据 | ❌ 无 |

**代码位置：** `utils.py` → `load_data_evaluate()`

```python
# Subject-dependent: 同一被试的两个 session
train_data = A01T.mat  # Session 1 (Training)
test_data = A01E.mat   # Session 2 (Evaluation)

# LOSO: 留一被试做测试
for subject in 1..9:
    if subject == target:
        test_data = subject_all_data
    else:
        train_data += subject_all_data
```

**问题：** 目前没有验证集 (Validation Set)！

**改进方案：**
```
┌─────────────────────────────────────────────────────┐
│ 新的划分策略 (建议)                                  │
├─────────────────────────────────────────────────────┤
│ 训练集 (Train):      70% of trials                  │
│ 验证集 (Validation): 10% of trials (超参调优)        │
│ 测试集 (Test):       20% of trials (最终评估)        │
│                                                     │
│ 或使用 K-Fold 交叉验证:                              │
│ - 5-Fold CV on training data                        │
│ - Hold-out test set for final evaluation            │
└─────────────────────────────────────────────────────┘
```

---

#### Q2: 测试结果太好，需要多次交叉测试

**教授建议：** 多做几次交叉验证，确认结果稳定性

**改进方案：**

1. **多次随机种子测试**
   ```python
   for seed in [42, 123, 456, 789, 1024]:
       train_model(seed=seed)
       results.append(evaluate())
   
   print(f"Mean: {np.mean(results):.2f}% ± {np.std(results):.2f}%")
   ```

2. **K-Fold 交叉验证**
   ```python
   from sklearn.model_selection import StratifiedKFold
   
   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   for train_idx, val_idx in skf.split(X, y):
       # 训练和验证
   ```

3. **跨被试 (Cross-Subject) 测试**
   - 对全部 9 个被试进行 LOSO 评估
   - 报告平均值和标准差

**TODO：** 实现 K-Fold 交叉验证，报告 mean ± std

---

#### Q3: 模拟机械臂还是物理机械臂？

**回答：** 两者都做

| 环境 | 文件 | 用途 |
|------|------|------|
| **模拟 (PyBullet)** | `gym_control.py`, `arm_gym_env.py` | 快速迭代、算法验证 |
| **物理 (SO-101)** | `phy_control.py` | 真实环境验证、最终演示 |

**工作流程：**
```
模拟环境训练 → 模拟环境测试 → 物理机械臂部署 → 物理环境测试
```

---

#### Q4: 系统需要多少 Channel？

**当前支持的数据集：**

| 数据集 | Channels | Classes | 采样率 | 描述 |
|--------|----------|---------|--------|------|
| **IV-2a** | 22 EEG + 3 EOG | 4 | 250 Hz | 左手、右手、双脚、舌头 |
| **IV-2b** | 3 (C3, Cz, C4) | 2 | 250 Hz | 左手、右手 |
| **GigaScience** | 64 | 11 | 1000 Hz | 11种直觉运动任务 |

**代码中的定义：**

```python
# gym_control.py / phy_control.py
def number_class_channel(dataset_type: str) -> Tuple[int, int]:
    if dataset_type == "A":
        return 4, 22  # IV-2a: 4 classes, 22 channels
    if dataset_type == "B":
        return 2, 3   # IV-2b: 2 classes, 3 channels
```

**系统设计：**
- 系统应该能够**自适应**不同数量的 channel
- 输入层维度由数据集自动确定
- CTNet 架构通过 `number_channel` 参数配置

---

### 会议后 Action Items

| 优先级 | 任务 | 状态 |
|--------|------|------|
| 🔴 高 | 添加验证集划分 | TODO |
| 🔴 高 | 实现 K-Fold 交叉验证 | TODO |
| 🔴 高 | 多随机种子测试，报告 mean ± std | TODO |
| 🟡 中 | Controller/Limiter: 关节限位保护 | TODO |
| 🟡 中 | Smoother + Delay: 减少高频颤动 | TODO |
| 🟢 低 | 双数据集对比 (IV-2b + GigaScience) | TODO |

---

### 下次会议准备

1. 展示 K-Fold 交叉验证结果
2. 报告多种子测试的 mean ± std
3. 演示模拟/物理机械臂控制
4. 准备不同 channel 数据集的对比

---

*记录时间：2026-02-03*

---

## 2026-02-17 周会

### 会议内容

#### 导师反馈 1: Position vs Time 图

**要求：** 添加一个 Position vs Time 图，用于展示系统的 overall performance

**具体内容：**
- 用**两种颜色**分别表示：
  - 🔵 **蓝色虚线**：目标位置（应该在哪里）
  - 🔴 **红色实线**：实际位置（实际在哪里）
- 通过两个位置的差异计算 **误差 (Error)**
- 用误差曲线展示系统的整体控制性能

**实现方案：**

```
┌─────────────────────────────────────────────────────────────┐
│  Position vs Time 图设计                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  子图 1: Y 位置 (左右)                                       │
│  ┌───────────────────────────────────────┐                  │
│  │ --- Target Y (蓝色虚线)               │                  │
│  │ ─── Actual Y (红色实线)               │                  │
│  │ ███ Error 区域 (橙色填充)             │                  │
│  └───────────────────────────────────────┘                  │
│                                                             │
│  子图 2: Z 位置 (上下)                                       │
│  ┌───────────────────────────────────────┐                  │
│  │ --- Target Z (蓝色虚线)               │                  │
│  │ ─── Actual Z (红色实线)               │                  │
│  │ ███ Error 区域 (橙色填充)             │                  │
│  └───────────────────────────────────────┘                  │
│                                                             │
│  子图 3: 位置误差曲线                                        │
│  ┌───────────────────────────────────────┐                  │
│  │ ─── Position Error (绿色)             │                  │
│  │ --- Target Threshold (红色虚线)       │                  │
│  │ 显示: Final Error, Mean Error         │                  │
│  └───────────────────────────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**代码修改：** `scripts/rl_control_test.py`
- 新增 `visualize_position_vs_time()` 函数
- 新增 `visualize_overall_performance()` 函数
- 修改 `run_rl_control_episode()` 添加轨迹记录功能
- 修改 `RLControlResult` 添加 `sample_trajectory` 字段

**输出文件：**
- `outputs/rl_control_test/position_vs_time.png`
- `outputs/rl_control_test/overall_performance.png`

---

#### 导师反馈 2: 继续之前计划的未完成任务

**待完成任务列表：**

| 优先级 | 任务 | 状态 |
|--------|------|------|
| 🔴 高 | Position vs Time 图 | ✅ 已实现 |
| 🟡 中 | Controller/Limiter: 关节限位保护 | 🔄 待完成 |
| 🟡 中 | Smoother + Delay: 减少高频颤动 | 🔄 待完成 |
| 🟢 低 | 扩展动作空间：张开/闭合等 | 🔄 待完成 |

---

### 会议后 Action Items

| 优先级 | 任务 | 状态 |
|--------|------|------|
| ✅ 完成 | 添加 Position vs Time 图 | Done |
| ✅ 完成 | 物理机械臂 RL 控制 | Done |
| 🔄 进行中 | Controller/Limiter: 关节限位保护 | TODO |
| 🔄 进行中 | Smoother + Delay: 减少高频颤动 | TODO |
| 🔄 待定 | 扩展动作空间 | TODO |

---

*记录时间：2026-02-17*

---

## 2026-02-17 物理机械臂控制成功 🎉

### 里程碑

**首次实现 BCI → RL → 物理机械臂的完整控制链路！**

```
┌─────────────────────────────────────────────────────────────┐
│  BCI 控制系统完整流程                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 EEG 信号 → 🧠 CTNet 分类 → 🎯 意图识别 → 🤖 RL 决策 → 🦾 机械臂运动  │
│                                                             │
│  IV-2a 数据集 (22 通道, 4 类)                                 │
│  ↓                                                          │
│  CTNet 分类: left / right / up / down                       │
│  ↓                                                          │
│  RL Agent (TransformerDQN): 选择最优动作序列                  │
│  ↓                                                          │
│  SO-101 物理机械臂: 实际运动到目标位置                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 测试结果

| 指标 | 结果 |
|------|------|
| **EEG 分类准确率** | 100% (8/8) ✅ |
| **RL 控制到达率** | 100% (8/8) ✅ |
| **平均步数** | 8.2 步 |
| **平均奖励** | 10.31 |

### 4 方向控制能力

| 方向 | 物理控制 | 说明 |
|------|----------|------|
| **left** | ✅ 成功 | shoulder_pan 向左 |
| **right** | ✅ 成功 | shoulder_pan 向右 |
| **up** | ✅ 成功 | wrist_flex 向上 |
| **down** | ✅ 成功 | wrist_flex 向下 |

### 关键改进

1. **重新训练 4 方向 RL 模型**
   - 之前模型只会 "down" 动作
   - 新模型 `rl_4direction_policy.pth` 4 个方向都能正确控制

2. **修复 PyTorch 2.6 兼容性问题**
   - `torch.load()` 添加 `weights_only=False`
   - `numpy.bool_` 转换为 Python `bool` 后 JSON 序列化

3. **串口权限处理**
   - `sudo chmod 666 /dev/ttyACM1` 临时解决
   - `sudo usermod -aG dialout $USER` 永久解决

### 相关文件

| 文件 | 功能 |
|------|------|
| `scripts/train_rl_4direction.py` | 4 方向 RL 训练脚本 |
| `scripts/rl_physical_control.py` | 物理机械臂控制脚本 |
| `outputs/rl_4direction_policy.pth` | 训练好的 RL 模型 |
| `outputs/rl_physical_control/results.json` | 物理控制测试结果 |

### 下一步

| 优先级 | 任务 |
|--------|------|
| 🔴 高 | Controller/Limiter: 关节限位保护 |
| 🔴 高 | Smoother + Delay: 减少高频颤动 |
| 🟡 中 | 扩展动作空间 (张开/闭合) |
| 🟢 低 | 多被试测试验证泛化性 |

---

*记录时间：2026-02-17 (下午)*

