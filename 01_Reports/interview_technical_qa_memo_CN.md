# 面试技术问答备忘录

这份文档用于面试前快速复习，重点覆盖 EEGTransformer + Transformer DQN + 机械臂闭环控制项目中容易被追问的技术细节。

## 0. 一分钟技术概述

我做的是一个 BCI 到机械臂控制的闭环系统。EEG 侧使用 EEGTransformer / CTNet 风格的 CNN-Transformer 网络，把运动想象 EEG 分类成离散意图；控制侧使用 Transformer DQN，根据机械臂当前状态、目标状态和距离输出离散动作的 Q-value；动作再通过 PyBullet 仿真环境或 SO-101 串口控制接口执行。

可以背这一句：

> EEGTransformer 负责从噪声 EEG 中识别运动想象意图，Transformer DQN 负责根据状态反馈选择闭环控制动作，PyBullet / OpenBCI / SO-101 串口接口负责仿真和硬件控制链路验证。

## 1. 必须记住的关键数字

| 问题 | 答案 |
|---|---|
| 基础 RL 输入维度 | 5 |
| EEG-aware RL 输入维度 | 7 |
| 4 方向 RL 输出维度 | 4 |
| 8 方向 RL 输出维度 | 8 |
| 基础 RL state | `[y, z, target_y, target_z, distance]` |
| EEG-aware RL state | `[y, z, target_y, target_z, distance, eeg_pred, confidence]` |
| 4 方向动作 | left, right, up, down |
| 8 方向动作 | left, right, up, down, up_left, up_right, down_left, down_right |
| DQN 输出 | 每个离散动作的 Q-value，不是坐标或关节角 |
| 执行动作选择 | `argmax_a Q(s, a)` |
| RL 算法 | Double DQN + target network / soft update |
| 折扣因子 | `gamma = 0.99` |
| soft update 系数 | `tau = 0.005` |
| replay buffer size | 100,000 |
| 基础 RL 步长 | 0.05，归一化 2D 平面 |
| E2E 到达半径 | 0.1 |
| 8 方向 smooth 版本到达半径 | 0.15 |
| 归一化工作空间 | 每个轴 `[-1, 1]` |
| EEG 分类输出 | 4 类 logits / probabilities |
| PhysioNet EEG 输入形状 | 通常是 `[B, 1, 64, 1000]` |
| EEGTransformer `d_model` | 64 |
| EEGTransformer attention heads | 8 |
| EEGTransformer encoder layers | 2 |
| EEGTransformer FFN dim | 256 |
| EEGTransformer patch size | 25 samples |
| PhysioNet 统一采样长度 | 1000 |
| SO-101 结构 | 5 个机械臂关节 + 1 个夹爪舵机 |
| 项目里的实际控制抽象 | 2D 末端运动 |
| 主要控制关节 | `shoulder_pan` 控左右，`wrist_flex` 控上下 |
| 串口波特率 | 1,000,000 |

## 2. 系统架构问题

### Q: 你到底做了什么？

我做了一个 EEG 到机械臂目标到达的闭环控制系统，分成三层：

1. EEGTransformer 把 EEG 运动想象 trial 分类成离散意图。
2. Transformer DQN 根据当前状态和目标状态选择控制动作。
3. PyBullet / SO-101 串口控制接口执行动作，并把新的状态反馈给 RL。

### Q: 整个系统的输入和输出是什么？

输入：

- EEG epoch / EEG trial。
- 机械臂当前状态和目标状态。

输出：

- 离散控制动作，例如 left、right、up、down。
- 底层控制器把离散动作转换成仿真末端位移或物理关节增量。

### Q: EEG 是不是直接控制机械臂？

不是。EEGTransformer 只是给出高层意图预测，这个预测可能有噪声。真正执行动作的是 RL 控制器，它结合机械臂当前状态、目标状态和 EEG 预测，选择最终动作。

推荐回答：

> EEG 不是直接控制机械臂，而是作为 noisy high-level intent cue。Transformer DQN 再根据闭环状态反馈决定实际动作。

### Q: 为什么说这是闭环控制？

因为控制器每一步都会读取当前状态：

```text
当前位置 -> 目标位置 -> 距离 -> 选择动作 -> 执行动作 -> 更新状态 -> 下一步继续决策
```

它不是一次性把 EEG class 映射成动作，而是每一步都根据反馈纠正。

### Q: 你的核心技术贡献是什么？

核心不是单纯做 EEG 分类，而是把一个有噪声的 EEG 意图识别模块接到 RL 闭环控制器中，让系统在分类不完美的情况下仍然能完成目标到达。具体包括：

- EEG 分类模型实现。
- EEG label 到控制意图的映射。
- RL state / action / reward 设计。
- Transformer DQN 控制策略训练。
- 4 方向和 8 方向动作空间实验。
- PyBullet 仿真和 SO-101 控制接口。
- end-to-end EEG + RL 评估。

## 3. EEG / BCI 问题

### Q: EEG 任务是什么？

Motor Imagery，也就是运动想象 EEG。被试根据 cue 想象左手、右手、双手或双脚运动。

### Q: EEG 分类模型是什么？

使用 EEGTransformer / CTNet 风格模型，本质是 CNN + Transformer：

```text
EEG epoch
-> CNN patch embedding
-> positional embedding
-> Transformer encoder
-> global average pooling
-> classification head
-> class logits
```

### Q: EEGTransformer 的输入是什么？

PhysioNet 4 类任务中，典型输入是：

```text
[B, 1, 64, 1000]
```

含义：

- `B`: batch size。
- `1`: CNN 输入通道维。
- `64`: EEG 通道数。
- `1000`: resample 后的时间采样点。

### Q: EEGTransformer 的输出是什么？

输出是 4 类 logits：

```text
output_dim = 4
```

经过 softmax 得到 class probability：

```python
probs = softmax(logits)
pred = argmax(probs)
confidence = max(probs)
```

### Q: 4 个 EEG class 是什么？

PhysioNet 4-class 映射：

```text
0 -> left
1 -> right
2 -> hands/up
3 -> feet/down
```

### Q: label 是怎么离散化的？

label 不是从 EEG 连续值阈值切出来的，而是由数据集里的 event cue 决定。

PhysioNet 中：

- `T0` 是 rest，被过滤掉。
- left/right imagery runs 中：`T1 -> left`，`T2 -> right`。
- hands/feet imagery runs 中：`T1 -> hands/up`，`T2 -> feet/down`。

最后映射成整数：

```text
left  -> 0
right -> 1
hands -> 2
feet  -> 3
```

### Q: EEG class 怎么映射到控制方向？

```text
0 left      -> move left
1 right     -> move right
2 hands/up  -> move up
3 feet/down -> move down
```

### Q: EEGTransformer 是否直接输出控制量？

不是。它输出的是分类 logits / probabilities，不是关节角、末端坐标或动作速度。

准确说：

> EEGTransformer 输出离散运动想象类别；这个类别作为高层意图输入到控制模块。

### Q: 为什么使用迁移学习 / fine-tuning？

因为 EEG 有很强的 subject variability。不同被试的 EEG 分布不同，直接泛化很困难。迁移学习可以让预训练或 pooled model 适配特定被试或少通道设置。

### Q: 为什么 EEG 里用 Transformer？

EEG motor imagery trial 中存在时间依赖关系。CNN 适合提取局部时空特征，Transformer encoder 适合建模不同时间 patch 之间的长程依赖。

### Q: EEG 分类用什么 loss？

多分类交叉熵：

```text
CrossEntropyLoss(logits, label)
```

### Q: EEG 预处理做了什么？

高层流程：

1. 用 MNE 读取 EDF 和 event annotations。
2. 根据 cue onset 切 EEG epoch。
3. 过滤 rest trial。
4. 重采样到统一长度，例如 1000。
5. 标准化 / 归一化。
6. 输入 EEGTransformer 分类。

### Q: EEG 为什么难？

因为 EEG：

- 信噪比低。
- 非平稳。
- 个体差异大。
- 受电极位置影响。
- 容易有眼电、肌电等 artefact。

所以不能假设 EEG 分类永远正确，需要闭环控制来纠错。

## 4. RL 输入输出问题

### Q: RL 的输入是什么？

基础版本输入 5 维：

```text
[y, z, target_y, target_z, distance]
```

含义：

- `y, z`: 当前末端在归一化 2D 平面的坐标。
- `target_y, target_z`: 目标位置。
- `distance`: 当前点到目标点的欧氏距离。

### Q: EEG-aware RL 输入是什么？

EEG-aware 版本是 7 维：

```text
[y, z, target_y, target_z, distance, eeg_pred, confidence]
```

比基础版本多了：

- `eeg_pred`: EEGTransformer 预测的意图类别。
- `confidence`: 分类置信度。

### Q: RL 输出维度是多少？

取决于动作空间：

4 方向：

```text
output_dim = 4
```

8 方向：

```text
output_dim = 8
```

### Q: RL 输出的具体含义是什么？

DQN 输出的是每个动作的 Q-value：

```text
Q(s, left)
Q(s, right)
Q(s, up)
Q(s, down)
...
```

不是直接输出坐标，也不是直接输出关节角。

### Q: 最终动作怎么选？

评估时：

```text
action = argmax(Q(s, a))
```

训练时会用 epsilon-greedy：

- 以 `epsilon` 概率随机探索。
- 否则选 Q-value 最大的动作。

### Q: 4 方向动作如何定义？

```text
0 -> left
1 -> right
2 -> up
3 -> down
```

在 2D 控制平面中，对应沿 y/z 轴移动固定步长。

### Q: 8 方向动作如何定义？

```text
0 -> left       (-1.0,  0.0)
1 -> right      ( 1.0,  0.0)
2 -> up         ( 0.0,  1.0)
3 -> down       ( 0.0, -1.0)
4 -> up_left    (-0.707,  0.707)
5 -> up_right   ( 0.707,  0.707)
6 -> down_left  (-0.707, -0.707)
7 -> down_right ( 0.707, -0.707)
```

`0.707` 约等于 `1 / sqrt(2)`，用于让对角线动作的长度和水平/垂直动作接近一致。

### Q: 为什么用离散动作，不用连续动作？

因为项目目标是先验证 EEG 意图识别和闭环控制链路。离散动作更容易训练、调试和评估，也适合 DQN。

如果扩展到高自由度灵巧手或连续控制，更适合用：

```text
PPO / SAC / TD3
```

### Q: reward 怎么设计？

reward 包含：

- 每步惩罚，鼓励更短路径。
- 距离改善奖励。
- 到达目标奖励。
- 远离目标惩罚。
- 边界惩罚。
- 振荡惩罚。
- 8 方向 smooth 版本里还有方向一致性奖励和方向改变惩罚。

### Q: 为什么 state 里要放 distance？

虽然 distance 可以由位置算出来，但直接给网络一个距离特征可以让学习更稳定，因为它明确告诉模型离目标还有多远。

### Q: 到达目标的条件是什么？

```text
distance(current_position, target_position) < target_radius
```

常见值：

```text
target_radius = 0.1
target_radius = 0.15
```

### Q: episode 什么时候结束？

两种情况：

- 到达目标。
- 达到最大步数。

### Q: 工作空间范围是什么？

归一化 2D 平面：

```text
y, z in [-1, 1]
```

超出边界会被 clip，并可能受到 boundary penalty。

## 5. DQN / Transformer DQN 问题

### Q: 你用的 RL 算法是什么？

Double DQN，带 replay buffer 和 target network，有些实验中使用 soft update。

### Q: 为什么用 Double DQN？

普通 DQN 容易 overestimate Q-value，因为同一个网络同时选择动作和评估动作。Double DQN 用 policy network 选择下一步动作，用 target network 评估该动作，能减轻过估计问题。

### Q: Double DQN target 怎么写？

```text
y = r + gamma * Q_target(s_next, argmax_a Q_policy(s_next, a))
```

如果是 terminal state：

```text
y = r
```

### Q: DQN loss 是什么？

Smooth L1 loss / Huber loss：

```text
loss = SmoothL1Loss(Q_pred, Q_target)
```

相比 MSE，它对极端 TD error 更稳。

### Q: replay buffer 存什么？

```text
(state, action, reward, next_state, done)
```

作用：

- 打破连续样本的相关性。
- 重复利用历史经验。
- 提高训练稳定性。

### Q: Transformer DQN 的输入 shape 是什么？

标准版本：

```text
[batch_size, seq_len, state_dim]
```

多数实验中：

```text
[batch_size, 1, 5]
```

或 EEG-aware：

```text
[batch_size, 1, 7]
```

### Q: Transformer DQN 的输出 shape 是什么？

```text
[batch_size, action_dim]
```

即：

```text
[batch_size, 4]
```

或：

```text
[batch_size, 8]
```

### Q: Transformer DQN 的结构是什么？

```text
state input
-> linear projection
-> positional encoding
-> Transformer encoder
-> fully connected Q head
-> action_dim Q-values
```

常见配置：

```text
d_model = 64
n_heads = 4
n_layers = 2
d_ff = 256
dropout = 0.1
```

### Q: smooth 8-direction Transformer DQN 是什么？

这是一个序列版 Transformer DQN，用历史状态序列帮助生成更平滑的动作：

```text
seq_len = 10
state_dim = 5
action_dim = 8
```

它还使用 action embedding，并在 reward 中鼓励方向一致、惩罚频繁变向。

### Q: seq_len=1 时为什么还用 Transformer？

标准版本中，Transformer DQN 主要作为一种 Q-network 架构，与 CNN+LSTM 和 LightTransformer 对比。后续 smooth 版本才更充分使用历史序列信息。

### Q: RL 网络对比了哪些模型？

对比过：

- CNN+LSTM DQN。
- LightTransformer DQN。
- Transformer DQN。

### Q: EEGTransformer 和 Transformer DQN 是不是同一个模型？

不是。

- EEGTransformer: 输入 EEG，输出 EEG class logits。
- Transformer DQN: 输入控制状态，输出动作 Q-values。

这两个模型都用了 Transformer 思想，但任务、输入输出和训练目标不同。

## 6. 机械臂 / 控制接口问题

### Q: 机械臂是什么？

目标硬件接口是 SO-101 机械臂，使用串口控制。仿真侧使用 PyBullet。

### Q: 机械臂几轴？

更严谨说法：

> SO-101 接口中处理的是 5 个机械臂关节加 1 个夹爪舵机，也可以说是 6 个舵机通道。但我的目标到达实验并没有让 RL 直接控制全部 6 个通道，而是把任务抽象成 2D 末端运动。

舵机 / 关节名称：

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

### Q: 实际控制了哪些关节？

简化物理控制中主要映射：

```text
left/right -> shoulder_pan
up/down    -> wrist_flex
```

其他关节保持在 home / mid pose 附近，用于稳定实验。

### Q: 为什么不用所有关节？

因为项目重点是验证 EEG 到 RL 闭环控制链路，而不是做完整灵巧操作。2D 抽象能降低硬件风险，让实验更可控、更容易复现。

### Q: 离散动作如何变成机械臂运动？

仿真中：

```text
action -> 更新归一化 y/z 坐标 -> PyBullet 环境执行
```

物理串口控制中：

```text
action -> 选择关节方向 -> radian increment -> servo ticks -> serial command
```

### Q: RL 是否直接输出 motor ticks？

不是。RL 只输出离散动作选择。环境 wrapper 负责把动作翻译成 servo ticks 或目标位置。

### Q: 串口参数是什么？

常见设置：

```text
baud = 1,000,000
timeout = 0.02 s
```

V2 平滑控制中：

```text
joint_step_rad = 0.12
move_time_ms = 500
action_delay_ms = 600
```

### Q: 如何保证硬件安全？

包括：

- 先仿真后硬件。
- 工作空间归一化边界。
- soft limit。
- step size 限制。
- home / mid pose 初始化。
- 平滑运动时间控制。
- 不声称真人实时 EEG 硬件闭环结果。

## 7. PyBullet / 仿真问题

### Q: 为什么用 PyBullet？

PyBullet 用于 simulation-first validation。它可以在不损坏硬件的情况下验证动作映射、目标到达、轨迹和闭环逻辑。

### Q: PyBullet 仿真了什么？

仿真机械臂控制环境和目标到达任务。RL 仍然使用简化的 2D 控制状态。

### Q: 这是完整 Sim2Real 吗？

不能这么说。更准确是：

> PyBullet 用于闭环控制逻辑和动作接口的仿真验证，物理侧主要验证串口控制和动作映射，不是完整动力学匹配的 Sim2Real benchmark。

### Q: 如果被问 Sim2Real limitation 怎么答？

可以答：

> 我目前验证的是 simulation-first control pipeline 和 hardware-interface execution。真实接触动力学、摩擦、延迟和多自由度连续控制还需要更完整的 Sim2Real 设计。

## 8. OpenBCI / BrainFlow 问题

### Q: OpenBCI 在项目里做什么？

OpenBCI 是目标 EEG 采集硬件接口。BrainFlow 提供读取 OpenBCI 或 synthetic board 数据流的软件 API。

### Q: 是否做了真人实时 EEG 控制？

要谨慎回答：

> archived online-control evidence 使用的是 BrainFlow synthetic board / simulation-oriented validation，不是 live human OpenBCI recording。

### Q: 为什么用 synthetic BrainFlow？

因为它安全、可重复，而且能验证软件链路：

```text
stream -> epoch -> classifier -> control state/action -> execution
```

### Q: offline 和 realtime 有什么区别？

offline：

- 使用保存好的 EEG trial。
- EEGTransformer 分类。
- 预测结果进入 RL 评估。

realtime / online-style：

- BrainFlow 读取数据流。
- 构造 EEG epoch。
- 分类。
- 映射成 intent 或 RL state。
- 执行控制动作。

## 9. 实验结果问题

### Q: end-to-end evaluation 是什么？

把 EEG 分类和 RL 控制接起来：

```text
EEGTransformer prediction
-> EEG-aware RL state
-> Transformer DQN
-> target-reaching evaluation
```

### Q: with EEG 和 without EEG 比较说明什么？

说明 EEG 预测作为 noisy intent feature 是否能帮助 RL 控制，以及 RL 是否能在 EEG 分类不完美时保持稳定目标到达。

### Q: 关键 E2E 数字是什么？

可以记：

- EEG 分类准确率约 82.22%。
- without EEG: 5D state，约 99.0% reach rate。
- with EEG: 7D state，约 98.7% reach rate。
- 记录中 EEG-aware agent 训练更快。

但如果面试官没有问具体数字，不要主动堆数字，重点讲技术链路。

### Q: 为什么 EEG 不完美但目标到达率还高？

因为 RL 是闭环控制，不是盲目执行 EEG 分类结果。它每一步都有当前位置、目标位置和距离反馈，所以可以纠正错误。

### Q: 做了哪些 ablation？

包括：

- CNN+LSTM vs LightTransformer vs Transformer DQN。
- 4 方向 vs 8 方向。
- fixed step vs adaptive step。
- 有无 EEG prediction feature。
- channel reduction。
- PhysioNet fine-tuning。
- OpenBCI-compatible electrode subset。

### Q: 为什么需要 ablation？

为了说明性能不是偶然来的，而是由模型结构、动作空间、状态设计、奖励设计等因素共同影响。

## 10. 数据集问题

### Q: 用了哪些数据集？

项目涉及：

- BCI Competition IV-2a。
- BCI Competition IV-2b。
- PhysioNet EEG Motor Movement/Imagery Dataset。

主线 4 类 EEG-to-control 映射主要基于 PhysioNet 风格 motor imagery。

### Q: 为什么用 PhysioNet？

PhysioNet 被试更多，包含 left/right 和 hands/feet motor imagery runs，适合训练和评估多被试 EEG 分类模型。

### Q: 为什么也用 BCI IV-2a / IV-2b？

它们是经典 BCI motor imagery benchmark：

- IV-2a: 4 类，22 通道。
- IV-2b: 2 类，3 通道。

### Q: 不同通道数怎么处理？

根据数据集调整模型输入通道：

```text
PhysioNet -> 64 channels
IV-2a -> 22 channels
IV-2b -> 3 channels
OpenBCI-oriented -> 8 channels or fewer
```

### Q: 为什么做 channel reduction？

实际硬件不一定有 64 通道。channel reduction 用来验证少通道、OpenBCI-compatible 设置下是否还能保留有用的 motor imagery 信息。

### Q: motor imagery 重要通道有哪些？

常见重要区域是 sensorimotor cortex 附近：

```text
C3, C4, Cz, FC3, FC4, CP3, CP4, FCz
```

## 11. 数学和算法问题

### Q: 什么是 Q-value？

Q-value 表示在状态 `s` 下执行动作 `a` 后，未来期望累计 reward：

```text
Q(s, a) = expected discounted future return
```

### Q: Bellman target 是什么？

普通 DQN：

```text
y = r + gamma * max_a Q_target(s_next, a)
```

Double DQN：

```text
y = r + gamma * Q_target(s_next, argmax_a Q_policy(s_next, a))
```

### Q: gamma 是什么？

`gamma` 是未来奖励折扣因子。`gamma = 0.99` 表示 agent 比较重视长期目标到达，而不是只看当前一步。

### Q: tau 是什么？

soft update 系数：

```text
target = tau * policy + (1 - tau) * target
```

作用是让 target network 缓慢跟随 policy network，提高训练稳定性。

### Q: 什么是 reward shaping？

在最终到达奖励之外，加入中间奖励，例如距离变近就奖励。这样 agent 不需要等到偶然到达目标才获得学习信号。

### Q: 为什么惩罚 oscillation？

因为 agent 可能学到左右来回震荡但不接近目标的行为。惩罚振荡可以让轨迹更稳定。

### Q: 为什么用 Huber loss？

Huber / Smooth L1 对大误差不如 MSE 敏感，DQN 中 TD error 可能波动较大，所以 Huber 更稳定。

## 12. 设计选择问题

### Q: 为什么不直接把 EEG class 映射成动作？

直接映射是 open-loop，分类错了动作就错了。RL 加入状态反馈，可以通过后续动作修正错误。

### Q: 为什么把 EEG prediction 放进 RL state？

这样 EEG 是一个建议信号，而不是强制动作。RL 可以根据目标位置、当前距离和 confidence 决定是否信任它。

### Q: 为什么 state 里要有 target position？

同一个当前位置，在不同目标下最优动作不同。没有目标位置，policy 不知道该往哪里去。

### Q: 为什么当前任务用 DQN，而不是 PPO / SAC？

因为当前动作空间是离散的，DQN 很适合。PPO / SAC 更适合连续动作空间，例如高自由度机械臂或灵巧手。

### Q: 如何扩展到灵巧手？

需要：

- 换成连续控制算法，如 PPO / SAC / TD3。
- 加入 vision、tactile、force、proprioception。
- 使用更真实的仿真环境，如 MuJoCo、Isaac Gym、ManiSkill。
- 加入 contact reward、slip penalty、force constraint。

### Q: VLA / foundation policy 如何接入？

VLA 可以负责高层任务理解和视觉语言 grounding，RL 负责低层接触恢复和 reward optimisation。当前项目类似这个思想：EEG classifier 提供高层意图，RL 负责低层闭环控制。

## 13. 失败模式问题

### Q: EEG 分类错了怎么办？

RL 不会完全依赖 EEG。它还有当前位置、目标位置和距离反馈，所以可以在后续步骤纠错。

### Q: 主要 failure modes 有哪些？

- EEG class 混淆。
- subject shift。
- 轨迹震荡。
- 到达边界。
- reward shaping 不合理。
- 机械臂关节限位。
- 仿真和真实硬件差异。

### Q: 怎么 debug RL 不稳定？

看这些：

- learning curve。
- reach rate。
- mean reward。
- trajectory plot。
- per-direction success。
- action history。
- boundary failure case。
- oscillation case。

### Q: 如果 agent reward 高但行为不对怎么办？

不能只看 reward，要看轨迹和最终 distance。需要检查是否 reward hacking，比如卡边界、震荡或走无效路径。

### Q: classifier 置信度高但错了怎么办？

confidence 是 softmax maximum，不一定校准。可以进一步做 calibration 或 uncertainty-aware policy。目前设计里 RL 仍然依赖闭环状态反馈，因此不会完全盲信 classifier。

## 14. 实现细节问题

### Q: 用了哪些框架？

```text
PyTorch -> EEGTransformer 和 DQN
MNE -> EEG 数据读取和 epoch 切分
NumPy / SciPy -> 预处理和 resampling
PyBullet -> 机械臂仿真
BrainFlow -> OpenBCI / synthetic EEG stream
Serial protocol -> SO-101 控制
Matplotlib -> 图和轨迹分析
```

### Q: replay buffer 里的 state shape 是什么？

DQN 输入需要序列格式，所以 state 存成：

```text
[1, state_dim]
```

训练 batch 后是：

```text
[batch_size, 1, state_dim]
```

### Q: EEG classifier batch input shape 是什么？

PhysioNet 4-class：

```text
[batch_size, 1, 64, 1000]
```

### Q: confidence 是什么？

```text
confidence = max(softmax(logits))
```

它是模型对预测类别的 softmax confidence，不一定是校准后的真实概率。

### Q: `target_y, target_z` 和 action direction 有什么区别？

`target_y, target_z` 是最终目标位置；action direction 是当前 step 选择的一步运动方向。

### Q: adaptive step 是什么？

固定步长每一步移动同样距离。adaptive step 根据当前距离调节步长，距离远时步长大，接近目标时步长小，从而更快接近目标并减少 overshoot。

## 15. 容易说错的话

不要说：

> RL 直接输出机械臂关节角。

应该说：

> RL 输出离散动作的 Q-value，环境接口再把动作转换成关节增量或末端位移。

不要说：

> EEG 直接控制机械臂。

应该说：

> EEG 提供高层意图 cue，RL 根据闭环状态反馈决定动作。

不要说：

> 我做了真人实时 EEG 硬件闭环控制。

应该说：

> 我验证了 offline EEG-to-RL pipeline，以及 BrainFlow synthetic board / simulation-oriented online control path。

不要说：

> RL 控制了 SO-101 的全部 6 个轴。

应该说：

> SO-101 有 5 个机械臂关节加 1 个夹爪舵机，但我的目标到达实验抽象成 2D 末端控制，主要控制左右和上下两个自由度。

不要说：

> 这是完整的灵巧操作系统。

应该说：

> 这是面向 EEG 意图识别和闭环目标到达的验证系统，可迁移到更复杂的灵巧操作场景。

## 16. 高频快问快答

### RL 输出维度是多少？

4 方向版本是 4，8 方向版本是 8。

### RL 输入维度是多少？

基础版本是 5，加入 EEG prediction 和 confidence 后是 7。

### RL 输出的是什么？

离散动作的 Q-value vector。

### EEGTransformer 输出什么？

4 类 motor imagery logits / probabilities。

### 动作怎么选？

评估时选 Q-value 最大的动作。

### `state_dim = 5` 是什么？

```text
[y, z, target_y, target_z, distance]
```

### `state_dim = 7` 是什么？

5D state 加：

```text
[eeg_pred, confidence]
```

### action 是连续的吗？

不是，当前项目是离散动作。

### 四个动作是什么？

left、right、up、down。

### 八个动作是什么？

left、right、up、down、up_left、up_right、down_left、down_right。

### 机械臂是什么？

SO-101，串口控制；同时有 PyBullet 仿真环境。

### 机械臂几轴？

硬件上是 5 个 arm joints + 1 个 gripper servo；项目控制抽象主要使用 2D 运动自由度。

### 主要控制哪两个关节？

`shoulder_pan` 控左右，`wrist_flex` 控上下。

### 为什么用 RL？

为了让系统闭环纠错，而不是盲目执行 EEG 分类结果。

### 为什么用 Transformer DQN？

用于比较 self-attention Q-network，并支持后续序列状态控制扩展。

### 为什么不用 PPO / SAC？

当前动作空间是离散的，DQN 合适；连续高自由度控制更适合 PPO / SAC。

### 到达目标怎么判定？

当前点到目标点距离小于 `target_radius`。

### 最大 limitation 是什么？

物理控制验证是简化的 2D 目标到达，不是完整高自由度接触操作，也不是真人实时 EEG 硬件闭环证明。

## 17. 如果让你现场画 pipeline

画这个：

```text
EEG epoch
  -> EEGTransformer / CTNet
  -> class prediction + confidence
  -> RL state:
     [y, z, target_y, target_z, distance, eeg_pred, confidence]
  -> Transformer DQN
  -> Q-values over actions
  -> argmax action
  -> PyBullet / SO-101 serial controller
  -> new arm state
  -> feedback loop
```

## 18. 最推荐背的技术回答

> 我的系统不是让 EEG 直接控制机械臂，而是先用 EEGTransformer 把 motor imagery EEG 分类成离散意图，再把这个意图作为 noisy cue 加入 RL 状态。Transformer DQN 根据当前位置、目标位置、距离和 EEG cue 输出离散动作的 Q-value，并通过闭环反馈不断纠正动作，实现目标到达。

## 19. 如果被问“你的工作和普通 EEG 分类有什么区别？”

可以答：

> 普通 EEG 分类停在 offline accuracy，而我的项目把分类结果接到了闭环控制任务里。也就是说，我不仅训练了 EEG classifier，还设计了 RL state/action/reward，让 noisy classification 能在控制闭环中被利用和纠错。

## 20. 如果被问“这个项目和 embodied AI / robot learning 有什么关系？”

可以答：

> 这个项目虽然输入是 EEG，但系统结构和 embodied AI 很接近：感知模块提供高层语义或意图，低层 policy 根据环境反馈执行动作。未来如果把 EEG cue 换成 vision-language cue，或者把离散 DQN 换成连续控制 policy，就可以扩展到更复杂的 robot learning 或 dexterous manipulation。

