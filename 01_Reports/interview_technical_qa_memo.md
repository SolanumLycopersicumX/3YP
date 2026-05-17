# Interview Technical Q&A Memo

This memo is for quick interview preparation. It focuses on technical questions that are likely to be asked about the EEGTransformer + RL robotic-arm control project.

## 0. One-Minute Technical Summary

I built a BCI-to-control pipeline. The EEG side uses an EEGTransformer / CTNet-style CNN-Transformer model to classify motor-imagery EEG into discrete intent classes. The control side uses a Transformer DQN policy. The RL state contains the arm's current 2D end-effector position, target position, and distance; in the EEG-aware version it also includes the EEG predicted class and confidence. The DQN outputs Q-values over discrete movement actions, and the selected action is sent to either a PyBullet simulation or an SO-101 serial-control interface.

Short version:

> EEGTransformer decodes noisy EEG intent; Transformer DQN converts state feedback into discrete closed-loop movement actions; PyBullet / OpenBCI / serial SO-101 interfaces provide simulation and hardware-integration validation.

## 1. Key Numbers To Memorise

| Item | Answer |
|---|---|
| Basic RL input dimension | 5 |
| EEG-aware RL input dimension | 7 |
| 4-direction RL output dimension | 4 |
| 8-direction RL output dimension | 8 |
| Basic RL state | `[y, z, target_y, target_z, distance]` |
| EEG-aware RL state | `[y, z, target_y, target_z, distance, eeg_pred, confidence]` |
| 4-direction actions | left, right, up, down |
| 8-direction actions | left, right, up, down, up_left, up_right, down_left, down_right |
| DQN output | Q-value vector, not direct coordinates |
| Action selection | `argmax_a Q(s, a)` |
| Main RL algorithm | Double DQN with target network / soft update |
| RL discount factor | `gamma = 0.99` in end-to-end eval |
| Target soft update | `tau = 0.005` in end-to-end eval |
| Replay buffer size | 100,000 in end-to-end eval |
| Basic RL step size | 0.05 in normalised 2D plane |
| E2E target radius | 0.1 in normalised 2D plane |
| 8-dir smooth target radius | 0.15 |
| Normalised boundary | `[-1, 1]` for each axis |
| EEG classification output | 4 logits / class probabilities |
| PhysioNet EEG input shape | usually `[B, 1, 64, 1000]` after resampling |
| EEGTransformer `d_model` | 64 |
| EEGTransformer attention heads | 8 in PhysioNet 4-class classifier |
| EEGTransformer encoder layers | 2 |
| EEGTransformer FFN dim | 256 |
| EEGTransformer patch size | 25 samples |
| PhysioNet target sample length | 1000 |
| SO-101 hardware structure | 5 arm joints + 1 gripper servo |
| Main physical-control abstraction | 2D motion using two controlled joints |
| Main controlled joints | `shoulder_pan` for left/right, `wrist_flex` for up/down |
| Serial baud rate | 1,000,000 |

## 2. System Architecture Questions

### Q: What exactly did you build?

I built an EEG-to-robot-control pipeline. It has three parts:

1. EEGTransformer classifies motor-imagery EEG into discrete intent classes.
2. Transformer DQN uses closed-loop state feedback to select movement actions.
3. PyBullet / serial SO-101 control executes those actions in simulation or hardware-interface validation.

### Q: What is the input and output of the whole system?

Input:

- EEG trial or EEG epoch.
- Current arm state and target state for RL control.

Output:

- A discrete movement action such as left/right/up/down.
- The low-level control interface converts that action into simulated end-effector movement or serial joint increment commands.

### Q: Is EEG directly controlling the mechanical arm?

Not directly. EEGTransformer gives a high-level noisy intent prediction. The RL controller uses that prediction together with state feedback, then decides the final movement action. This is important because it makes the system closed-loop rather than simply replaying classifier outputs.

### Q: What makes it closed-loop?

The RL policy observes the current position, target position, and distance at every step. After each action, the environment updates the state, so the next action can correct previous mistakes. The control is not a one-shot mapping from EEG class to motor command.

### Q: What is the main technical contribution?

The main contribution is connecting a noisy EEG intention-recognition module with a reinforcement-learning controller that can still perform goal-reaching under imperfect classification. The work is not only EEG classification; it includes control-state design, action discretisation, reward design, RL training, simulation evaluation, and hardware-interface validation.

## 3. EEG / BCI Questions

### Q: What EEG task did you use?

Motor imagery. The classes correspond to imagined left hand, right hand, both hands / up, and both feet / down, depending on the dataset mapping.

### Q: What model did you use for EEG classification?

An EEGTransformer / CTNet-style architecture. It combines CNN-based patch embedding for local spatiotemporal feature extraction with a Transformer encoder for temporal dependency modelling.

### Q: What is the EEGTransformer input?

For PhysioNet, the common input shape is:

```text
[B, 1, 64, 1000]
```

where:

- `B` is batch size.
- `1` is the input channel dimension used by the CNN frontend.
- `64` is the EEG channel count.
- `1000` is the resampled time length.

### Q: What is the EEGTransformer output?

The classifier outputs logits over 4 classes:

```text
output dim = 4
```

After softmax, these become class probabilities. The predicted class is `argmax(logits)`, and confidence is the maximum softmax probability.

### Q: What are the four EEG labels?

For the PhysioNet 4-class setup:

```text
0 -> left
1 -> right
2 -> hands/up
3 -> feet/down
```

### Q: How were EEG labels discretised?

The labels come from event cues, not from thresholding continuous EEG values.

For PhysioNet:

- `T0` is rest and is filtered out.
- In left/right imagery runs, `T1 -> left`, `T2 -> right`.
- In hands/feet imagery runs, `T1 -> hands/up`, `T2 -> feet/down`.

Then the labels are mapped to integer classes:

```text
left  -> 0
right -> 1
hands -> 2
feet  -> 3
```

### Q: How are EEG labels mapped to control directions?

```text
0 left      -> move left
1 right     -> move right
2 hands/up  -> move up
3 feet/down -> move down
```

### Q: Did the classifier output continuous control values?

No. It outputs discrete motor-imagery class logits. Continuous EEG is first classified into discrete intent labels, and then the control system uses those labels as intent information.

### Q: What is the role of transfer learning?

EEG distributions vary strongly across subjects. Transfer learning / fine-tuning is used to adapt a general EEGTransformer model to subject-specific or reduced-channel settings rather than training every model completely from scratch.

### Q: Why use a Transformer for EEG?

EEG motor imagery contains temporal patterns across an epoch. CNN layers capture local temporal and spatial features, while the Transformer encoder can model longer-range relationships between temporal patches.

### Q: What is the EEGTransformer architecture?

High-level structure:

```text
EEG epoch
-> PatchEmbeddingCNN
-> learnable positional embedding
-> Transformer Encoder
-> global average pooling
-> classification head
-> 4-class logits
```

Key configuration:

```text
d_model = 64
n_heads = 8
n_layers = 2
d_ff = 256
patch_size = 25
dropout = 0.1
```

### Q: What loss did you use for EEG classification?

Cross-entropy loss over the discrete motor-imagery classes.

### Q: What preprocessing did you use?

At a high level:

- Load EEG recordings and event annotations with MNE.
- Segment trials into epochs around motor-imagery cue onset.
- Filter out rest trials for the 4-class control mapping.
- Resample epochs to a common length.
- Standardise / normalise data using training statistics where appropriate.

### Q: Why is EEG classification difficult?

Because EEG is low signal-to-noise, non-stationary, subject-dependent, and sensitive to electrode placement, fatigue, and artefacts. That is why the control system should not assume perfect classification.

## 4. RL Input / Output Questions

### Q: What is the RL state?

Basic state:

```text
[y, z, target_y, target_z, distance]
```

Dimension:

```text
state_dim = 5
```

EEG-aware state:

```text
[y, z, target_y, target_z, distance, eeg_pred, confidence]
```

Dimension:

```text
state_dim = 7
```

### Q: What does `y, z` mean?

They are the current normalised 2D end-effector coordinates in the control plane. The task abstracts the robotic arm endpoint motion into a 2D plane for stable goal-reaching experiments.

### Q: What is the RL output?

The DQN outputs Q-values for all candidate actions:

```text
q_values = Q(s, a)
```

Shape:

```text
[batch_size, action_dim]
```

For 4 directions:

```text
[batch_size, 4]
```

For 8 directions:

```text
[batch_size, 8]
```

### Q: Does RL output joint angles?

No. The RL network outputs Q-values over discrete actions. The chosen action is then converted by the environment into a position update or joint increment.

### Q: How is the final action selected?

```text
action = argmax(q_values)
```

During training, epsilon-greedy exploration is used, so sometimes the agent samples a random action.

### Q: What is the 4-direction action mapping?

```text
0 -> left
1 -> right
2 -> up
3 -> down
```

In the 2D plane, this corresponds to moving one fixed step along the y/z axes.

### Q: What is the 8-direction action mapping?

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

The `0.707` terms approximate `1 / sqrt(2)` so diagonal movement has similar magnitude to axis-aligned movement.

### Q: Why is the action space discrete?

The project goal was to validate robust EEG-to-control closed-loop behaviour. A discrete action space makes the task easier to train, easier to debug, and well-suited to DQN. For high-DOF dexterous manipulation, a continuous-action method such as PPO, SAC, or TD3 would be more appropriate.

### Q: What is the reward function?

The reward combines:

- Step penalty to encourage shorter paths.
- Distance-improvement reward.
- Reaching bonus.
- Penalty for moving away from the target.
- Boundary penalty.
- Oscillation penalty in some versions.
- Direction-smoothness reward / direction-change penalty in the smooth 8-direction version.

### Q: Why include distance in the state if the model can compute it from positions?

Including distance gives the network a direct scalar progress signal. It reduces the burden on the network to infer Euclidean distance from coordinates and can stabilise learning.

### Q: What is the target-reaching condition?

The episode is successful if:

```text
distance(current_position, target_position) < target_radius
```

Typical values:

```text
target_radius = 0.1 in end-to-end evaluation
target_radius = 0.15 in some 4-direction / 8-direction training
```

### Q: What terminates an episode?

An episode ends when:

- The arm reaches the target radius, or
- The maximum number of steps is reached.

### Q: What is the normalised workspace?

The 2D control plane is clipped to:

```text
y, z in [-1, 1]
```

## 5. DQN / Transformer DQN Questions

### Q: What exact RL algorithm did you use?

Double DQN with a target network and replay buffer. Some versions use soft target updates.

### Q: Why Double DQN?

Standard DQN can overestimate Q-values because the same network selects and evaluates actions. Double DQN reduces this by using the policy network to select the best next action and the target network to evaluate it.

### Q: What is the DQN target?

For non-terminal transitions:

```text
y = r + gamma * Q_target(s_next, argmax_a Q_policy(s_next, a))
```

For terminal transitions:

```text
y = r
```

### Q: What loss is used for DQN?

Smooth L1 loss / Huber loss between predicted Q-values and the target Q-values.

### Q: What is the replay buffer for?

The replay buffer stores transitions:

```text
(state, action, reward, next_state, done)
```

It breaks temporal correlation between consecutive samples and improves sample efficiency by reusing past experience.

### Q: What is epsilon-greedy?

During training:

- With probability `epsilon`, choose a random action for exploration.
- Otherwise choose `argmax(Q)`.

`epsilon` decays from high exploration to low exploration.

### Q: What is the Transformer DQN input shape?

For the standard DQN setup:

```text
[batch_size, seq_len, state_dim]
```

In most experiments:

```text
seq_len = 1
state_dim = 5 or 7
```

### Q: What is the Transformer DQN output shape?

```text
[batch_size, action_dim]
```

where `action_dim` is 4 or 8.

### Q: What is the Transformer DQN architecture?

High-level structure:

```text
state input
-> linear projection to d_model
-> positional encoding
-> Transformer encoder
-> final hidden state / pooling
-> fully connected Q head
-> action_dim Q-values
```

Typical settings:

```text
d_model = 64
n_heads = 4
n_layers = 2
d_ff = 256
dropout = 0.1
```

### Q: What is the smooth 8-direction model?

The smooth 8-direction version uses a sequence Transformer DQN. Instead of only one current state, it can consume a short history:

```text
seq_len = 10
state_dim = 5
action_dim = 8
```

It also uses action embeddings and rewards smoother trajectories by penalising frequent direction changes.

### Q: Why use a Transformer DQN if `seq_len = 1`?

For the standard version, the Transformer is mainly used as a comparable Q-network architecture against CNN+LSTM and lighter transformer variants. The later sequence version uses history more directly with `seq_len = 10`.

### Q: What baselines did you compare?

RL network comparison included:

- CNN+LSTM DQN.
- LightTransformer DQN.
- Transformer DQN.

The Transformer version achieved the strongest final reach rate in the architecture comparison.

### Q: What is the difference between the classifier Transformer and RL Transformer?

They are different networks:

- EEGTransformer classifies EEG time-series into motor-imagery classes.
- Transformer DQN estimates Q-values over movement actions from control state.

Do not describe them as the same model.

## 6. Mechanical Arm / Control Interface Questions

### Q: What robot arm did you use?

The hardware interface targets an SO-101 robotic arm using serial control. The report also uses PyBullet for simulation-first validation.

### Q: How many axes does the arm have?

The SO-101 interface treats the hardware as 6 servo-controlled channels:

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

A careful answer:

> It is a 5-DOF arm plus a gripper servo. In my goal-reaching control experiment, I abstracted the task to a 2D end-effector plane and mainly controlled two motion degrees: left/right and up/down.

### Q: Which joints are controlled in the simplified physical-control task?

The simplified serial environment maps:

```text
left/right -> shoulder_pan
up/down    -> wrist_flex
```

Other joints are kept near home/mid positions for stability.

### Q: Why only control two joints if the arm has more?

The goal was not full dexterous manipulation. The goal was to validate the EEG-to-control closed-loop pipeline safely. A 2D abstraction reduces hardware risk and makes the RL problem easier to evaluate reproducibly.

### Q: How does a discrete action become a physical movement?

In simulation:

```text
action -> update normalised y/z position -> PyBullet IK or environment movement
```

In serial physical control:

```text
action -> choose joint direction -> convert radian increment to servo ticks -> send serial command
```

### Q: Does the RL policy command raw motor ticks?

No. The RL policy chooses high-level discrete actions. The environment wrapper translates these actions into joint increments or target positions.

### Q: What serial settings are used?

The serial interface uses:

```text
baud = 1,000,000
timeout = 0.02 s
```

The V2 controller uses smooth motion settings such as:

```text
joint_step_rad = 0.12
move_time_ms = 500
action_delay_ms = 600
```

### Q: How do you prevent unsafe hardware movement?

Safety mechanisms include:

- Simulation-first validation.
- Normalised workspace bounds.
- Joint soft limits.
- Step-size limits.
- Home / mid pose initialisation.
- Smooth movement timing.
- Avoiding claims based on live human EEG control when only synthetic / offline validation was used.

## 7. PyBullet / Simulation Questions

### Q: Why use PyBullet?

PyBullet provides a controlled environment for validating the control loop before hardware deployment. It allows repeatable testing of state updates, action mappings, target reaching, and trajectory plots without risking hardware.

### Q: What does the PyBullet environment simulate?

It simulates the robotic arm end-effector movement and target-reaching task. The RL state is still the simplified 2D control representation.

### Q: Is this full physics-accurate Sim2Real?

No. It is better described as simulation-first control validation and hardware-interface validation. It does not prove full Sim2Real transfer for complex contact-rich manipulation.

### Q: What should you say if asked about Sim2Real limitations?

Say:

> The PyBullet setup validates the closed-loop control logic and action interface, but it is not a full dynamics-matched Sim2Real benchmark. The physical side mainly validates serial command execution and safe motion mapping.

## 8. OpenBCI / BrainFlow Questions

### Q: What is OpenBCI used for?

OpenBCI is the intended EEG acquisition hardware interface. BrainFlow provides a software API for reading EEG streams from OpenBCI boards or synthetic boards.

### Q: Did you run real live human EEG control?

Be careful:

> The archived online-control evidence uses BrainFlow synthetic board / simulation-oriented validation, not live human OpenBCI recordings.

### Q: Why use synthetic BrainFlow data?

For safety, repeatability, and policy constraints. It validates the software pipeline from data stream to classification/control interface without claiming unsupported live-subject performance.

### Q: What is the difference between offline and realtime mode?

Offline:

- Use stored EEG trials.
- Classify them with EEGTransformer.
- Feed predictions into the RL evaluation.

Realtime / online-style:

- Use BrainFlow stream or synthetic stream.
- Build an EEG epoch.
- Run classification.
- Convert classification to action or RL state.
- Execute a control step.

## 9. Experiment / Result Questions

### Q: What was the end-to-end evaluation?

The end-to-end evaluation connected:

```text
CTNet / EEGTransformer classification predictions
-> EEG-aware RL state
-> Transformer DQN control
-> target-reaching evaluation
```

It compared RL with and without EEG prediction features.

### Q: What does the with-EEG vs without-EEG comparison test?

It tests whether noisy EEG intent predictions can be useful as additional state information for RL control, and whether the RL policy can remain robust when classification is imperfect.

### Q: What are the key reported E2E numbers?

From the end-to-end summary:

- EEG classification accuracy: 82.22%.
- Without EEG state: 5D state, around 99.0% reach rate.
- With EEG state: 7D state, around 98.7% reach rate.
- The EEG-aware agent trained faster in the recorded run.

If the interviewer does not ask for numbers, focus on the technical pipeline rather than memorising every percentage.

### Q: Why is the reach rate high even with imperfect EEG?

Because the RL controller is closed-loop. It does not depend solely on the EEG label. It observes current position and target distance after every step and can correct wrong initial cues.

### Q: What ablations / comparisons did you run?

Relevant categories:

- CNN+LSTM vs LightTransformer vs Transformer DQN.
- 4-direction vs 8-direction action spaces.
- Fixed step vs adaptive step movement.
- EEG classification with and without fine-tuning.
- Channel reduction for OpenBCI-compatible electrode sets.
- End-to-end RL with and without EEG prediction features.

### Q: Why do you need ablations?

Ablations show which design choices matter. They separate the effect of the classifier, the RL architecture, action-space design, and input-state design.

## 10. Data / Dataset Questions

### Q: Which datasets are involved?

The project includes work with:

- BCI Competition IV-2a.
- BCI Competition IV-2b.
- PhysioNet EEG Motor Movement/Imagery Dataset.

The main 4-class EEG-to-control mapping uses PhysioNet-style motor imagery classes.

### Q: Why PhysioNet?

PhysioNet provides a larger multi-subject EEG motor imagery dataset and supports left/right and hands/feet imagery tasks. It is useful for training and evaluating a more general EEG classifier.

### Q: Why BCI IV-2a / IV-2b?

They are common motor imagery benchmarks. IV-2a supports 4-class MI classification, and IV-2b supports 2-class classification.

### Q: How do you handle different channel counts?

The model configuration changes based on dataset channel count. For example:

- PhysioNet: 64 channels.
- IV-2a: 22 channels.
- IV-2b: 3 channels.
- OpenBCI-oriented experiments reduce channels to 8 or fewer selected motor-cortex electrodes.

### Q: Why do channel reduction?

Real EEG hardware may not have 64 channels. Channel reduction tests whether a smaller, practical electrode set can retain useful classification performance for deployment-style scenarios.

### Q: Which channels are important for motor imagery?

Motor cortex channels around C3, C4, Cz and neighbouring FC/CP electrodes are important because left/right hand motor imagery is reflected around sensorimotor cortex regions.

## 11. Mathematical / Algorithm Questions

### Q: What is a Q-value?

A Q-value estimates the expected future return if the agent takes action `a` in state `s` and then follows its policy:

```text
Q(s, a) = expected discounted future reward
```

### Q: What is the Bellman target?

For DQN:

```text
y = r + gamma * max_a Q_target(s_next, a)
```

For Double DQN:

```text
y = r + gamma * Q_target(s_next, argmax_a Q_policy(s_next, a))
```

### Q: What is the role of `gamma`?

`gamma` discounts future rewards. A value close to 1, such as 0.99, makes the agent consider long-term target reaching rather than only immediate movement.

### Q: What is the role of `tau` in soft update?

Soft update slowly moves the target network toward the policy network:

```text
target = tau * policy + (1 - tau) * target
```

This stabilises training.

### Q: What is reward shaping?

Reward shaping adds intermediate signals, such as distance improvement, so the agent gets learning feedback before actually reaching the final target.

### Q: Why penalise oscillation?

Without an oscillation penalty, a policy can learn back-and-forth movement that does not efficiently reduce distance. Penalising repeated reversal improves trajectory stability.

### Q: Why use Huber loss / Smooth L1?

Huber loss is less sensitive to large TD errors than MSE, which makes DQN training more stable.

## 12. Design-Choice Questions

### Q: Why not just map EEG class directly to movement?

Direct mapping is open-loop and fragile. If the EEG classifier makes a mistake, the robot follows the wrong command. RL adds state feedback and can correct over multiple steps.

### Q: Why include EEG prediction in the RL state rather than using it as the action?

Using EEG as state makes it a suggestion or intent cue. The policy can decide how much to trust it based on the current target and position.

### Q: Why use target position in the state?

Because the policy needs to know where it is trying to go. Without target position, the same current position could require different actions for different targets.

### Q: Why use distance as a separate feature?

It gives the network explicit progress information and supports reward alignment.

### Q: Why use discrete 2D control instead of full joint-space control?

It isolates the core research question: can noisy EEG intent be integrated into closed-loop RL control? Full joint-space continuous control would add many confounding difficulties.

### Q: How would you extend this to dexterous manipulation?

I would replace discrete DQN with a continuous-control method such as SAC, PPO, or TD3; add richer observations such as vision, tactile, force, and proprioception; and use simulation environments designed for contact-rich manipulation such as MuJoCo, Isaac Gym, or ManiSkill.

### Q: How would a VLA or foundation policy fit into this?

A VLA could handle high-level task understanding and visual-language grounding. RL could handle low-level contact correction and reward optimisation. This project has a similar structure: EEG classifier provides high-level intent, RL performs low-level closed-loop control.

## 13. Failure-Mode Questions

### Q: What happens if EEG classification is wrong?

The initial intent cue may be wrong, but the RL controller still observes current position and target distance. It can correct over future steps. This is why the system is robust to imperfect classification.

### Q: What are the main failure modes?

- EEG classifier confusion between classes.
- Subject shift in EEG distribution.
- Repeated oscillatory movements.
- Boundary saturation.
- Poor reward shaping.
- Hardware joint limits.
- Mismatch between simulation abstraction and real arm dynamics.

### Q: How did you debug RL instability?

By monitoring:

- Learning curves.
- Reach rate.
- Mean reward.
- Trajectory plots.
- Per-direction success.
- Action histories.
- Failure cases near boundaries or with oscillation.

### Q: What if the RL agent exploits the reward?

Check trajectories and terminal conditions, not only reward. Add penalties for boundary contact, oscillation, or step count, and verify success by actual target distance.

### Q: What if the classifier is overconfident but wrong?

The EEG-aware RL design includes confidence but still keeps closed-loop state feedback. A further improvement would be confidence calibration or uncertainty-aware policy conditioning.

## 14. Implementation Questions

### Q: Which framework did you use?

Main frameworks:

- PyTorch for EEGTransformer and DQN models.
- MNE for EEG loading and event extraction.
- NumPy / SciPy for preprocessing and resampling.
- PyBullet for robotic-arm simulation.
- BrainFlow for EEG stream interface.
- Serial control for SO-101 hardware.
- Matplotlib for plotting.

### Q: What is stored in the replay buffer?

```text
state
action
reward
next_state
done
```

For the Transformer DQN, states are stored with shape:

```text
[1, state_dim]
```

because the model expects sequence format.

### Q: What is the batch input shape for DQN training?

```text
[batch_size, seq_len, state_dim]
```

Usually:

```text
[batch_size, 1, 5]
```

or:

```text
[batch_size, 1, 7]
```

### Q: What is the classifier batch input shape?

For PhysioNet 4-class:

```text
[batch_size, 1, 64, 1000]
```

### Q: How do you convert logits to prediction and confidence?

```python
probs = softmax(logits)
pred = argmax(probs)
confidence = max(probs)
```

### Q: What is the exact meaning of `confidence`?

It is the maximum softmax probability for the predicted EEG class. It is not a calibrated probability unless explicit calibration is performed.

### Q: What is the difference between `target_y, target_z` and action direction?

`target_y, target_z` define where the agent should eventually go. The action direction is the one-step movement chosen by the policy.

## 15. Things To Avoid Saying

Do not say:

> The RL model outputs the robotic arm joint angles.

Say:

> The RL model outputs Q-values over discrete movement actions, which are translated into joint increments by the control interface.

Do not say:

> EEG directly controls the robot.

Say:

> EEG provides an intent cue; RL performs closed-loop action selection.

Do not say:

> I proved full real-time human EEG control on hardware.

Say:

> I validated the offline EEG-to-RL pipeline and the online/synthetic BrainFlow + hardware-interface path. The archived online runs use synthetic board / simulation-oriented validation.

Do not say:

> The SO-101 was fully controlled in all six axes by RL.

Say:

> The hardware has 5 arm joints plus a gripper, but the RL experiment abstracts control to a 2D plane and mainly drives two joints for left/right and up/down motion.

## 16. Short Answers For Fast Interview Questions

### RL output dimension?

4 for the 4-direction DQN, 8 for the 8-direction DQN.

### RL input dimension?

5 for basic control, 7 when EEG prediction and confidence are included.

### What does DQN output?

Q-values over discrete actions.

### What does EEGTransformer output?

4-class motor-imagery logits / probabilities.

### How is an action selected?

Use `argmax` over Q-values during evaluation.

### What does `state_dim = 5` mean?

`[y, z, target_y, target_z, distance]`.

### What does `state_dim = 7` mean?

The 5D state plus `[eeg_pred, confidence]`.

### Is the action continuous?

No, it is discrete in this project.

### What are the four actions?

Left, right, up, down.

### What are the eight actions?

Left, right, up, down, up-left, up-right, down-left, down-right.

### What is the robot?

SO-101 robotic arm, controlled through serial interface; also simulated in PyBullet.

### How many arm axes?

5 arm joints plus 1 gripper servo; the project control abstraction mainly uses 2 motion degrees.

### What are the two main controlled joints?

`shoulder_pan` for left/right and `wrist_flex` for up/down.

### Why RL?

To make control closed-loop and robust to noisy EEG classification.

### Why Transformer DQN?

To compare a self-attention-based Q-network against other sequential models and support sequence-based control extensions.

### Why not PPO/SAC?

DQN matches the discrete action space used in this project. PPO/SAC would be better for continuous high-DOF dexterous manipulation.

### What is the reaching condition?

Distance to target below a fixed radius.

### What is the main limitation?

The physical-control validation is simplified and does not prove full contact-rich dexterous manipulation or live human EEG robustness.

## 17. If Asked To Draw The Pipeline

Draw:

```text
EEG epoch
  -> EEGTransformer / CTNet
  -> class prediction + confidence
  -> RL state: [y, z, target_y, target_z, distance, eeg_pred, confidence]
  -> Transformer DQN
  -> Q-values over actions
  -> argmax action
  -> PyBullet / SO-101 serial controller
  -> new arm state
  -> feedback loop
```

## 18. Best Two-Sentence Technical Answer

> My EEGTransformer maps each motor-imagery EEG epoch into a discrete intent class, while the Transformer DQN maps the current closed-loop control state into Q-values over discrete movement actions. The robotic arm is not controlled by raw EEG directly; EEG is used as a noisy high-level cue, and the RL policy uses state feedback to choose corrective actions for target reaching.

