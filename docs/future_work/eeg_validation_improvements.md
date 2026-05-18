# Future Work: EEG Validation Improvements

This note records two validation issues identified during interview preparation and defines the follow-up experiments needed to make the claims stronger.

## 1. Subject-Specific Fine-Tuning Evaluation

### Issue

The current PhysioNet fine-tuning results support subject-specific calibration, not subject-independent generalisation.

The fine-tuning scripts evaluate one participant at a time using 5-fold stratified cross-validation over that participant's epochs. This is a valid within-subject calibration experiment, but it does not prove performance on entirely unseen subjects. The current loader also returns only `data` and `labels`, without run/session/trial metadata, so it cannot enforce grouped splits by session or run.

### Future Experiment

Implement a stricter new-user calibration protocol:

1. Hold out target subjects from pooled pre-training.
2. For each held-out subject, split their data into calibration and test partitions.
3. Fine-tune only on the calibration partition.
4. Evaluate on held-out trials, preferably grouped by run/session when metadata is available.
5. Report this separately from the existing within-subject 5-fold CV results.

### Required Code Changes

- Extend the PhysioNet loader to return run/session/trial identifiers alongside `data` and `labels`.
- Add grouped split support, for example by subject for cross-subject evaluation and by run/session for within-subject future-session evaluation.
- Save split metadata in experiment outputs so the exact evaluation protocol is auditable.

### Claim Boundary

Until this is completed, the fine-tuning claim should be phrased as:

> Pretraining plus subject-specific fine-tuning improves personalised decoding on held-out trials from the same participant.

It should not be phrased as:

> Fine-tuning proves generalisation to unseen subjects.

## 2. EEG-Driven DQN Control Evaluation

### Issue

The current DQN control result validates a strong simulated reaching policy, but it does not fully prove that the policy compensates for EEG classification errors.

In the simplified reaching environment, the DQN can observe the target position directly. As a result, the policy can solve the task from arm state and target state alone. The EEG-augmented result shows that EEG predictions can be appended as features, and may accelerate training, but it does not prove that EEG evidence is necessary or that the DQN corrects EEG decoding errors.

### Future Experiment

Use the true motor-imagery label to define the intended target, and use the EEG classifier output only as a noisy observation.

For each EEG trial:

1. Use the true label as the intended movement direction.
2. Generate the target from the true label, not from the predicted label.
3. Feed the classifier prediction and confidence into the control policy as noisy evidence.
4. Evaluate whether the controller reaches the true-label target when the classifier prediction is wrong.

### Baselines

Compare at least four settings:

1. **Open-loop EEG-to-action:** directly execute the EEG-predicted action.
2. **DQN with target only:** current arm state plus true target position, no EEG evidence.
3. **DQN with noisy EEG only:** arm state plus EEG prediction/confidence, with target hidden or represented only through the intended label stream.
4. **DQN with target and noisy EEG:** target state plus EEG prediction/confidence.

Add control checks:

- Oracle EEG labels.
- Real EEG predictions.
- Shuffled EEG predictions.
- Synthetic error rates, for example 0%, 10%, 20%, 30%, and 40%.

### Success Criteria

The claim that DQN mitigates EEG errors is supported only if:

- open-loop EEG-to-action degrades with classifier errors;
- DQN with noisy EEG reaches the true-label target more often than open-loop control;
- shuffled EEG weakens or removes the benefit;
- the analysis reports performance specifically on trials where the classifier prediction is wrong.

### Claim Boundary

Until this is completed, the DQN result should be phrased as:

> Transformer DQN achieved near-perfect control in a simplified closed-loop reaching simulation. EEG predictions were tested as additional state features, but the current experiment does not prove that the policy depends on EEG or compensates for EEG decoding errors.

It should not be phrased as:

> The 99% reach rate proves that DQN corrects EEG classification errors.

