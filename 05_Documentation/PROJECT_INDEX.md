# 3YP Project Index

## Purpose

This file is a consolidated index for the `3YP` repository. It records:

- the purpose of each main folder
- the role of key text documents
- where project progress, planning, experiment aims, and experiment results are documented
- how generated experiment outputs and trained models are organised

It is intended to complement `README.md`, not replace the detailed report.

## Where To Find Core Project Description

### Project overview

- `README.md`: top-level folder overview, major contributions, quick start, current hardware validation status

### Project aims and experiment purpose

- `01_Reports/final_report_draft_v9.tex`
  - `Aims and Objectives`
  - `Proposed Approach`

### Experiment results and discussion

- `01_Reports/final_report_draft_v9.tex`
  - `Results and Discussion`
  - includes classification, channel reduction, RL, end-to-end evaluation, and preprocessing ablation

### Project progress

- `01_Reports/week_2026_03_24.tex`: weekly progress report and supervisor-feedback response
- `01_Reports/final_report_draft_v9.tex`
  - `Current Hardware Validation Status`
  - `Project Management`

### Project planning and management

- `01_Reports/final_report_draft_v9.tex`
  - `Project Management`
- `05_Documentation/MEETING_GUIDE.md`: weekly meeting / progress presentation guidance

## Top-Level Folder Index

### `01_Reports/`

Academic and management documents for the project.

- `final_report_draft_v9.tex` / `.pdf`: latest final report draft
- `final_report_draft_v8.tex` / `.pdf`: earlier report draft
- `final_report_draft_v9_feedback.pdf`: feedback copy of the report
- `week_2026_03_24.tex` / `.pdf`: weekly progress presentation
- `architecture_diagram.tex` / `.pdf`: system architecture figure source
- `Risk assessment form-11314389_v3.docx`: risk assessment / formal project paperwork
- `ZhengXu_CV_v5.tex` / `.pdf`: English CV
- `ZhengXu_CV_v5_CN.tex` / `.pdf`: Chinese CV

Generated LaTeX helper files such as `.aux`, `.log`, `.toc`, `.fls`, and `.fdb_latexmk` are build artefacts rather than primary source documents.

### `02_Code/`

All code written or assembled for the project, grouped by function.

#### `02_Code/EEG_Classification/`

Motor imagery EEG classification models, training, evaluation, and ablation scripts.

- `CTNet_model.py`: original main CTNet / EEGTransformer-style classifier entry point for BCI datasets
- `physionet_loader.py`: PhysioNet data loading utilities
- `train_physionet_ctnet.py`: PhysioNet pooled training
- `train_physionet_4class_ctnet.py`: PhysioNet 4-class training variant
- `finetune_physionet_ctnet.py`: subject-specific fine-tuning
- `eval_physionet_ctnet.py`: evaluation utilities
- `test_physionet_ctnet.py`: test/inference script
- `physionet_loso.py`: LOSO-style PhysioNet evaluation
- `channel_reduction_study.py`: channel reduction / ablation study
- `filter_vs_ica_experiment.py`: preprocessing comparison
- `finetune_filter_comparison.py`: filtered vs unfiltered fine-tuning comparison
- `cross_validation.py`: cross-validation support

#### `02_Code/Reinforcement_Learning/`

Closed-loop control policies and evaluation for robotic reaching.

- `dqn_model.py`: baseline DQN implementation
- `dqn_transformer.py`: Transformer-based DQN implementation
- `train_dqn_rl.py`: RL training entry point
- `train_rl_4direction.py`: 4-direction control training
- `train_rl_8direction.py`: 8-direction control training
- `train_rl_8direction_optimal.py`: optimised 8-direction training variant
- `train_rl_8direction_smooth.py`: smooth-control training variant
- `rl_control_test.py`: control test / evaluation
- `ctnet_dqn_e2e_eval.py`: EEG-to-control end-to-end evaluation
- `multi_subject_sequence_control.py`: multi-subject sequence evaluation

#### `02_Code/Physical_Control/`

Scripts for serial control of the SO-101 robotic arm and live EEG-driven control integration.

- `serial_arm_env.py`, `serial_arm_env_v2.py`: serial hardware environment wrappers
- `phy_control.py`: physical arm control entry point
- `rl_physical_control.py`: RL policy control on the physical arm
- `brainflow_physical_control.py`: BrainFlow + physical arm integration
- `realtime_eeg_control.py`: real-time EEG control script
- `openbci_stream.py`: OpenBCI stream handling
- `eeg_physical_control.py`, `eeg_physical_control_v2.py`: EEG-driven physical control variants
- `serial_go_home.py`, `serial_go_home_sync.py`: return arm to home pose
- `serial_go_return.py`, `serial_go_return_sync.py`: move arm to return pose
- `serial_save_home.py`, `serial_save_return.py`: save reference poses
- `serial_mid_test.py`, `serial_probe.py`: serial diagnostics / intermediate motion tests
- `test_8dir_comprehensive.py`, `test_8dir_comprehensive_v2.py`, `test_8dir_physical.py`, `test_8dir_smooth.py`: physical control test scripts
- `serial_home.json`, `serial_return.json`: saved arm pose data

#### `02_Code/Simulation/`

PyBullet and Gym-based simulation environment for development and evaluation.

- `pybullet_arm_env.py`: PyBullet arm environment
- `arm_gym_env.py`: Gym wrapper
- `gym_control.py`: simulation control entry point
- `control_simulation.py`: simulation runner / helper
- `pb_joint_tuner.py`: joint tuning utility
- `pb_probe_urdf.py`: URDF probing utility
- `pb_set_home.py`: simulation home-pose utility

#### `02_Code/Utils/`

Shared helpers, plotting tools, and dependency list.

- `utils.py`: shared project utilities
- `requirements.txt`: Python dependencies
- `plot_training_metrics.py`: training metric plots
- `plot_dqn_training.py`: RL training plots
- `plot_baseline_results.py`: baseline plotting
- `compare_datasets.py`: dataset comparison helper
- `compare_dqn_v2.py`: DQN comparison helper

#### `02_Code/configs/`

- `phase3_example.json`: example configuration file

#### `02_Code/drivers/`

Low-level hardware driver code.

- `so101_serial.py`: SO-101 serial driver

#### `02_Code/lerobot/`

Vendored external framework subtree from Hugging Face LeRobot. This is reference / backup infrastructure and is not the main project codebase authored for `3YP`.

### `03_Experiments/`

Generated experiment outputs, summary JSON files, plots, logs, and fine-tuned checkpoints.

#### `03_Experiments/Channel_Reduction/`

Results for the electrode/channel reduction study.

- `channel_reduction/channel_reduction_summary.json`: summary metrics
- `channel_reduction/channel_reduction_comparison.png`: comparison plot
- `channel_reduction/importance/`: per-channel importance outputs
- `channel_reduction/64ch/`, `32ch/`, `16ch/`, `8ch/`, `4ch/`, `2ch/`: outputs for each channel-count setting
- `32ch/ablation_top32`, `16ch/ablation_top16`, `8ch/ablation_top8`: ablation-ranked subsets
- `32ch/domain_motor_cortex_32`, `16ch/domain_motor_cortex_16`, `8ch/domain_motor_cortex_8_openbci`, `4ch/domain_motor_cortex_4`, `2ch/domain_motor_cortex_2`: motor-cortex subset experiments

#### `03_Experiments/Filter_Ablation/`

Results for preprocessing ablations.

- `ablation_with_filter/`: checkpoints and pooled results using the 8-30 Hz filter
- `ablation_no_filter/`: checkpoints and pooled results without the filter
- `filter_ablation/filter_ablation_results.json`: main filter ablation summary
- `filter_ablation_finetune/filter_ablation_finetune_results.json`: fine-tuning comparison summary
- `filter_vs_ica/filter_vs_ica_results.json`: filter vs ICA comparison
- `ica_comparison/ica_comparison_result.json`: ICA-only comparison results

#### `03_Experiments/DQN_Training/`

Results for RL training and control testing.

- `dqn_policy_full.pth`, `dqn_state.pth`: trained policy/state weights stored in the experiment area
- `dqn_training_curve.png`: overall DQN training plot
- `rl_4direction/`: 4-direction RL results
- `rl_8direction/`: 8-direction RL results
- `rl_8direction_optimal/`: improved 8-direction variant
- `rl_8direction_smooth/`: smoother-control variant
- `rl_control_test/`: evaluation plots and `rl_control_comparison.json`
- `rl_physical_control/results.json`: physical-control evaluation summary
- `rl_control_test_log.txt`, `rl_control_test_log2.txt`: raw RL test logs
- `rl_arm.gif`, `rl_summary.png`: presentation-friendly RL visuals

#### `03_Experiments/E2E_Evaluation/`

End-to-end EEG-classification-to-control evaluation outputs.

- `ctnet_dqn_e2e/e2e_summary.json`: main end-to-end summary
- `ctnet_dqn_e2e/eeg_labels.npy`, `eeg_predictions.npy`, `eeg_confidences.npy`: saved classifier outputs for RL integration
- `multi_subject_sequence/results.json`: multi-subject sequence evaluation
- `multi_subject_sequence/*.png`: per-subject and overall plots
- `rl_control_test/`: duplicated / comparable control evaluation plots for the E2E stage

#### `03_Experiments/architecture_comparison/`

- `summary.json`: architecture comparison summary
- `comparison.png`: architecture comparison plot

#### `03_Experiments/physionet_ctnet_finetune/`

Subject-specific 64-channel fine-tuning results.

- `finetune_summary.json`: aggregate fine-tuning summary
- `finetune_comparison.png`: aggregate comparison figure
- `subject_003`, `subject_007`, `subject_009`, `subject_038`, `subject_043`, `subject_046`, `subject_048`, `subject_050`, `subject_055`, `subject_070/`
  - each subject folder contains `model_ft_sXXX.json` and `model_ft_sXXX.pth`

#### `03_Experiments/physionet_ctnet_finetune_8ch/`

Subject-specific 8-channel fine-tuning results.

- `finetune_summary.json`: aggregate 8-channel summary
- `finetune_comparison.png`: aggregate comparison figure
- `subject_003`, `subject_007`, `subject_009`, `subject_038`, `subject_048`, `subject_050`, `subject_070/`
  - each subject folder contains `model_ft_sXXX.json` and `model_ft_sXXX.pth`

#### Other experiment summary files in `03_Experiments/`

- `cv_all_subjects.json`, `cv_results.json`, `cv_subject*.json`: cross-validation outputs
- `comparison_*.png`: exported comparison figures used for analysis or presentation

### `04_Trained_Models/`

Saved trained model weights and pooled evaluation artefacts.

#### `04_Trained_Models/physionet_ctnet/`

PhysioNet model outputs for pooled training and evaluation.

- `physionet_ctnet_joint.pth`: pooled checkpoint
- `physionet_ctnet_S001.pth`: subject-specific / example checkpoint
- `physionet_ctnet_results.png`, `physionet_joint_training.png`: training visuals
- `pool/model_pool.pth`: pooled model
- `pool/model_pool.json`, `pool/metrics_pool.json`, `pool/norm_params.json`: metadata and metrics
- `pool/confusion_matrix_pool.png`, `pool/training_curve_pool.png`: evaluation plots
- `pool/eval/report_*.txt`: text evaluation reports

#### `04_Trained_Models/physionet_ctnet_109sub/`

Large-scale pooled PhysioNet training outputs.

- `checkpoint_ep50.pth`, `checkpoint_ep100.pth`, `checkpoint_ep150.pth`: intermediate checkpoints
- `pool/model_pool.pth`: pooled checkpoint
- `pool/model_pool.json`, `pool/metrics_pool.json`, `pool/norm_params.json`: metadata and metrics
- `pool/pool_per_subject.json`, `pool/pool_per_subject.png`: per-subject breakdown
- `pool/confusion_matrix_pool.png`, `pool/training_curve_pool.png`: evaluation plots
- `pool/eval/report_full_eval.txt`: text evaluation report

#### `04_Trained_Models/pretrained_models/`

Saved pretrained CTNet models for benchmark datasets.

- `2a/`: BCI Competition IV-2a subject models
- `2b/`: BCI Competition IV-2b subject models
- `new/`: additional model set plus `pred_true.xlsx`, `process_train.xlsx`, `result_metric.xlsx`

#### Standalone RL model files

- `dqn_policy_full.pth`
- `dqn_state.pth`
- `test_dqn.pth`

### `05_Documentation/`

Human-readable guidance documents, images, and demo media.

- `CTNet_使用指南.md`: CTNet model usage guide
- `参数修改指南.md`: training parameter modification guide
- `数据集划分说明.md`: dataset split explanation
- `训练功能说明.md`: training feature / early stopping explanation
- `MEETING_GUIDE.md`: weekly meeting presentation guide
- `architecture.png`, `overall system diagram.png`: architecture figures
- `gym_arm.gif`, `gym_summary.png`: simulation demo media
- `rl_arm.gif`, `rl_summary.png`: RL demo media
- `phy_control.webm`, `phy_control_arm.mp4`: physical-arm demonstration videos

### `06_Data/`

Datasets and labels used by the project.

#### `06_Data/BCICIV_2a_gdf/`

BCI Competition IV-2a raw `.gdf` recordings.

- naming pattern: `A01T.gdf`, `A01E.gdf`, ..., `A09T.gdf`, `A09E.gdf`
- `T` denotes training session, `E` denotes evaluation session

#### `06_Data/BCICIV_2b_gdf/`

BCI Competition IV-2b raw `.gdf` recordings.

- naming pattern: `B0101T.gdf` ... `B0905E.gdf`

#### `06_Data/true_labels/`

Label files corresponding to benchmark datasets.

- `A*.mat`: IV-2a labels
- `B*.mat`: IV-2b labels
- `2b/`: extra label-related subfolder

#### `06_Data/physionet_raw/`

Raw PhysioNet EEGMMIDB data stored by subject folder.

- current repository contents include `S001` to `S050`
- each subject folder contains multiple `SXXXRYY.edf` files plus matching `.edf.event` files
- `physionet_3sub.mat` and `physionet_10sub.mat`: smaller prepared subsets
- `robots.txt`: source-site metadata snapshot

#### `06_Data/MNE-eegbci-data/`

MNE-downloaded EEGBCI cache / dataset mirror.

### `07_References/`

Reference material retained for provenance and comparison.

- `README.md`: notes on references
- `Original_CTNet/README.md`: original upstream CTNet description
- `Original_CTNet/ctnet.py`: original reference implementation
- `Original_CTNet/LICENSE`: upstream licence
- `Original_CTNet/architecture.png`: original architecture figure

## Text Document Index

The main human-authored text documents in this repository are:

- `README.md`
- `01_Reports/final_report_draft_v8.tex`
- `01_Reports/final_report_draft_v9.tex`
- `01_Reports/week_2026_03_24.tex`
- `01_Reports/architecture_diagram.tex`
- `01_Reports/ZhengXu_CV_v5.tex`
- `01_Reports/ZhengXu_CV_v5_CN.tex`
- `01_Reports/Risk assessment form-11314389_v3.docx`
- `05_Documentation/CTNet_使用指南.md`
- `05_Documentation/参数修改指南.md`
- `05_Documentation/数据集划分说明.md`
- `05_Documentation/训练功能说明.md`
- `05_Documentation/MEETING_GUIDE.md`
- `07_References/README.md`
- `07_References/Original_CTNet/README.md`
- `04_Trained_Models/physionet_ctnet/pool/eval/report_train_set.txt`
- `04_Trained_Models/physionet_ctnet/pool/eval/report_unseen.txt`
- `04_Trained_Models/physionet_ctnet_109sub/pool/eval/report_full_eval.txt`
- `03_Experiments/DQN_Training/rl_control_test_log.txt`
- `03_Experiments/DQN_Training/rl_control_test_log2.txt`

## Recommended Maintenance Rule

When adding a new major folder, script, or report, update:

1. `README.md` for top-level visibility
2. this file for repository indexing
3. the final report or weekly report if the change affects aims, progress, results, or conclusions
