# Experiment Result Index

This folder contains both the final report evidence files and several earlier exploratory outputs that were kept for reproducibility. To avoid ambiguity, use the files below as the authoritative sources for the dissertation figures and numbers.

## Final report result sources

- RL architecture comparison:
  `architecture_comparison_v2/summary_v2.json`
  `architecture_comparison_v2/comparison_v2.png`

- BCI Competition IV dataset comparison:
  `dataset_comparison/dataset_comparison_log.txt`
  `dataset_comparison/results_2a.json`
  `dataset_comparison/results_2b.json`

- PhysioNet fine-tuning:
  `physionet_ctnet_finetune/finetune_summary.json`

- 8-channel PhysioNet fine-tuning:
  `physionet_ctnet_finetune_8ch/finetune_summary.json`

- Channel reduction:
  `Channel_Reduction/channel_reduction/channel_reduction_summary.json`
  `Channel_Reduction/channel_reduction/importance/ablation_top_channels.json`

- End-to-end EEG + RL evaluation:
  `E2E_Evaluation/ctnet_dqn_e2e/e2e_summary.json`

- BrainFlow / online-control integration evidence:
  `brainflow_physical_control/summary.json`
  `realtime_eeg_control/realtime_control_sim.json`
  Note: these archived runs use the synthetic BrainFlow board and simulation-oriented validation, not live human OpenBCI recordings.

## Older or alternate outputs retained in the repo

- `architecture_comparison/summary.json`
  Earlier RL comparison output. Retained for history, but not the version cited in the final report.

- `dataset_comparison/summary.json`
  Alternate summary file that does not match the final report numbers. Use `dataset_comparison_log.txt` and the per-dataset result files for report traceability.

- `dataset_comparison/comparison_results.json`
  Early exploratory comparison on a reduced subset. Retained for completeness, not as the report source.
