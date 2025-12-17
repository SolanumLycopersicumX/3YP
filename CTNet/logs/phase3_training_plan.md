## Training plan (Phase 3 kick-off)

- **Environment**: `conda activate /home/tomato/A-Brain-Computer-Interface-Control-System-Design-Based-on-Deep-Learning/CTNet/bd`
- **GPU check (host shell)**:
  ```bash
  python - <<'PY'
  import torch
  print(torch.__version__, torch.cuda.is_available())
  if torch.cuda.is_available():
      print(torch.cuda.get_device_name(0))
  PY
  ```
- **Subject-specific (BCI IV-2a default)**:
  ```bash
  CTNET_TYPE=A \
  CTNET_EVALUATE_MODE=LOSO-No \
  CTNET_EPOCHS=1000 \
  CTNET_N_SUBJECT=9 \
  ./bd/bin/python main_subject_specific.py
  ```
  - For a quick smoke test: reduce epochs (`CTNET_EPOCHS=5`) or limit subjects (`CTNET_N_SUBJECT=1`).
  - Outputs: `CTNet_A_heads_<H>_depth_<D>_*` folder containing `result_metric.xlsx`, `process_train.xlsx`, `pred_true.xlsx`, and checkpoints.

- **Subject-specific (BCI IV-2b)**:
  ```bash
  CTNET_TYPE=B \
  CTNET_EVALUATE_MODE=LOSO-No \
  CTNET_EPOCHS=1000 \
  CTNET_N_SUBJECT=9 \
  ./bd/bin/python main_subject_specific.py
  ```
  - Ensure `numberClassChannel('B')` picks the correct channel count (3 channels).

- **Cross-subject (LOSO) sweep**:
  ```bash
  CTNET_TYPE=A \
  CTNET_EVALUATE_MODE=LOSO \
  CTNET_EPOCHS=1000 \
  ./bd/bin/python main_subject_specific.py
  ```
  (Repeat for `CTNET_TYPE=B`.)

- **Monitoring tips**:
  - Training prints per-subject accuracy and saves intermediate Excel logs.
  - Use `watch -n 30 nvidia-smi` in another terminal to monitor GPU load/VRAM.
  - If CUDA OOM occurs, lower `CTNET_N_AUG`, batch size (set inside `ExP`), or heads/depth.

- **Post-run checklist**:
  - Record best accuracies and kappa from `result_metric.xlsx`.
  - Copy summary metrics into a new report (`logs/results_subject_specific.md`) for comparison with the paper targets.

- **Automation helpers**:
  - 批量运行：在仓库根目录执行  
    ```bash
    ./bd/bin/python scripts/run_phase3_experiments.py \
      --config configs/phase3_example.json
    ```  
    或直接指定单次实验参数：  
    ```bash
    ./bd/bin/python scripts/run_phase3_experiments.py \
      --type A --evaluate-mode LOSO-No --epochs 1000 \
      --n-subject 9 --heads 2 --depth 6 --run-tag full_a
    ```  
    运行记录追加到 `logs/phase3_runs.csv`，每个结果目录会生成 `run_metadata.json`。
  - 自动汇总：在上述命令后追加 `--auto-summarize`（默认统计 `A_heads_*`、`B_heads_*`、`CTNet_*`）。
    可借助 `--summary-pattern 'A_heads_*' --summary-pattern 'B_heads_*' \
      --summary-output logs/phase3_summary.csv --summary-group-by type evaluate_mode`
    自定义匹配范围与输出位置。
  - 清理旧目录：若需自动归档早期遗留的空指标目录，可加入  
    `--cleanup-empty --cleanup-pattern 'CTNet_*' --cleanup-archive archive/old_runs`。
    被归档的目录会按时间戳移动到指定位置，防止干扰汇总。
  - 指标汇总：  
    ```bash
    ./bd/bin/python scripts/summarize_results.py \
      --pattern 'CTNet_A*' --output logs/phase3_summary_A.csv
    ```  
    支持 `--group-by type evaluate_mode` 统计平均值，或加 `--include-empty` 查看缺失指标的目录。
