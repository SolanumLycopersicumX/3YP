## Environment checkpoint (Phase 1)

- **Date**: 2025-11-01
- **Machine**: NVIDIA GeForce RTX 4070 Ti (driver 580.95.05, CUDA 13.0)  
  ```
  $ nvidia-smi
  ```

- **Environment path**: `/home/tomato/A-Brain-Computer-Interface-Control-System-Design-Based-on-Deep-Learning/CTNet/bd`
- **Python**: 3.10.19 (`conda create -p ./bd python=3.10`)
- **Package sources**:  
  - Numerical stack installed via `conda -c conda-forge`: `numpy==1.23.5`, `scipy==1.10.1`, `pandas==1.5.3`, `scikit-learn==1.2.2`, `h5py==3.8.0`, `matplotlib==3.7.3`, `tqdm==4.65.0`  
  - CUDA-enabled PyTorch via `conda -c pytorch -c nvidia`: `pytorch==1.13.1`, `torchvision==0.14.1`, `torchaudio==0.13.1`, `pytorch-cuda=11.7`  
  - Extras from pip: `mne==1.5.1`, `einops==0.6.1`, `torchsummary==1.5.1`, `openpyxl==3.1.2`, `opencv-python==4.8.1.78`

- **Verification**:
  ```
  $ ./bd/bin/python - <<'PY'
  import numpy, pandas, scipy, sklearn, mne, torch
  print('versions ok')
  print('torch', torch.__version__, torch.cuda.is_available())
  PY
  ```
  (Inside Codex harness CUDA returns `False` because GPU access is sandboxed.  
  On the host machine run the same check after `conda activate /.../bd`; expect `torch 1.13.1+cu117 True` and device name `NVIDIA GeForce RTX 4070 Ti`.)

- **Usage**:
  ```
  $ conda activate /home/tomato/.../CTNet/bd
  $ python main_subject_specific.py  # with CTNET_* overrides as needed
  ```
