# EEG and Arm Observer Dashboard

This local Streamlit dashboard shows the closed-loop BCI control pipeline in one screen:

- raw EEG versus 8-30 Hz preprocessed EEG
- PyBullet robotic-arm simulation frame
- Y-Z end-effector trajectory
- CTNet class probabilities, predicted class, true label when available, and executed action

## Install

From the repository root:

```bash
pip install -r 02_Code/Utils/requirements.txt
```

## Start

Run from the repository root:

```bash
streamlit run 02_Code/Visualization/dashboard_app.py
```

By default, the dashboard loads `04_Trained_Models/physionet_ctnet_109sub/pool/model_pool.pth`. You can override the model checkpoint path in the sidebar.

## Offline PhysioNet Mode

Use `Offline PhysioNet` to replay local PhysioNet epochs. The dashboard loads epochs through `02_Code/EEG_Classification/physionet_loader.py`, displays raw and preprocessed waveforms, runs CTNet inference, and drives the PyBullet arm from the CTNet-derived action.

Controls:

- `Subject`: PhysioNet subject id
- `Start epoch` and `Stop epoch`: ordered replay range
- `Step`: advance by one epoch
- `Reset`: return to the first selected epoch and clear the trajectory

## BrainFlow Synthetic Mode

Use `BrainFlow synthetic` to test the online data path without EEG hardware. The dashboard displays CTNet inference on the synthetic EEG stream. Synthetic EEG has no ground-truth motor-imagery label, so the classification panel marks true label as unavailable.

When `Use scripted demo action in synthetic mode` is enabled, the arm follows a stable scripted sequence while the CTNet prediction remains visible separately. The panel labels the executed action source as either `scripted demo` or `CTNet prediction`.

## Exports

The `Export log` button writes:

- `03_Experiments/Visualization/dashboard_run.jsonl`
- `03_Experiments/Visualization/dashboard_run.csv`

## Limitations

- The dashboard is a local observer and demo tool.
- It does not control the physical arm.
- It does not train or fine-tune CTNet.
- BrainFlow synthetic mode is not live human EEG validation.
