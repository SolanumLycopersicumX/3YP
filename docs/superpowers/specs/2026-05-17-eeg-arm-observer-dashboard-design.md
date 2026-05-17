# EEG and Arm Observer Dashboard Design

Date: 2026-05-17

## Purpose

The project currently has simulation, EEG preprocessing, classification, and control components, but no unified observer interface that shows the full closed-loop process at once. This feature adds a local Streamlit dashboard that simultaneously displays:

- raw EEG versus preprocessed EEG
- PyBullet robotic-arm simulation frames
- end-effector trajectory in the Y-Z plane
- CTNet classification result, probabilities, and mapped control action

The dashboard is for experiment inspection, debugging, and dissertation/demo evidence. It does not replace the existing training scripts or physical-arm control scripts.

## Approved Scope

The dashboard will support two input modes:

1. Offline PhysioNet replay
   - Load local PhysioNet EEGMMIDB EDF data from `06_Data/physionet_raw/`.
   - Select subject and trial/epoch.
   - Show the raw epoch, filtered epoch, CTNet prediction, true label, mapped action, PyBullet arm frame, and trajectory.

2. BrainFlow synthetic streaming
   - Use the existing `OpenBCIStream(board_type="synthetic")` path.
   - Show raw and filtered rolling EEG windows.
   - Run CTNet inference on the processed synthetic epoch.
   - Also support a scripted demo action sequence for stable arm movement.
   - Clearly label that synthetic EEG has no ground-truth motor-imagery label.

The default layout is the approved balanced lab console:

- top left: PyBullet simulation viewport
- top right: raw versus preprocessed EEG waveform view
- bottom left: Y-Z trajectory plot
- bottom right: classification probabilities, predicted class, true label when available, mapped action, and data-source status

## Non-Goals

- No new model training.
- No changes to CTNet architecture.
- No live human EEG validation.
- No physical-arm control from this dashboard in the first version.
- No full web product or remote deployment. This is a local research tool.

## Architecture

Create a new visualization package:

```text
02_Code/Visualization/
+-- README.md
+-- dashboard_app.py
+-- data_sources.py
+-- eeg_pipeline.py
+-- arm_visualizer.py
+-- plotting.py
```

### `dashboard_app.py`

Streamlit entry point. Owns:

- sidebar controls
- run mode selection
- playback and stepping controls
- page layout
- session state
- export buttons

Default command:

```bash
streamlit run 02_Code/Visualization/dashboard_app.py
```

### `data_sources.py`

Defines a common interface for dashboard input sources.

Offline source:

- loads PhysioNet epochs through the existing `EEG_Classification/physionet_loader.py`
- returns raw epoch, label, subject id, epoch index, sampling rate, and metadata

Synthetic source:

- wraps `Physical_Control/openbci_stream.py`
- returns rolling raw EEG windows and stream metadata
- marks labels as unavailable

### `eeg_pipeline.py`

Owns EEG processing and classification:

- preserves raw EEG for display
- applies 8-30 Hz bandpass filtering
- resamples to the model input length when needed
- pads/trims channels when needed
- normalizes using the selected CTNet model metadata
- loads `04_Trained_Models/physionet_ctnet_109sub/pool/model_pool.pth`
- returns prediction, class probabilities, confidence, and mapped action

### `arm_visualizer.py`

Owns simulation state:

- creates `PyBulletArmEnv` in `rgb_array` mode
- applies one action per dashboard step
- stores current Y-Z position and trajectory history
- returns the latest rendered frame
- can fall back to trajectory-only display if PyBullet initialization fails

### `plotting.py`

Contains pure plotting helpers:

- raw versus filtered EEG traces
- stacked channel view
- Y-Z trajectory plot
- class probability bars

Keeping plotting code separate keeps the Streamlit page thin and easier to test.

## Data Model

Use a shared frame structure for both modes:

```python
@dataclass
class DashboardFrame:
    mode: str
    raw_eeg: np.ndarray
    filtered_eeg: np.ndarray
    sampling_rate: float
    channel_names: list[str]
    pred_class: int | None
    pred_name: str | None
    probabilities: np.ndarray | None
    confidence: float | None
    true_label: int | None
    true_name: str | None
    action: int | None
    action_name: str | None
    arm_rgb: np.ndarray | None
    trajectory_yz: list[tuple[float, float]]
    status: dict[str, Any]
```

The UI renders this structure without knowing which data source produced it.

## Offline PhysioNet Flow

1. User selects `Offline PhysioNet`.
2. User selects subject and epoch/trial.
3. Source loads raw epoch.
4. Pipeline creates filtered copy with 8-30 Hz bandpass.
5. Pipeline resamples to 1000 samples if needed.
6. Pipeline normalizes with model metadata from `model_pool.json` or `norm_params.json`.
7. CTNet returns probabilities and predicted class.
8. Class maps to action:
   - `Left` -> left
   - `Right` -> right
   - `Hands/Up` -> up
   - `Feet/Down` -> down
9. PyBullet executes the action and renders a frame.
10. UI updates all four panels.

## Synthetic Flow

1. User selects `BrainFlow synthetic`.
2. Dashboard starts `OpenBCIStream(board_type="synthetic")`.
3. Each step gets a rolling raw EEG epoch.
4. Pipeline filters, adapts, normalizes, and runs CTNet inference.
5. UI displays CTNet prediction and probabilities.
6. If scripted demo is enabled, mechanical-arm action comes from the scripted sequence.
7. If scripted demo is disabled, mechanical-arm action comes from CTNet prediction.
8. UI displays a persistent note: synthetic EEG has no ground-truth MI label.

This keeps the demonstration honest: real model inference is visible, but stable demo movement is still available.

## UI Controls

Sidebar controls:

- mode: `Offline PhysioNet` or `BrainFlow synthetic`
- subject and epoch selector for offline mode
- start, pause, step, reset
- playback speed
- channel count: 8, 16, 32, 64
- channel selection
- scripted demo action toggle for synthetic mode
- model path override
- export current snapshot/log

Main panels:

- PyBullet simulation viewport
- raw versus filtered EEG waveform panel
- Y-Z trajectory panel
- classification and action panel

Default EEG display:

- show 8 channels for readability
- allow expanding to more channels
- use consistent time axis for raw and filtered signals

## Export Outputs

The dashboard should support lightweight exports:

- per-step JSON/CSV log with mode, prediction, probabilities, action, position, and timestamps
- optional PNG snapshot for plots
- optional saved PyBullet frame for evidence

Outputs should go under `03_Experiments/Visualization/` by default.

## Error Handling

The dashboard must fail visibly and narrowly:

- missing `streamlit`: document install command in README
- missing `brainflow`: disable synthetic mode with a clear warning
- missing `mne`: disable offline PhysioNet mode with a clear warning
- missing model file: show path and let user override it
- missing EDF data: show expected directory and available alternatives
- PyBullet failure: continue with EEG, classification, and trajectory-only state
- synthetic mode labels: always show that ground-truth labels are unavailable

## Testing and Verification

Unit-level checks:

- EEG preprocessing preserves expected shapes.
- Raw and filtered arrays have aligned channel/time dimensions.
- Class-to-action mapping is correct.
- `DashboardFrame` can be created for offline and synthetic-style data.

Smoke checks:

- Offline mode loads one PhysioNet epoch.
- CTNet inference returns four probabilities.
- PyBullet returns one RGB frame or a controlled fallback.
- Trajectory updates after one action.

Manual demo checks:

- Start Streamlit dashboard.
- Confirm all four panels are visible at once.
- Offline mode shows true label and predicted label.
- Synthetic mode shows CTNet prediction plus scripted demo option.
- Synthetic mode clearly states that no ground-truth MI label is available.

## Documentation Updates

Add `02_Code/Visualization/README.md` with:

- purpose
- dependencies
- startup command
- offline mode usage
- synthetic mode usage
- known limitations
- export output paths

Update the top-level `README.md` quick start with the dashboard command once implementation is complete.
