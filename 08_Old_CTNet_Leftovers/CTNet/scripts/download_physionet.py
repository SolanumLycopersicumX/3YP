#!/usr/bin/env python3
"""
Download PhysioNet EEGMMIDB subjects S051-S109 (Motor Imagery runs only).
Already downloaded: S001-S050 (in physionet_raw/ and MNE cache).

Usage:
    python scripts/download_physionet.py

After download, run training:
    python scripts/train_physionet_ctnet.py --subjects $(seq 1 109) --mode pool \
        --epochs 500 --batch-size 128 --patience 80 \
        --output-dir outputs/physionet_ctnet_109sub
"""

import sys
import time

try:
    from mne.datasets import eegbci
except ImportError:
    print("ERROR: MNE not installed. Run: pip install mne")
    sys.exit(1)

MI_RUNS = [4, 6, 8, 10, 12, 14]

SUBJECTS = list(range(51, 110))

print("=" * 60)
print(f"  Downloading PhysioNet EEGMMIDB: S051-S109")
print(f"  Runs: {MI_RUNS} (Motor Imagery)")
print(f"  Total: {len(SUBJECTS)} subjects")
print("=" * 60)

success = 0
failed = []
t0 = time.time()

for i, sub in enumerate(SUBJECTS):
    print(f"\n[{i+1}/{len(SUBJECTS)}] Subject S{sub:03d}...", end=" ", flush=True)
    try:
        eegbci.load_data(sub, MI_RUNS)
        print("OK")
        success += 1
    except Exception as e:
        print(f"FAILED: {e}")
        failed.append(sub)

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"  Done in {int(elapsed//60)}m {int(elapsed%60)}s")
print(f"  Success: {success}/{len(SUBJECTS)}")
if failed:
    print(f"  Failed:  {failed}")
print(f"{'=' * 60}")
