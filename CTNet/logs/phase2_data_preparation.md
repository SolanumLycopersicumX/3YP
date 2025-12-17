## Data preparation checkpoint (Phase 2)

- **Date**: 2025-11-01
- **Script**: `./bd/bin/python prepare_dataset.py`
- **Source data**: `BCICIV_2a_gdf/A0xT/E.gdf`, labels in `true_labels/A0xT/E.mat`
- **Output**: `mymat_raw/A0xT.mat` & `A0xE.mat` regenerated for subjects A01–A09  
  Each file contains `data` of shape `(288, 22, 1000)` and `label` of shape `(288, 1)`.
- **Warnings observed**: Duplicate channel names auto-renamed by MNE; expected for BCI IV-2a and does not affect saved arrays.
- **BCI IV-2b**: `preprocess_2b.py` updated to write into `mymat_raw/` and executed with `BCICIV_2b_gdf/B0x0yT/E.gdf` & `true_labels/B0x0yT/E.mat`.  
  Output files `mymat_raw/B0xT.mat` (≈400×3×1000) and `B0xE.mat` (≈320×3×1000) verified for subjects B01–B09.
- **Notes**: MNE warns about highpass/lowpass swap in raw metadata; acknowledged, affects only logged message.

