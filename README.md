# Eyes-Open vs. Eyes-Closed Relative Alpha Power Classification - Live Test

## 1. General Overview

- Demonstrates the functionality of the g.tec Nautilus Research 8ch EEG cap for a real-time
  BCI-style classification task.
- The 4 parietal + occipital channels (P3, P4, O1, O2) are extracted, and relative alpha power
  (alpha band power / broadband power, log-scaled) is calculated as a single feature per channel,
  both for the pretraining dataset (GIPSA "EEG Alpha Waves") and for live test data.
- A **logistic regression** model is pretrained on GIPSA and used for live classification.
  Logistic regression was chosen over a deep-learning approach (e.g. EEGNet):
  alpha-blocking is a well-established, strongly separable univariate effect, so a linear model
  is expected to reach comparable accuracy while staying fully interpretable (per-channel weights and
  contributions can be inspected directly - see `demo_visual.py`).
- Since the GIPSA training hardware differs from the g.tec setup used here, live features are
  normalized against a short per-subject baseline recording (`calibration.py`) rather than
  against the raw GIPSA feature scale - see the "Known limitations" section below.

## 2. Setup

```bash
pip install -r requirements.txt --break-system-packages
```

Requires Simulink to be actively streaming EEG data via LSL before running  `calibration.py`/
`live_classify.py` ==> a working LSL connection with expected stream name `EEG_measurement_data_stream`,
28 channels (see `config.py`'s `CHANNEL_GROUPS` for the full layout), 250 Hz.

## 3. File Structure

- **Folders:**
    - `GIPSA_features_analysis/`: training features and plots from `analyze_features.py` &
      `analyze_psd_gipsa.py`
    - `recordings/`: subject-specific recording data (calibration raw data & features, live
      session CSV logs), plots from `analyze_features.py` / `analyze_psd_calibration.py`
    - `testing/`: various testing/diagnostic scripts, notably `test_lsl_stream.py`,
      `diagnose_lsl.py`, `diagnose_rate_over_time.py`; (future: `test_multi_device.py`)

- **Scripts** (sorted by run order / importance):
    - **Run-time:**
        - `config.py`: central settings and constants used by almost all other scripts
        - `relative_alpha_power_features.py`: extracts the relative alpha power feature
          (log-scaled and, downstream, baseline z-scored)
        - `pretrain_gipsa.py`: run once to download the GIPSA dataset, extract features, and
          train the model
        - `lsl_stream.py`: manages the incoming LSL stream from Simulink/the EEG recording
          (future: `multi_device_lsl.py` for additional devices, e.g. eye tracking, GSR)
        - `calibration.py`: collects a short baseline calibration recording for a new subject
          before a live test
        - `live_classify.py`: runs live classification and publishes predictions via an LSL
          marker outlet for other applications/ adaptive systems to further utilize 
          classification outputs
        - `demo_visual.py`: opens a live-updating multi-panel plot during a live test; run in a
          second terminal alongside `live_classify.py`

    - **Post-analysis:**
        - `analyze_features.py`: channel-wise eyes-open-vs-closed comparison plots, for either
          GIPSA or calibration data; also supports unsupervised exploration of an unlabeled
          `live_session_log.csv`
        - `psd_analysis.py`: shared PSD computation/plotting helpers - not run directly, imported
          by the two scripts below
        - `analyze_psd_gipsa.py`: PSD comparison plots for a single GIPSA subject or the grand
          average across all subjects
        - `analyze_psd_calibration.py`: PSD comparison plots for a single calibration session or
          the grand average across all recorded sessions

- **Other:**
    - `alpha_classifier.joblib`: trained logistic regression model (from `pretrain_gipsa.py`) -
      regenerable, not meant to be hand-edited
    - `gipsa_training_features.npz`: cached training features (from `pretrain_gipsa.py`) - also
      regenerable
    - `requirements.txt`: Python packages needed to run all scripts

## 4. Run-Time Order / Commands (+examples)

If you've never run this before, start from step 1. Otherwise, start from step 2.

**Initial download and training on GIPSA data:**
```bash
python pretrain_gipsa.py
```

**Start a live test** (Simulink must already be streaming):
```bash
python calibration.py
python live_classify.py
python demo_visual.py   # in a second terminal
```
Press Ctrl+C in each terminal to stop once finished.

**Post-experiment analysis:**
```bash
# per-channel condition comparison (with example path, enter respective path of interest)
python analyze_features.py live recordings/subject_08/2026-09-21_131657/live_session_log.csv 
python analyze_features.py calibration recordings/subject_08/2026-09-21_131657
python analyze_features.py training gipsa_training_features.npz

# PSD comparison (with example path, enter respective path of interest)
python analyze_psd_calibration.py session recordings/subject_01/2026-09-18_143022
python analyze_psd_calibration.py all
python analyze_psd_gipsa.py subject 3
python analyze_psd_gipsa.py all
```

## 5. Known Limitations / Open Questions

- g.tec's OSCAR artifact-removal algorithm is proprietary/ patented; its exact frequency response
  (e.g. whether/how it attenuates low-frequency content) is undocumented/ disclosed, so GIPSA's raw,
  unfiltered training signal and the OSCAR-filtered live signal are not perfectly comparable
  preprocessing-wise. However, this trade-off is taken as a measure to minimize artifacts within the EEG.
- Per-subject baseline z-scoring (rather than a global scaler) was chosen to compensate for
  hardware/electrode differences between GIPSA and this setup, but this has not yet been
  validated against real (non-table-test) recordings.
- All pipeline testing so far has used the EEG cap resting on a table (no real scalp contact) -
  results discussed in the codebase/analysis so far reflect electrical artifacts, not verified
  alpha-blocking.