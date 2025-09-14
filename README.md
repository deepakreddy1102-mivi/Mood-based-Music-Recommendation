# Mood-based Music Recommendation using Physiological Signals

**Goal.** Predict a user's *Mood* from physiological signals (Heart Rate, Skin Temperature, Blink Rate, Time of Day) and map it to a music genre.

## Setup
Run these commands in **Command Prompt** (Windows):

    cd /d "D:\Data Science - py\mood-music"
    ".venv\Scripts\activate"
    jupyter lab

Kernel: **Python (mood-music)**

## How to run
Open `notebooks/01_mood_music_pipeline.ipynb` and run cells top to bottom (or use **Kernel → Restart & Run All**).

## Pipeline
EDA → cleaning (median/mode + IQR cap) → 80/20 stratified split → preprocessing (scaler + OHE) → models (LogReg, DT, RF, GB) with 5-fold CV → test evaluation → mood→genre rule → artifacts.

## Key results
- Final model: RandomForestClassifier
- Test accuracy: **0.8367**
- Reports:
  - `reports/per_class_metrics.csv`
  - `reports/overall_metrics.csv`
  - `reports/confusion_matrix.png`
  - `reports/feature_importance.png`

## Files
- Raw: `data_raw/mood_music_dataset.csv`
- Cleaned: `data_processed/mood_music_cleaned.csv`
- Model: `reports/mood_music_model.joblib`
- Metrics: `reports/metrics.json`
