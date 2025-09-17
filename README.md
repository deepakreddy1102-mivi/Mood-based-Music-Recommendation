# Mood-based Music Recommendation using Physiological Signals

**Goal.** Predict a user's *Mood* from physiological signals (Heart Rate, Skin Temperature, Blink Rate, Time of Day) and map it to a music genre.

## Pipeline
EDA → cleaning (median/mode + IQR cap) → 80/20 stratified split → preprocessing (scaler + OHE) → models (LogReg, DT, RF, GB) with 5-fold CV → test evaluation → mood→genre rule → artifacts.

## Key results & iNSIGHTS

- **Test accuracy:** 0.8367 on a stratified 20% hold-out.
- **Per-class F1:** Happy 0.899, Relaxed 0.732, Sad 0.779, Stressed 0.944.
- **Top features (RandomForest):** Blink Rate, Heart Rate, Skin Temperature (Time of Day one-hots are minor).
- **Confusions mainly between:** Relaxed ↔ Sad (human-plausible overlap).
- **Deliverables included:** cleaned dataset (`data_processed/`), trained model (`reports/mood_music_model.joblib`), metrics (`reports/metrics.json`, `overall_metrics.csv`, `per_class_metrics.csv`), figures (`reports/confusion_matrix.png`, `reports/feature_importance.png`), batch predictions + genres (`reports/test_predictions_with_genres.csv`), and notebook export (`reports/notebook_export.html`).


## Files
- Raw: `data_raw/mood_music_dataset.csv`
- Cleaned: `data_processed/mood_music_cleaned.csv`
- Model: `reports/mood_music_model.joblib`
- Metrics: `reports/metrics.json`
