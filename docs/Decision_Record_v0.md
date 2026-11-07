# Decision Record v0: Predictive Maintenance (AML 3404, Fall 2026)

Status: DRAFT for team review. No decision is final until the team confirms it.

## 1. Starting point

- Baseline: `CapstoneProposal.docx`, a template written before the data was available. Its scope (simulated vibration signals, baseline models, a small neural network, comparison) is kept as the objective, but not as a constraint on the data.
- Real data available: Microsoft Azure Predictive Maintenance dataset (Kaggle, arnabbiswas1), in `Datasets/`.

## 2. Dataset findings (verified against the CSV files)

| File              | Rows (approx.) | Content                                                                    |
| ----------------- | -------------- | -------------------------------------------------------------------------- |
| PdM_telemetry.csv | 876,100        | 100 machines × 8,760 h (2015). Columns: volt, rotate, pressure, vibration |
| PdM_failures.csv  | 761            | Failures by component (comp1–comp4)                                       |
| PdM_errors.csv    | 3,919          | Errors (error1–error5)                                                    |
| PdM_maint.csv     | 3,286          | Maintenance records by component                                           |
| PdM_machines.csv  | 100            | model and age per machine                                                  |

- Failure events: 761 in 876,100 machine-hours (≈ 0.09 % per hour). This describes the events, not the model's label.
- With the D1 label (failure within the next 24 h): 17,184 positive rows, rate **≈ 1.96 %** (verified with `make_labels.py`). Imbalance of roughly 1:50.
- Date range: 2015-01-01 06:00 to 2016-01-01 06:00 (one full year, 100 machines).
- Telemetry is **one value per hour**, not a raw signal. No high-frequency FFT is possible.

## 3. Proposed decisions

| #  | Decision         | Proposal                                                                                                                                                       | Alternative              | Justification                                                                                   |
| -- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------- |
| D1 | Target variable  | Machine failure within the next 24 h (y = 1)                                                                                                                   | Next 48 h or 72 h        | Longer windows give more positives, but the alert loses temporal precision                      |
| D2 | Unit of analysis | One row per machine and hour, with trailing windows                                                                                                            | One row per failure      | Supports a daily prediction and keeps hours without failures                                    |
| D3 | Input features   | 24 h rolling means and standard deviations (volt, rotate, pressure, vibration); error count in 24 h; hours since last maintenance per component; age and model | Telemetry only           | Errors and maintenance are degradation signals; they must be measured without using future data |
| D4 | Partition        | Temporal: train 2015-01 to 2015-08 (train_end 2015-08-31), validate 2015-09 to 2015-10, test 2015-11 to 2015-12 (adjusted to a one-year calendar)              | Random split by machine  | Keeps consecutive hours of the same machine out of both training and test sets                  |
| D5 | Metrics          | Precision, recall, F1 for the positive class, PR-AUC, confusion matrix                                                                                         | Accuracy                 | With ~1.96 % positives, accuracy looks high even when the model misses every failure            |
| D6 | Models           | Baselines: logistic regression, decision tree, random forest. Small neural network (MLP)                                                                       | Sequential models (LSTM) | Baselines are interpretable, and the MLP meets the proposal's neural-network requirement        |
| D7 | Data origin      | State that this is a 2015 dataset with real failure labels; it is not vibration from a physical motor                                                          | —                       | Avoids overclaiming the link to the prototype                                                   |

## 4. Data leakage rules

- Feature windows for hour t use only data timestamped in (t − 24 h, t]. The reading at hour t is known when the prediction is made, so it is included; nothing after t is used.
- Scaling and imputation are fitted on the training set only.
- The failure label is never used as an input feature.
- Maintenance records are used only if they occurred before the prediction hour.

## 5. Assumptions to document

- Each hourly value is representative of that hour.
- Failures are recorded at the hour indicated.
- `age` and `model` do not change during the period.

## 6. Limitations

- One year, 100 machines, and few failures per component.
- Data origin: to be confirmed on the Kaggle page whether the data is synthetic.
- The model does not transfer directly to the physical prototype, which does not measure vibration yet.
- [ ] 7. Findings from the baseline

Baseline: logistic regression, balanced class weights, threshold 0.5, fitted on train (2015-01 to 2015-08), scored on validation (2015-09 to 2015-10). Saved to `results/metrics/baseline_logreg.csv`. The test set has not been used.

| Partition  | Precision | Recall | F1    | PR-AUC | ROC-AUC |
| ---------- | --------- | ------ | ----- | ------ | ------- |
| Validation | 0.477     | 1.000  | 0.646 | 0.856  | 0.998   |

- **F1 — Sensor regime shift.** Mean sensor values are at one level for 0–48 h before a failure and at the baseline level from 49 h onward (e.g., rotate ≈ 418 vs. ≈ 448). The shift is step-like, not gradual. It explains why any sensor-based model separates the positive class so well. The ROC-AUC of sensors alone is 0.997, although no single sensor exceeds 0.70.
- **F2 — Where the false positives are.** Of the 2 991 validation false positives, 230 fall 25–48 h before a failure (8.5 % of those negatives), 244 fall 49–96 h before, and 2 147 are more than 96 h from any failure. Most false positives are therefore not explained by the regime shift alone.
- **Implications.**
  - The task is closer to detecting an ongoing degradation state than to forecasting a failure. This must be stated as a limitation.
  - Part of this regime may come from how the dataset was generated (likely synthetic). This must be confirmed before the result is claimed as real degradation.
  - Precision at 0.5 is low partly because of balanced class weights. The threshold should be chosen on validation, not fixed at 0.5.
  - Sensitivity check needed: repeat the analysis with a 48 h label (D1) to see whether the 25–48 h false positives become true positives.
- **Leakage check.** The no-future-data test passes (`tests/test_features.py`). The high scores come from the signal in past readings, not from future data.

### Model comparison (validation, threshold 0.5)

Same split and features for all models. Saved to `results/metrics/model_comparison.csv`. The earlier `baseline_logreg.csv` is from the single-model run.

| Model               | Precision | Recall | F1    | PR-AUC | ROC-AUC | FP  | FN  |
| ------------------- | --------- | ------ | ----- | ------ | ------- | --- | --- |
| Logistic regression | 0.477     | 1.000  | 0.646 | 0.856  | 0.998   | 2991 | 0   |
| Decision tree       | 0.891     | 0.985  | 0.936 | 0.972  | 0.989   | 327  | 41  |
| Random forest       | 0.994     | 0.997  | 0.995 | 0.9997 | 1.000   | 17   | 9   |
| MLP (oversampled)   | 0.971     | 0.984  | 0.978 | 0.997  | 1.000   | 79   | 44  |

- **Interpretation caution.** Near-perfect scores (random forest PR-AUC 0.9997) are not yet credible as predictive performance. Given F1, the sensor regime shift is a likely explanation: any model that can detect the regime separates the classes almost perfectly, and tree models find the thresholds easily. The comparison shows that the task is easy under this label, not that the models predict failures well. The 48 h label sensitivity check and an analysis of the 25–48 h errors remain necessary.
- The test set has not been evaluated for any model.

## 8. Decision status

- **D1 (24 h) and D4 (temporal split):** applied as provisional assumptions. They can change if the team prefers 48 or 72 h.
- **Open:** should the simulated-signal and FFT track remain a separate section?
