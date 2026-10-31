# Predictive Maintenance with Machine Learning (AML 3404, Fall 2026)

**Status:** in development. So far only the data pipeline exists.

## Problem and objective

Predict whether a rotating machine will fail within the next 24 hours, using operational telemetry (voltage, rotation, pressure and vibration), error logs and maintenance records. Baseline models are compared against a small neural network.

Design decisions and their alternatives are documented in [docs/Decision_Record_v0.md](docs/Decision_Record_v0.md).

## Team

[To be completed: team member names and student numbers]

## Dataset

Microsoft Azure Predictive Maintenance (Kaggle: `arnabbiswas1/microsoft-azure-predictive-maintenance`).

- 100 machines, one year (2015), one reading per hour.
- The original CSV files live in `Datasets/` and are **not** committed to the repository.
- Positive rate with the 24-hour label: ≈ 1.96 %.

## Structure

```
Capstone Proj/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/                  # Decision record and documentation
├── src/
│   ├── data/              # Loading, labels and temporal split
│   ├── features/          # Rolling sensor, error and maintenance features
│   └── models/            # Baseline model training and evaluation
├── tests/                 # Unit tests for the pipeline
├── Datasets/              # Original CSV files (not versioned)
└── *.ipynb                # Exploration notebooks
```

## Installation

```bash
py -m pip install -r requirements.txt
```

## Usage

Run from any folder. By default the scripts read `Datasets/` at the project root; another folder can be passed as an argument:

```bash
py src/data/load_data.py
py src/data/make_labels.py
py src/features/build_features.py
py tests/test_features.py
py src/models/train_baseline.py                  # validation metrics only
py src/models/train_baseline.py --evaluate-test  # also reports the test set (final use only)
```

## Limitations

- One year and 100 machines; few failures per component.
- Telemetry is an hourly value, not a raw vibration signal.
- Data origin still to be confirmed (possibly synthetic).
- The model does not transfer directly to a physical prototype.
