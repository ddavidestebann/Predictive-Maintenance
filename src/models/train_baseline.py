"""
Train and evaluate the baseline model: logistic regression.

Objective
    Build the full supervised dataset (features + label), split it by time,
    fit a logistic regression baseline and report classification metrics on
    the validation set. The test set is only evaluated when explicitly requested,
    so that it is not used for model choices (see docs/Decision_Record_v0.md, D4-D6).

Pipeline
    load_data.load_raw -> make_labels -> build_features -> temporal_split
    -> preprocessing fitted on train only -> LogisticRegression

Inputs
    data_dir : folder with the raw CSV files (default: Datasets/).
    --evaluate-test : optional flag to also report the test set.

Outputs
    Metrics printed to the console and saved to results/metrics/baseline_logreg.csv.

Dependencies
    pandas, numpy, scikit-learn.

Integration with the project
    Uses the three earlier modules in src/data and src/features. The metrics
    file feeds the results section of the report.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))

from build_features import build_features  # noqa: E402
from load_data import DEFAULT_DATA_DIR, load_raw  # noqa: E402
from make_labels import make_labels  # noqa: E402
from split import temporal_split  # noqa: E402

# Temporal split from the decision record (D4): train Jan-Aug, validate Sep-Oct,
# test Nov-Dec 2015. Bounds are inclusive; the data is hourly.
TRAIN_END = "2015-08-31 23:00:00"
VAL_END = "2015-10-31 23:00:00"

# Columns that are identifiers or are handled separately, not model inputs.
ID_COLUMNS = ["machineID", "datetime", "y"]
CATEGORICAL_COLUMNS = ["model"]


def build_dataset(data_dir: str | Path) -> pd.DataFrame:
    """Return features and label in one table, one row per machine-hour."""
    tables = load_raw(data_dir)

    labeled = make_labels(tables["telemetry"], tables["failures"], horizon_hours=24)
    features = build_features(tables)

    # Join the label onto the feature table by machine and timestamp.
    # validate="one_to_one" stops the script if a machine-hour is duplicated.
    dataset = features.merge(
        labeled[["machineID", "datetime", "y"]],
        on=["machineID", "datetime"],
        how="inner",
        validate="one_to_one",
    )
    if len(dataset) != len(features):
        raise RuntimeError("Label join changed the row count; check the input data.")
    return dataset


def make_pipeline_for(numeric_columns: list[str]) -> Pipeline:
    """Return the preprocessing + logistic regression pipeline for these columns.

    Preprocessing is part of the pipeline, so it is fitted only when `fit` is
    called on the training data, which prevents leakage into validation and test.

    - Numeric columns: median imputation, then standard scaling.
    - Categorical columns: one-hot encoding; unseen categories are ignored.
    - Classifier: logistic regression with balanced class weights, because
      positives are about 2 % of rows.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    classifier = LogisticRegression(class_weight="balanced", max_iter=2000)
    return Pipeline(steps=[("preprocess", preprocess), ("classifier", classifier)])


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute classification metrics for the positive class.

    PR-AUC (average precision) does not depend on the threshold and is the most
    informative metric for this imbalanced problem (D5).
    """
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_and_evaluate(data_dir: str | Path, evaluate_test: bool) -> pd.DataFrame:
    """Run the full baseline: split, fit on train, score validation (and test)."""
    dataset = build_dataset(data_dir)
    train, val, test = temporal_split(dataset, TRAIN_END, VAL_END)

    feature_columns = [c for c in dataset.columns if c not in ID_COLUMNS]
    numeric_columns = [c for c in feature_columns if c not in CATEGORICAL_COLUMNS]

    model = make_pipeline_for(numeric_columns)
    model.fit(train[feature_columns], train["y"])

    rows = []
    partitions = {"validation": val}
    if evaluate_test:
        partitions["test"] = test

    for name, part in partitions.items():
        scores = model.predict_proba(part[feature_columns])[:, 1]
        metrics = evaluate(part["y"].to_numpy(), scores)
        metrics["partition"] = name
        metrics["rows"] = len(part)
        metrics["positive_rate"] = part["y"].mean()
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("partition")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the logistic regression baseline.")
    parser.add_argument("data_dir", nargs="?", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also report the test set. Use only for final reporting.",
    )
    args = parser.parse_args()

    results = train_and_evaluate(args.data_dir, args.evaluate_test)
    pd.set_option("display.width", 120)
    print(results.round(4).to_string())

    output_dir = PROJECT_ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_logreg.csv"
    results.to_csv(output_path)
    print(f"\nSaved metrics to {output_path}")
