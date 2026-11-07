"""
Train and evaluate the baseline models on the same temporal split.

Objective
    Build the full supervised dataset (features + label), split it by time, fit
    each model on the training period and report classification metrics on the
    validation period. The test set is only evaluated when explicitly requested,
    so that it is not used for model choices (see docs/Decision_Record_v0.md, D4-D6).

Models (D6)
    - logistic_regression : linear baseline, balanced class weights.
    - decision_tree       : interpretable tree, balanced class weights, limited depth.
    - random_forest       : ensemble of trees, balanced subsample weights.
    - mlp                 : small neural network (scikit-learn MLPClassifier). It has
                            no class weights, so the training set is balanced by
                            resampling positives. Validation and test are never resampled.

Pipeline
    load_data.load_raw -> make_labels -> build_features -> temporal_split
    -> preprocessing fitted on train only -> classifier

Inputs
    data_dir        : folder with the raw CSV files (default: Datasets/).
    --models        : optional list of model names to run (default: all).
    --evaluate-test : optional flag to also report the test set.

Outputs
    Metrics printed to the console and saved to results/metrics/model_comparison.csv.

Dependencies
    pandas, numpy, scikit-learn.

Integration with the project
    Uses the modules in src/data and src/features. The metrics file feeds the
    results section of the report.
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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

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

RANDOM_STATE = 42

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


def build_classifier(name: str):
    """Return an untrained classifier for the given model name.

    The names match the MODEL_NAMES list below and the entries of D6.
    """
    if name == "logistic_regression":
        return LogisticRegression(class_weight="balanced", max_iter=2000)
    if name == "decision_tree":
        return DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=RANDOM_STATE)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=50,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "mlp":
        # early_stopping is off: its internal validation split would be random
        # and would mix neighbouring hours from the training period.
        return MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=50,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model name: {name}")


MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest", "mlp"]
# Models without class weights need the training set balanced by resampling.
NEEDS_OVERSAMPLING = {"mlp"}


def make_pipeline_for(numeric_columns: list[str], classifier) -> Pipeline:
    """Return the preprocessing + classifier pipeline.

    Preprocessing is part of the pipeline, so it is fitted only when `fit` is
    called on the training data, which prevents leakage into validation and test.

    - Numeric columns: median imputation, then standard scaling.
    - Categorical columns: one-hot encoding; unseen categories are ignored.
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
    return Pipeline(steps=[("preprocess", preprocess), ("classifier", classifier)])


def oversample_positives(train: pd.DataFrame) -> pd.DataFrame:
    """Return the training set with positives resampled to match negatives.

    Only the training partition is passed here, so no validation or test
    information is used.
    """
    positives = train[train["y"] == 1]
    negatives = train[train["y"] == 0]
    sampled_positives = positives.sample(n=len(negatives), replace=True, random_state=RANDOM_STATE)
    balanced = pd.concat([negatives, sampled_positives])
    return balanced.sample(frac=1.0, random_state=RANDOM_STATE)  # shuffle


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


def train_and_evaluate(data_dir: str | Path, model_names: list[str], evaluate_test: bool) -> pd.DataFrame:
    """Fit each requested model on train and score validation (and test if asked)."""
    dataset = build_dataset(data_dir)
    train, val, test = temporal_split(dataset, TRAIN_END, VAL_END)

    feature_columns = [c for c in dataset.columns if c not in ID_COLUMNS]
    numeric_columns = [c for c in feature_columns if c not in CATEGORICAL_COLUMNS]

    partitions = {"validation": val}
    if evaluate_test:
        partitions["test"] = test

    rows = []
    for model_name in model_names:
        fit_data = oversample_positives(train) if model_name in NEEDS_OVERSAMPLING else train
        pipeline = make_pipeline_for(numeric_columns, build_classifier(model_name))
        pipeline.fit(fit_data[feature_columns], fit_data["y"])

        for part_name, part in partitions.items():
            scores = pipeline.predict_proba(part[feature_columns])[:, 1]
            metrics = evaluate(part["y"].to_numpy(), scores)
            metrics["model"] = model_name
            metrics["partition"] = part_name
            metrics["rows"] = len(part)
            metrics["positive_rate"] = part["y"].mean()
            rows.append(metrics)
        print(f"Finished {model_name}", flush=True)

    return pd.DataFrame(rows).set_index(["model", "partition"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare the baseline models.")
    parser.add_argument("data_dir", nargs="?", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
        help="Models to run (default: all).",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also report the test set. Use only for final reporting.",
    )
    args = parser.parse_args()

    results = train_and_evaluate(args.data_dir, args.models, args.evaluate_test)
    pd.set_option("display.width", 160)
    print(results.round(4).to_string())

    output_dir = PROJECT_ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_comparison.csv"
    results.to_csv(output_path)
    print(f"\nSaved metrics to {output_path}")
