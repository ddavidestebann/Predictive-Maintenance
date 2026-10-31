"""
Tests for src/features/build_features.py using small synthetic tables.

Objective
    Check the window logic, the maintenance logic and, most importantly, that
    features at hour t do not change when data after t changes (no leakage).

Run
    py -m pytest tests            (if pytest is installed)
    py tests/test_features.py     (runs the same checks without pytest)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the src/ modules importable without installing the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))

from build_features import (  # noqa: E402
    add_error_counts,
    add_hours_since_maintenance,
    add_rolling_sensor_features,
)


def make_telemetry(hours: int = 48) -> pd.DataFrame:
    """Hourly telemetry for two machines with deterministic values."""
    rows = []
    start = pd.Timestamp("2015-01-01 00:00")
    for machine_id in (1, 2):
        for h in range(hours):
            rows.append(
                {
                    "datetime": start + pd.Timedelta(hours=h),
                    "machineID": machine_id,
                    "volt": 100.0 + h + machine_id,
                    "rotate": 400.0 + h,
                    "pressure": 100.0 + 2 * h,
                    "vibration": 40.0 + (h % 5),
                }
            )
    return pd.DataFrame(rows)


def test_rolling_mean_matches_manual_window():
    """The 24 h mean at hour t must equal the mean of readings in (t-24 h, t]."""
    telemetry = make_telemetry()
    out = add_rolling_sensor_features(telemetry)

    row = out[(out["machineID"] == 1)].iloc[30]  # hour 30 of machine 1
    t = row["datetime"]
    window = telemetry[
        (telemetry["machineID"] == 1)
        & (telemetry["datetime"] > t - pd.Timedelta(hours=24))
        & (telemetry["datetime"] <= t)
    ]
    assert len(window) == 24, "a full 24 h window should contain 24 hourly readings"
    assert np.isclose(row["volt_mean_24h"], window["volt"].mean())
    assert np.isclose(row["vibration_std_24h"], window["vibration"].std())


def test_error_window_is_right_closed():
    """An error exactly 24 h before t is outside the window; one at t is inside."""
    telemetry = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2015-01-02 00:00")],
            "machineID": [1],
        }
    )
    errors = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2015-01-01 00:00"), pd.Timestamp("2015-01-02 00:00")],
            "machineID": [1, 1],
            "errorID": ["error1", "error1"],
        }
    )
    out = add_error_counts(telemetry, errors)
    # Only the event at t (2015-01-02 00:00) counts; the one at t-24 h does not.
    assert out.loc[0, "error1_count_24h"] == 1


def test_hours_since_maintenance():
    """Maintenance at t gives 0 hours; before any record the value is NaN."""
    telemetry = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2015-01-01 00:00", "2015-01-01 05:00", "2015-01-01 10:00"]
            ),
            "machineID": [1, 1, 1],
        }
    )
    maint = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2015-01-01 05:00")],
            "machineID": [1],
            "comp": ["comp1"],
        }
    )
    out = add_hours_since_maintenance(telemetry, maint)
    assert np.isnan(out.loc[0, "hours_since_comp1"])  # no maintenance yet
    assert out.loc[1, "hours_since_comp1"] == 0.0  # maintenance at t
    assert out.loc[2, "hours_since_comp1"] == 5.0  # five hours later


def test_features_do_not_use_future_data():
    """Changing readings after hour t must not change features at hour t."""
    base = make_telemetry()
    changed = base.copy()
    future = changed["datetime"] > pd.Timestamp("2015-01-01 20:00")
    changed.loc[future, ["volt", "rotate", "pressure", "vibration"]] = 999.0

    a = add_rolling_sensor_features(base).reset_index(drop=True)
    b = add_rolling_sensor_features(changed).reset_index(drop=True)

    same_past = a["datetime"] <= pd.Timestamp("2015-01-01 20:00")
    feature_cols = [c for c in a.columns if c.endswith("_24h")]
    pd.testing.assert_frame_equal(
        a.loc[same_past, feature_cols].reset_index(drop=True),
        b.loc[same_past, feature_cols].reset_index(drop=True),
    )


if __name__ == "__main__":
    tests = [
        test_rolling_mean_matches_manual_window,
        test_error_window_is_right_closed,
        test_hours_since_maintenance,
        test_features_do_not_use_future_data,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} tests passed.")
