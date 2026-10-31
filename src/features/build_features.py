"""
Build the input features for each machine-hour.

Objective
    Turn the raw tables into one row per (machineID, datetime) with features that
    describe the machine's recent condition and maintenance history. Every feature
    uses only data in the window (t - 24 h, t], so nothing after the prediction
    hour leaks into the inputs (see docs/Decision_Record_v0.md, D3 and section 4).

Features
    - Rolling 24 h mean and standard deviation of volt, rotate, pressure, vibration.
    - Number of each error type (error1..error5) in the last 24 h.
    - Hours since the last maintenance of each component (comp1..comp4).
      NaN if the component has no maintenance record before t.
    - Static machine attributes: age and model.

Inputs
    tables : dict returned by load_data.load_raw(), with keys
             "telemetry", "errors", "maint" and "machines".

Outputs
    DataFrame with one row per telemetry row. Columns: machineID, datetime,
    the feature columns above. The label `y` is not included; it comes from
    make_labels.py and is joined by the caller.

Dependencies
    pandas, numpy.

Integration with the project
    Pipeline order: load_data -> make_labels -> build_features -> split.
    Missing values (NaN) are left in place here. Imputation must be fitted on
    the training partition only, in the modelling step.
"""

import numpy as np
import pandas as pd

WINDOW = "24h"
SENSORS = ["volt", "rotate", "pressure", "vibration"]
ERROR_TYPES = ["error1", "error2", "error3", "error4", "error5"]
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]


def add_rolling_sensor_features(telemetry: pd.DataFrame, window: str = WINDOW) -> pd.DataFrame:
    """Add rolling mean and standard deviation of each sensor per machine.

    The pandas time-based window `24h` is right-closed, so it covers
    (t - 24 h, t] and includes the current reading.

    Parameters
    ----------
    telemetry : pandas.DataFrame
        Columns: machineID, datetime and the sensor columns.
    window : str, default "24h"
        Rolling window length.

    Returns
    -------
    pandas.DataFrame
        Telemetry rows sorted by machine and time, with `<sensor>_mean_24h`
        and `<sensor>_std_24h` columns added.
    """
    df = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    pieces = []
    for _, group in df.groupby("machineID", sort=False):
        rolling = group.set_index("datetime")[SENSORS].rolling(window)
        feats = pd.concat(
            [rolling.mean().add_suffix("_mean_24h"), rolling.std().add_suffix("_std_24h")],
            axis=1,
        )
        feats.index = group.index  # restore the row alignment with `df`
        pieces.append(feats)

    return df.join(pd.concat(pieces))


def _count_events_in_window(
    telemetry: pd.DataFrame,
    events: pd.DataFrame,
    window: pd.Timedelta,
) -> np.ndarray:
    """Count, for each telemetry row, the events of its machine in (t - window, t].

    Uses binary search on sorted event times per machine.
    Returns an integer array aligned with `telemetry` rows.
    """
    counts = np.zeros(len(telemetry), dtype=int)
    machine_ids = telemetry["machineID"].to_numpy()

    for machine_id, event_rows in events.groupby("machineID"):
        event_times = np.sort(event_rows["datetime"].to_numpy())
        mask = machine_ids == machine_id
        t = telemetry.loc[mask, "datetime"].to_numpy()

        upto_t = np.searchsorted(event_times, t, side="right")
        upto_t_minus_window = np.searchsorted(event_times, t - window.to_timedelta64(), side="right")
        counts[mask] = upto_t - upto_t_minus_window

    return counts


def add_error_counts(telemetry: pd.DataFrame, errors: pd.DataFrame, window: str = WINDOW) -> pd.DataFrame:
    """Add a count of each error type in the 24 h window ending at each row."""
    result = telemetry.copy()
    window_td = pd.Timedelta(window)

    for error_id in ERROR_TYPES:
        events = errors[errors["errorID"] == error_id]
        result[f"{error_id}_count_24h"] = _count_events_in_window(result, events, window_td)

    return result


def add_hours_since_maintenance(telemetry: pd.DataFrame, maint: pd.DataFrame) -> pd.DataFrame:
    """Add hours since the last maintenance of each component.

    Only maintenance at or before t is used. If no earlier record exists for
    the machine, the value is NaN.
    """
    result = telemetry.copy()
    machine_ids = result["machineID"].to_numpy()

    for component in COMPONENTS:
        comp_events = maint[maint["comp"] == component]
        last_time = np.full(len(result), np.datetime64("NaT"), dtype="datetime64[ns]")

        for machine_id, event_rows in comp_events.groupby("machineID"):
            event_times = np.sort(event_rows["datetime"].to_numpy())
            mask = machine_ids == machine_id
            t = result.loc[mask, "datetime"].to_numpy()

            # Index of the last maintenance at or before t; -1 means none yet.
            idx = np.searchsorted(event_times, t, side="right") - 1
            has_prior = idx >= 0
            machine_last = np.full(len(t), np.datetime64("NaT"), dtype="datetime64[ns]")
            machine_last[has_prior] = event_times[idx[has_prior]]
            last_time[mask] = machine_last

        hours = (result["datetime"].to_numpy() - last_time) / np.timedelta64(1, "h")
        result[f"hours_since_{component}"] = hours

    return result


def build_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run all feature steps and return the feature table.

    Parameters
    ----------
    tables : dict[str, pandas.DataFrame]
        Output of load_data.load_raw().

    Returns
    -------
    pandas.DataFrame
        One row per telemetry row, with machineID, datetime and all features.
    """
    features = add_rolling_sensor_features(tables["telemetry"])
    features = add_error_counts(features, tables["errors"])
    features = add_hours_since_maintenance(features, tables["maint"])

    # Static machine attributes, one value per machine.
    machines = tables["machines"][["machineID", "age", "model"]]
    features = features.merge(machines, on="machineID", how="left", validate="many_to_one")

    return features


if __name__ == "__main__":
    # Quick check on the raw data folder.
    # Usage: py build_features.py [optional_data_folder]   (default: Datasets/)
    import sys
    from pathlib import Path

    # load_data.py lives in src/data/; add that folder to the import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from load_data import DEFAULT_DATA_DIR, load_raw

    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    features = build_features(load_raw(folder))
    print(f"Rows: {len(features):,}  columns: {len(features.columns)}")
    print("Missing values per column (only non-zero shown):")
    missing = features.isna().sum()
    print(missing[missing > 0].to_string())
