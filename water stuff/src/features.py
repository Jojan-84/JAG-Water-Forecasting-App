"""Feature logic shared by the notebook, the verification script and the Streamlit app.

The functions here are copied from the notebook (Steps 5-6 and the `predict_next_day` helper in Step 15)
so training and prediction cannot drift apart. If you change a feature, change it here and
have the notebook import it too.

Units: every water amount is in cubic metres (m3) per day. 1 m3 = 1000 liters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHORT_RATIO = 0.6   # a day below 60% of its recent norm is flagged as "short" (probably partly recorded)
SHORT_WINDOW = 14   # days of history used for the recent norm
MIN_REPORTING = 20  # a day counts as "system was recording" if >= this many apartments have data
CAT_COLS = ["dow", "month"]
NUM_COLS = ["lag_1", "lag_7", "roll_7", "apt_avg"]
FEATURES = CAT_COLS + NUM_COLS


# --------------------------------------------------------------------------- training table
def calendar_info(daily_events: pd.DataFrame, min_reporting: int = MIN_REPORTING):
    """Return (all_days, valid_day). valid_day[d] is True when the system was recording on day d."""
    all_days = pd.date_range(daily_events["date"].min(), daily_events["date"].max(), freq="D")
    reporting = daily_events.groupby("date")["apartment"].nunique().reindex(all_days, fill_value=0)
    return all_days, reporting >= min_reporting


def _apartment_series(daily_events, apartment, all_days, valid_day, zero_fill=True):
    g = daily_events[daily_events["apartment"] == apartment]
    s = g.set_index("date")["consumption"].reindex(all_days)
    if zero_fill:
        # silent apartment on a day the system was recording -> 0 (zero readings were filtered out of the raw data)
        s = s.where(~(valid_day & s.isna()), 0.0)
    return s


def build_model_table(daily_events: pd.DataFrame, zero_fill: bool = True) -> pd.DataFrame:
    """Identical to the notebook's build_model_table()."""
    all_days, valid_day = calendar_info(daily_events)
    parts = []
    for apt in sorted(daily_events["apartment"].unique()):
        s = _apartment_series(daily_events, apt, all_days, valid_day, zero_fill)
        t = pd.DataFrame({"date": all_days, "apartment": apt, "consumption": s.values})
        t["lag_1"] = s.shift(1).values
        t["lag_7"] = s.shift(7).values
        t["roll_7"] = s.shift(1).rolling(7, min_periods=7).mean().values
        t["apt_avg"] = s.shift(1).expanding().mean().values
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    t["dow"] = t["date"].dt.dayofweek
    t["month"] = t["date"].dt.month
    t = t.rename(columns={"consumption": "target"})
    return t.dropna().reset_index(drop=True)


def history_window(daily_events, apartment, target_date, all_days, valid_day) -> pd.Series:
    """The 7 daily totals before `target_date` (oldest first), built exactly like the training lags."""
    s = _apartment_series(daily_events, apartment, all_days, valid_day)
    target = pd.Timestamp(target_date)
    return s.loc[target - pd.Timedelta(days=7): target - pd.Timedelta(days=1)]


def flag_short_days(daily_events: pd.DataFrame, threshold: float = SHORT_RATIO, window: int = SHORT_WINDOW) -> pd.DataFrame:
    """Flag days whose average total across reporting apartments is far below the previous `window` days' median.

    Returns one row per calendar day: avg_L, recent_norm_L, ratio, apartments, short (bool).
    """
    all_days = pd.date_range(daily_events["date"].min(), daily_events["date"].max(), freq="D")
    avg = daily_events.groupby("date")["consumption"].mean().reindex(all_days) * 1000
    apts = daily_events.groupby("date").size().reindex(all_days, fill_value=0)
    norm = avg.shift(1).rolling(window, min_periods=7).median()
    out = pd.DataFrame({"avg_L": avg, "recent_norm_L": norm, "ratio": avg / norm, "apartments": apts})
    out["short"] = out["ratio"] < threshold
    return out


# --------------------------------------------------------------------------- prediction
def validate_inputs(values, typical, max_daily):
    """Return (errors, warnings) as lists of plain-language messages. Values are in m3."""
    errors, warnings = [], []
    arr = np.array([np.nan if v is None else v for v in values], dtype=float)

    if len(arr) != 7:
        errors.append(f"Exactly 7 daily totals are needed (oldest first); got {len(arr)}.")
    elif np.isnan(arr).any() or np.isinf(arr).any():
        errors.append("Daily totals cannot be empty or non-numeric.")
    else:
        if (arr < 0).any():
            errors.append("Water use cannot be negative.")
        limit = max_daily * 1.5
        if arr.max() > limit:
            errors.append(
                f"A daily total of {arr.max():.2f} m3 is above {limit:.2f} m3, which is far outside anything seen "
                "in the training data. Please check the units (m3, not liters)."
            )
        if (arr == 0).all():
            warnings.append("All 7 days are zero, so the forecast reflects an empty apartment.")

    if typical is None or not np.isfinite(typical):
        errors.append("The apartment's typical daily use is missing.")
    elif typical < 0:
        errors.append("Typical daily use cannot be negative.")
    elif typical > max_daily * 1.5:
        errors.append(f"Typical daily use of {typical:.2f} m3 is far outside the training data; check the units.")
    return errors, warnings


def make_feature_row(values, typical, target_date) -> pd.DataFrame:
    d = pd.Timestamp(target_date)
    vals = [float(v) for v in values]
    return pd.DataFrame([{
        "dow": d.dayofweek, "month": d.month,
        "lag_1": vals[-1], "lag_7": vals[0], "roll_7": float(np.mean(vals)),
        "apt_avg": float(typical),
    }])[FEATURES]


def predict_next_day(pipeline, values, typical, target_date, max_daily) -> float:
    """values: 7 daily totals in m3, oldest first (the last one is yesterday). Returns m3 for target_date."""
    errors, _ = validate_inputs(values, typical, max_daily)
    if errors:
        raise ValueError(" ".join(errors))
    row = make_feature_row(values, typical, target_date)
    return max(0.0, float(pipeline.predict(row)[0]))


# --------------------------------------------------------------------------- CSV upload
def parse_week_csv(df: pd.DataFrame):
    """Read an uploaded CSV with columns `date` and `consumption` (or `value`).

    Returns (values, last_date, errors). Uses the last 7 rows, which must be 7 consecutive days.
    """
    cols = {str(c).strip().lower(): c for c in df.columns}
    vcol = cols.get("consumption", cols.get("value"))
    if "date" not in cols or vcol is None:
        return None, None, ["The CSV needs two columns: `date` and `consumption` (or `value`)."]
    dates = pd.to_datetime(df[cols["date"]], errors="coerce")
    vals = pd.to_numeric(df[vcol], errors="coerce")
    errors = []
    if dates.isna().any():
        errors.append("Some dates could not be read. Use the format YYYY-MM-DD.")
    if vals.isna().any():
        errors.append("Some consumption values are empty or not numbers.")
    if len(df) < 7:
        errors.append(f"At least 7 rows are needed; the file has {len(df)}.")
    if errors:
        return None, None, errors
    tidy = pd.DataFrame({"date": dates, "v": vals}).sort_values("date").tail(7)
    if tidy["date"].duplicated().any() or not (tidy["date"].diff().dropna() == pd.Timedelta(days=1)).all():
        return None, None, ["The last 7 rows must be 7 consecutive calendar days with no gaps or duplicates."]
    return tidy["v"].tolist(), tidy["date"].iloc[-1], []
