"""Apartment next-day water forecast (Streamlit).

Run:  streamlit run app.py
Needs: models/best_model.joblib, models/model_info.json, models/apartment_baselines.json,
       data/processed/daily_features.csv, data/processed/daily_totals.csv
"""
import json
import re
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import features as F  # noqa: E402

MODEL_PATH = ROOT / "models" / "best_model.joblib"
INFO_PATH = ROOT / "models" / "model_info.json"
BASELINES_PATH = ROOT / "models" / "apartment_baselines.json"
FEATURES_PATH = ROOT / "data" / "processed" / "daily_features.csv"
TOTALS_PATH = ROOT / "data" / "processed" / "daily_totals.csv"

st.set_page_config(page_title="Apartment Water Forecast", page_icon="💧", layout="centered")


# ----------------------------------------------------------------------------- loading
def require(path: Path):
    if not path.exists():
        st.error(f"Missing file: `{path.relative_to(ROOT)}`. Copy it from your notebook output into the app folder.")
        st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_features():
    return pd.read_csv(FEATURES_PATH, parse_dates=["date"])


@st.cache_data
def load_totals():
    totals = pd.read_csv(TOTALS_PATH, parse_dates=["date"])
    all_days, valid_day = F.calendar_info(totals)
    return totals, all_days, valid_day


@st.cache_data
def load_short_days():
    return F.flag_short_days(load_totals()[0])


for p in (MODEL_PATH, INFO_PATH, BASELINES_PATH, FEATURES_PATH, TOTALS_PATH):
    require(p)

try:
    model = load_model()
except Exception as exc:  # most often a scikit-learn version mismatch
    st.error(
        "The saved model could not be loaded. This usually means the scikit-learn version here differs from the one "
        "used for training. Pin the training version in `requirements.txt`.\n\n"
        f"Details: `{type(exc).__name__}: {exc}`"
    )
    st.stop()

info = load_json(INFO_PATH)
baselines = load_json(BASELINES_PATH)
features_df = load_features()
totals, ALL_DAYS, VALID_DAY = load_totals()
SHORT_DAYS = load_short_days()
MAX_DAILY = float(info["max_daily_seen_in_training_m3"])
TEST_MAE = float(info["test_metrics"]["MAE"])
CUTOFF = pd.Timestamp(info["cutoff_date"])


# ----------------------------------------------------------------------------- display
def show_result(values, typical, target_date, actual=None, note=None):
    """Validate, predict and draw the result. `values` are 7 daily totals in m3, oldest first."""
    errors, warnings = F.validate_inputs(values, typical, MAX_DAILY)
    for w in warnings:
        st.warning(w)
    if errors:
        for e in errors:
            st.error(e)
        return

    pred = F.predict_next_day(model, values, typical, target_date, MAX_DAILY)
    target = pd.Timestamp(target_date)
    lo, hi = max(0.0, pred - TEST_MAE), pred + TEST_MAE

    c1, c2 = st.columns(2)
    c1.metric(f"Forecast for {target:%a %d %b %Y}", f"{pred * 1000:,.0f} L", help=f"{pred:.3f} m³")
    c2.metric("Typical error", f"± {TEST_MAE * 1000:,.0f} L",
              help="Average absolute error on days the model never saw during training.")
    st.write(
        f"Expected somewhere around **{lo * 1000:,.0f}–{hi * 1000:,.0f} L** ({pred:.3f} m³). "
        "The error figure is an average, so any single day can be off by more."
    )
    if typical > 0:
        diff = (pred / typical - 1) * 100
        st.caption(f"That is {abs(diff):.0f}% {'above' if diff >= 0 else 'below'} this apartment's "
                   f"typical day ({typical * 1000:,.0f} L).")
    if note:
        st.caption(note)

    hist_dates = [target - pd.Timedelta(days=7 - i) for i in range(7)]
    rows = [{"label": d.strftime("%a %d %b"), "liters": v * 1000, "kind": "Recorded"} for d, v in zip(hist_dates, values)]
    rows.append({"label": target.strftime("%a %d %b"), "liters": pred * 1000, "kind": "Forecast"})
    domain, colors = ["Recorded", "Forecast"], ["#8aa4b8", "#1f77b4"]
    if actual is not None:
        rows.append({"label": target.strftime("%a %d %b") + " ", "liters": actual * 1000, "kind": "Actual"})
        domain.append("Actual")
        colors.append("#2ca02c")
        st.success(f"Actual use that day: **{actual * 1000:,.0f} L** "
                   f"(forecast was off by {abs(pred - actual) * 1000:,.0f} L).")
    chart_df = pd.DataFrame(rows)
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=list(chart_df["label"]), title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("liters:Q", title="Liters per day"),
            color=alt.Color("kind:N", scale=alt.Scale(domain=domain, range=colors), title=None),
            tooltip=["label", alt.Tooltip("liters:Q", format=",.0f"), "kind"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, width="stretch")

    with st.expander("Exactly what the model was given"):
        st.dataframe(F.make_feature_row(values, typical, target_date).rename(columns={
            "dow": "weekday (0=Mon)", "month": "month", "lag_1": "yesterday (m³)", "lag_7": "7 days ago (m³)",
            "roll_7": "7-day average (m³)", "apt_avg": "apartment typical use (m³)"}), hide_index=True)


def parse_text_values(text):
    parts = [p for p in re.split(r"[,\s;]+", text.strip()) if p]
    vals, bad = [], []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            bad.append(p)
            vals.append(np.nan)
    return vals, bad


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About this model")
    st.write(f"**Model:** {info['model_name']}")
    st.write("**Predicts:** one apartment's total water use on the next day.")
    st.write("**Unit:** cubic metres (1 m³ = 1000 liters).")
    st.write(f"**Trained on:** {info['train_period'][0]} to {info['train_period'][1]}")
    st.write(f"**Held-out test days start:** {info['cutoff_date']}")
    t = info["test_metrics"]
    st.write(f"**Test error:** MAE {t['MAE'] * 1000:,.0f} L · RMSE {t['RMSE'] * 1000:,.0f} L · R² {t['R2']:.2f}")

# ----------------------------------------------------------------------------- main
st.title("💧 Apartment water forecast")
st.caption("Predict an apartment's total water use tomorrow from its last 7 days.")

tab_hist, tab_own, tab_how = st.tabs(["Recorded apartment", "My own numbers", "How it works"])

# ---- Tab 1: use recorded history -------------------------------------------------
with tab_hist:
    st.write("Pick an apartment and a day from the dataset. The app rebuilds the 7 days before it, "
             "forecasts the day, and compares with what really happened.")
    apt = st.selectbox("Apartment", sorted(features_df["apartment"].unique()))
    # The dataset's final day is only partly recorded (its totals are about half a normal day), so it is not offered.
    last_day = features_df["date"].max()
    sub = features_df[(features_df["apartment"] == apt) & (features_df["date"] < last_day)]
    dmin, dmax = sub["date"].min().date(), sub["date"].max().date()
    chosen = st.date_input("Day to predict", value=dmax, min_value=dmin, max_value=dmax)
    row = sub[sub["date"] == pd.Timestamp(chosen)]
    if row.empty:
        ts = pd.Timestamp(chosen)
        before, after = sub.loc[sub["date"] < ts, "date"].max(), sub.loc[sub["date"] > ts, "date"].min()
        near = []
        if pd.notna(before):
            near.append(f"{before:%d %b %Y} ({(ts - before).days} days earlier)")
        if pd.notna(after):
            near.append(f"{after:%d %b %Y} ({(after - ts).days} days later)")
        st.warning("This apartment has no complete 7-day history before that day: it needs 7 days in a row of readings, "
                   "and gaps or system outages break that. "
                   + ("Nearest dates that work: " + " and ".join(near) + "." if near else "Try another date."))
    else:
        r = row.iloc[0]
        window = F.history_window(totals, apt, chosen, ALL_DAYS, VALID_DAY)
        vals = window.tolist()
        if len(vals) != 7 or not np.allclose(
                [vals[-1], vals[0], np.mean(vals)], [r["lag_1"], r["lag_7"], r["roll_7"]], atol=1e-9):
            st.warning("The rebuilt 7 days do not match the stored features for this row; "
                       "run `python verify_features.py` to find out why.")
        chosen_ts = pd.Timestamp(chosen)
        if chosen_ts in SHORT_DAYS.index and SHORT_DAYS.loc[chosen_ts, "short"]:
            d = SHORT_DAYS.loc[chosen_ts]
            st.warning(f"This day looks only partly recorded: apartments averaged {d['avg_L']:.0f} L against a recent norm of "
                       f"{d['recent_norm_L']:.0f} L ({int(d['apartments'])} apartments reporting). The 'actual' value is probably "
                       "too low, so a big miss here says little about the model.")
        prev_ts = chosen_ts - pd.Timedelta(days=1)
        if prev_ts in SHORT_DAYS.index and SHORT_DAYS.loc[prev_ts, "short"]:
            st.info("The day before looks partly recorded, so the 'yesterday' value fed to the model is unusually low "
                    "and the forecast will be too.")
        phase = ("a held-out **test** day: the model never saw it" if pd.Timestamp(chosen) >= CUTOFF
                 else "a **training** day: the model learned from it, so it flatters the model")
        st.caption(f"{chosen:%d %b %Y} is {phase}.")
        show_result(vals, float(r["apt_avg"]), chosen, actual=float(r["target"]))

# ---- Tab 2: user-entered numbers -------------------------------------------------
with tab_own:
    unit = st.radio("Unit of the numbers you enter", ["m³", "liters"], horizontal=True)
    scale = 1.0 if unit == "m³" else 1 / 1000
    method = st.radio("How do you want to provide the last 7 days?", ["Type them in", "Upload a CSV"], horizontal=True)

    values_m3, target_date, ready = None, None, False
    if method == "Type them in":
        default = "0.08, 0.09, 0.07, 0.10, 0.08, 0.12, 0.11" if unit == "m³" else "80, 90, 70, 100, 80, 120, 110"
        text = st.text_input("Last 7 daily totals, oldest first (separate with commas or spaces)",
                             value=default, key=f"vals_{unit}")
        # Only weekday and month matter to the model (not the year), so a one-year window is plenty.
        target_date = st.date_input("Day to predict", value=(features_df["date"].max() + pd.Timedelta(days=1)).date(),
                                    min_value=features_df["date"].min().date(),
                                    max_value=(features_df["date"].max() + pd.Timedelta(days=365)).date(),
                                    key="own_date", help="The model uses only the weekday and the month of this date.")
        parsed, bad = parse_text_values(text)
        if bad:
            st.error("These entries are not numbers: " + ", ".join(f"`{b}`" for b in bad)
                     + ". Use a dot for decimals.")
        elif not parsed:
            st.error("Please enter 7 daily totals.")
        else:
            values_m3, ready = [v * scale for v in parsed], True
    else:
        st.caption("CSV with columns `date` (YYYY-MM-DD) and `consumption`. The last 7 consecutive days are used and "
                   "the forecast is for the day after the last date.")
        up = st.file_uploader("Upload CSV", type="csv")
        if up is not None:
            try:
                parsed_df = pd.read_csv(up)
            except Exception as exc:
                st.error(f"That file could not be read as a CSV ({type(exc).__name__}).")
            else:
                v, last_date, errs = F.parse_week_csv(parsed_df)
                if errs:
                    for e in errs:
                        st.error(e)
                else:
                    values_m3, ready = [x * scale for x in v], True
                    target_date = (last_date + pd.Timedelta(days=1)).date()
                    st.info(f"Using {(last_date - pd.Timedelta(days=6)):%d %b} to {last_date:%d %b %Y}; "
                            f"forecasting {target_date:%d %b %Y}.")

    source = st.selectbox("Apartment's typical daily use", [
        "Average of the 7 days above", "A recorded apartment's long-run average", "Type my own"])
    typical = None
    if source == "Average of the 7 days above":
        if ready and not any(pd.isna(values_m3)):
            typical = float(np.mean(values_m3))
    elif source == "A recorded apartment's long-run average":
        pick = st.selectbox("Apartment", sorted(baselines), key="own_apt")
        typical = float(baselines[pick])
        st.caption(f"{pick}: {typical:.3f} m³ ({typical * 1000:,.0f} L) per day over the training period.")
    else:
        entered = st.number_input(f"Typical daily use ({unit})", min_value=0.0, value=None, step=0.01 if unit == "m³" else 10.0)
        typical = None if entered is None else entered * scale

    if ready:
        if typical is None and source != "Average of the 7 days above":
            st.info("Enter the typical daily use to see a forecast.")
        else:
            show_result(values_m3, typical if typical is not None else np.nan, target_date,
                        note="The typical-use figure here is an approximation of the training feature, "
                             "which was each apartment's running average.")

# ---- Tab 3: explanation ---------------------------------------------------------
with tab_how:
    st.subheader("What the model does")
    st.write(
        f"A **{info['model_name']}** model turns six numbers into a forecast of one apartment's water use on a given day. "
        "It was trained on one apartment-day per row, using only information available the day before."
    )
    st.table(pd.DataFrame({
        "Input": ["weekday", "month", "yesterday", "7 days ago", "7-day average", "apartment typical use"],
        "Meaning": ["Day of week of the day being predicted", "Month of the day being predicted",
                    "Total use the day before (m³)", "Total use 7 days before the target day (m³)",
                    "Mean of the last 7 daily totals (m³)", "Apartment's average daily use before that day (m³)"],
    }))
    st.subheader("How to read the output")
    st.write("The forecast is the apartment's **total water use for the whole day**, shown in liters (and m³ on hover). "
             "The ± figure is the model's average miss on days it never trained on.")
    st.subheader("Limits")
    st.markdown(
        "- Values far above anything in the training data are rejected, as are negative or empty entries.\n"
        "- Accuracy differs between apartments; low-use apartments are harder to predict in relative terms.\n"
        "- Forecasts for dates outside the data's seasons rely on the model's month effects and are less reliable."
    )
