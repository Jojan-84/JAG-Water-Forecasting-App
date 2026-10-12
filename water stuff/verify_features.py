"""Check that the app's feature code reproduces what the notebook produced.

Run from the app folder:  python verify_features.py
PASS means: rebuilding the features from daily_totals.csv matches daily_features.csv, and predictions built
the app's way (7 raw days + typical use) equal the model's predictions on the stored feature rows.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import features as F  # noqa: E402

totals = pd.read_csv(ROOT / "data/processed/daily_totals.csv", parse_dates=["date"])
saved = pd.read_csv(ROOT / "data/processed/daily_features.csv", parse_dates=["date"])
rebuilt = F.build_model_table(totals)

print(f"Saved table:   {len(saved):>7} rows")
print(f"Rebuilt table: {len(rebuilt):>7} rows")
ok = len(saved) == len(rebuilt)

m = saved.merge(rebuilt, on=["date", "apartment"], suffixes=("_saved", "_new"), how="outer", indicator=True)
only = m["_merge"].value_counts()
print("Rows present in only one table:", int(only.get("left_only", 0) + only.get("right_only", 0)))
ok &= (only.get("left_only", 0) + only.get("right_only", 0)) == 0

both = m[m["_merge"] == "both"]
for col in ["target", "lag_1", "lag_7", "roll_7", "apt_avg", "dow", "month"]:
    diff = (both[f"{col}_saved"] - both[f"{col}_new"]).abs().max()
    print(f"  max |difference| in {col:<8}: {diff:.3e}")
    ok &= bool(diff < 1e-9)

# end-to-end: predict the way the app does and compare with the model on the stored feature rows
model = joblib.load(ROOT / "models/best_model.joblib")
info = json.load(open(ROOT / "models/model_info.json", encoding="utf-8"))
max_daily = info["max_daily_seen_in_training_m3"]
all_days, valid_day = F.calendar_info(totals)
sample = saved.sample(min(300, len(saved)), random_state=0)
direct = model.predict(sample[F.FEATURES])
via_app = []
for _, r in sample.iterrows():
    w = F.history_window(totals, r["apartment"], r["date"], all_days, valid_day).tolist()
    # bypass the input range check so any training row can be compared
    row = F.make_feature_row(w, r["apt_avg"], r["date"])
    via_app.append(float(model.predict(row)[0]))
gap = np.abs(direct - np.array(via_app)).max()
print(f"  max |difference| app path vs stored features (m3): {gap:.3e}")
ok &= bool(gap < 1e-9)

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
