"""
train_model.py
================
Trains the FAAN ENU Flight Delay prediction model.

Two models are trained and saved:
  1. A classifier  -> predicts is_delayed (0/1)
  2. A regressor    -> predicts delay_minutes, but ONLY used when is_delayed = 1

IMPORTANT — Data leakage:
We deliberately EXCLUDE any column that would not be known before the flight
actually departs: actual_departure, delay_minutes (as a classifier feature),
delay_category, delay_cause, flight_id, flight_number, date, route (string
duplicate of origin+destination). Using these would let the model "cheat"
and give unrealistically high accuracy that falls apart on real predictions.

Run:
    python train_model.py
Outputs (into ./model/):
    classifier.pkl        - trained delay classifier
    regressor.pkl          - trained delay-minutes regressor
    preprocessor.pkl        - fitted ColumnTransformer (encodes categoricals, scales numerics)
    feature_importance.json - top features driving predictions (used by the API to explain results)
    metadata.json            - training metrics + feature list, for reference / defense slides
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# NOTE: this points directly at your actual CSV location on your machine.
# If you ever move the project folder, update this path to match.
# ---------------------------------------------------------------------------
DATA_PATH = Path(r"C:\Users\OWNER\Desktop\Project\enu_flight_delays.csv")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. Feature engineering
#    Only features that are knowable BEFORE the flight departs.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "departure_hour",
    "flight_duration_min",
    "distance_km",
    "temperature_c",
    "visibility_km",
    "wind_speed_kph",
    "humidity_pct",
    "is_weekend",
    "is_peak_hour",
    "is_holiday_period",
    "is_international",
]

CATEGORICAL_FEATURES = [
    "day_of_week",
    "month",
    "season",
    "airline",
    "flight_type",
    "origin",
    "destination",
    "weather_condition",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = df[FEATURE_COLUMNS].copy()
y_class = df["is_delayed"].copy()

# Regressor target: delay_minutes, trained ONLY on rows that were actually delayed
delayed_mask = df["is_delayed"] == 1
X_delayed = X[delayed_mask].copy()
y_minutes = df.loc[delayed_mask, "delay_minutes"].copy()

print(f"  Delayed flights: {delayed_mask.sum()} ({delayed_mask.mean():.1%})")

# ---------------------------------------------------------------------------
# 3. Preprocessing pipeline
#    OneHotEncode categoricals, scale numerics. Fit ONCE, reuse for both models.
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

# ---------------------------------------------------------------------------
# 4. Train/test split (classifier)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class
)

print("\nTraining classifier (is_delayed)...")
clf_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)
clf_pipeline.fit(X_train, y_train)

y_pred = clf_pipeline.predict(X_test)
y_proba = clf_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"  Accuracy : {acc:.3f}")
print(f"  Precision: {prec:.3f}")
print(f"  Recall   : {rec:.3f}")
print(f"  F1       : {f1:.3f}")
print(f"  ROC-AUC  : {auc:.3f}")
print()
print(classification_report(y_test, y_pred, target_names=["On Time", "Delayed"]))

# ---------------------------------------------------------------------------
# 5. Train regressor (delay_minutes), delayed flights only
# ---------------------------------------------------------------------------
print("Training regressor (delay_minutes | delayed)...")
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_delayed, y_minutes, test_size=0.2, random_state=RANDOM_STATE
)

reg_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)
reg_pipeline.fit(Xd_train, yd_train)

yd_pred = reg_pipeline.predict(Xd_test)
mae = mean_absolute_error(yd_test, yd_pred)
print(f"  MAE (minutes): {mae:.2f}")

# ---------------------------------------------------------------------------
# 6. Feature importance (from the classifier) — used to explain predictions
#    in the UI's "Contributing factors" panel.
# ---------------------------------------------------------------------------
print("\nComputing feature importances...")
ohe = clf_pipeline.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
all_feature_names = NUMERIC_FEATURES + cat_feature_names

importances = clf_pipeline.named_steps["model"].feature_importances_
importance_pairs = sorted(
    zip(all_feature_names, importances), key=lambda x: x[1], reverse=True
)

# Roll one-hot columns back up to their original human-readable field name
# (e.g. "weather_condition_Rain" -> "weather_condition") and sum their weight.
rolled = {}
for name, weight in importance_pairs:
    base = name
    for cat_col in CATEGORICAL_FEATURES:
        if name.startswith(cat_col + "_"):
            base = cat_col
            break
    rolled[base] = rolled.get(base, 0.0) + float(weight)

rolled_sorted = sorted(rolled.items(), key=lambda x: x[1], reverse=True)
FRIENDLY_NAMES = {
    "weather_condition": "Weather condition",
    "departure_hour": "Time of departure",
    "airline": "Airline",
    "is_peak_hour": "Peak hour traffic",
    "wind_speed_kph": "Wind speed",
    "visibility_km": "Visibility",
    "humidity_pct": "Humidity",
    "temperature_c": "Temperature",
    "is_holiday_period": "Holiday period",
    "is_weekend": "Weekend travel",
    "season": "Season",
    "month": "Month",
    "day_of_week": "Day of week",
    "origin": "Origin airport",
    "destination": "Destination airport",
    "flight_type": "Flight type",
    "distance_km": "Flight distance",
    "flight_duration_min": "Flight duration",
    "is_international": "International flight",
}
top_factors = [
    {"feature": k, "label": FRIENDLY_NAMES.get(k, k), "weight": round(v, 4)}
    for k, v in rolled_sorted[:8]
]

with open(MODEL_DIR / "feature_importance.json", "w") as f:
    json.dump(top_factors, f, indent=2)

print("Top contributing factors:")
for item in top_factors:
    print(f"  {item['label']:<24} {item['weight']:.4f}")

# ---------------------------------------------------------------------------
# 7. Save everything
# ---------------------------------------------------------------------------
print("\nSaving model artifacts...")
joblib.dump(clf_pipeline, MODEL_DIR / "classifier.pkl")
joblib.dump(reg_pipeline, MODEL_DIR / "regressor.pkl")

metadata = {
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "classifier_metrics": {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
    },
    "regressor_metrics": {"mae_minutes": round(mae, 2)},
    "delay_category_bins": {
        "No Delay": "0 min",
        "Minor (15-30 min)": "15-30 min",
        "Moderate (30-60 min)": "30-60 min",
        "Major (60+ min)": "60+ min",
    },
    "trained_rows": int(len(df)),
    "random_state": RANDOM_STATE,
}
with open(MODEL_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Done. Artifacts saved to {MODEL_DIR}/")
print("  - classifier.pkl")
print("  - regressor.pkl")
print("  - feature_importance.json")
print("  - metadata.json")
