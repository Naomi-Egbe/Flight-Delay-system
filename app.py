"""
app.py
=======
Flask backend for the FAAN ENU Flight Delay Prediction System.

Routes:
    GET  /            -> serves the dashboard (templates/index.html)
    POST /predict      -> runs the trained model on submitted flight details
    GET  /health         -> simple health check
    GET  /meta            -> returns model metadata (accuracy, features) for display

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_DIR = Path("model")

# ---------------------------------------------------------------------------
# Load model artifacts ONCE at startup (not per-request — that would be slow)
# ---------------------------------------------------------------------------
classifier = None
regressor = None
feature_importance = []
metadata = {}

try:
    classifier = joblib.load(MODEL_DIR / "classifier.pkl")
    regressor = joblib.load(MODEL_DIR / "regressor.pkl")
    with open(MODEL_DIR / "feature_importance.json") as f:
        feature_importance = json.load(f)
    with open(MODEL_DIR / "metadata.json") as f:
        metadata = json.load(f)
    print("Model artifacts loaded successfully.")
except FileNotFoundError:
    print("WARNING: model artifacts not found. Run `python train_model.py` first.")

NUMERIC_FEATURES = metadata.get("numeric_features", [])
CATEGORICAL_FEATURES = metadata.get("categorical_features", [])

DELAY_CATEGORY_BOUNDARIES = [
    (0, 15, "No Delay"),
    (15, 30, "Minor (15-30 min)"),
    (30, 60, "Moderate (30-60 min)"),
    (60, float("inf"), "Major (60+ min)"),
]


def categorize_delay(minutes: float) -> str:
    for low, high, label in DELAY_CATEGORY_BOUNDARIES:
        if low <= minutes < high:
            return label
    return "Major (60+ min)"


def derive_calendar_features(flight_date: str, scheduled_departure: str) -> dict:
    """
    Given a date (YYYY-MM-DD) and time (HH:MM), derive the calendar-based
    features the model expects: day_of_week, month, season, departure_hour,
    is_weekend, is_peak_hour.
    """
    dt = pd.to_datetime(flight_date)
    day_of_week = dt.day_name()
    month = dt.month_name()
    is_weekend = 1 if dt.dayofweek >= 5 else 0

    # Nigeria's rough dry/rainy season split (used consistently with the
    # training dataset's `season` labeling — adjust here if your dataset
    # defines season boundaries differently).
    if dt.month in (11, 12, 1, 2):
        season = "Dry Season"
    elif dt.month in (3,):
        season = "Harmattan"
    else:
        season = "Rainy Season"

    departure_hour = int(scheduled_departure.split(":")[0])
    is_peak_hour = 1 if departure_hour in (6, 7, 8, 17, 18, 19) else 0

    return {
        "day_of_week": day_of_week,
        "month": month,
        "season": season,
        "departure_hour": departure_hour,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": classifier is not None,
    })


@app.route("/meta")
def meta():
    return jsonify(metadata)


@app.route("/predict", methods=["POST"])
def predict():
    if classifier is None or regressor is None:
        return jsonify({
            "error": "Model not loaded. Run train_model.py first, then restart the server."
        }), 503

    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Missing or invalid JSON body"}), 400

    required = [
        "airline", "origin", "destination", "flight_date", "scheduled_departure",
        "weather_condition", "temperature_c", "visibility_km", "wind_speed_kph",
        "humidity_pct",
    ]
    missing = [f for f in required if f not in payload or payload[f] in (None, "")]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        calendar_features = derive_calendar_features(
            payload["flight_date"], payload["scheduled_departure"]
        )

        origin = payload["origin"]
        destination = payload["destination"]
        is_international = 1 if payload.get("is_international") else 0
        # Simple domestic/international default: Addis Ababa (ADD) is the
        # only non-Nigerian airport in the dataset's route set.
        if "ADD" in (origin, destination):
            is_international = 1

        row = {
            "departure_hour": calendar_features["departure_hour"],
            "flight_duration_min": float(payload.get("flight_duration_min", 60)),
            "distance_km": float(payload.get("distance_km", 450)),
            "temperature_c": float(payload["temperature_c"]),
            "visibility_km": float(payload["visibility_km"]),
            "wind_speed_kph": float(payload["wind_speed_kph"]),
            "humidity_pct": float(payload["humidity_pct"]),
            "is_weekend": calendar_features["is_weekend"],
            "is_peak_hour": calendar_features["is_peak_hour"],
            "is_holiday_period": 1 if payload.get("is_holiday_period") else 0,
            "is_international": is_international,
            "day_of_week": calendar_features["day_of_week"],
            "month": calendar_features["month"],
            "season": calendar_features["season"],
            "airline": payload["airline"],
            "flight_type": "International" if is_international else "Domestic",
            "origin": origin,
            "destination": destination,
            "weather_condition": payload["weather_condition"],
        }

        X = pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        is_delayed = bool(classifier.predict(X)[0])
        confidence = float(classifier.predict_proba(X)[0][1 if is_delayed else 0])

        if is_delayed:
            predicted_minutes = float(regressor.predict(X)[0])
            predicted_minutes = max(predicted_minutes, 1.0)
            delay_category = categorize_delay(predicted_minutes)
        else:
            predicted_minutes = 0.0
            delay_category = "No Delay"

        factors = [
            {"label": item["label"], "weight": f"{item['weight'] * 100:.1f}%"}
            for item in feature_importance[:5]
        ]

        return jsonify({
            "is_delayed": is_delayed,
            "confidence": round(confidence, 3),
            "predicted_delay_minutes": round(predicted_minutes, 1),
            "delay_category": delay_category,
            "factors": factors,
        })

    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)