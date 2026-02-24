from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import requests

app = Flask(__name__)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
risk_model = joblib.load("risk_model.pkl")
disease_model = joblib.load("disease_model.pkl")
risk_cols = joblib.load("risk_columns.pkl")
disease_cols = joblib.load("disease_columns.pkl")

# --------------------------------------------------
# CITY COORDINATES (South India)
# --------------------------------------------------
CITY_COORDS = {
    "chennai": (13.0827, 80.2707),
    "coimbatore": (11.0168, 76.9558),
    "madurai": (9.9252, 78.1198),
    "bengaluru": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
    "kochi": (9.9312, 76.2673)
}

# --------------------------------------------------
# HOME PAGE (LOAD FRONTEND)
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# MANUAL PREDICTION
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.json

        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        rainfall = float(data["rainfall"])
        wind_speed = float(data["wind_speed"])
        season = data["season"]

        row = pd.DataFrame([{
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall,
            'wind_speed': wind_speed,
            'season_Monsoon': 1 if season == "Monsoon" else 0,
            'season_Summer': 1 if season == "Summer" else 0,
            'season_Winter': 1 if season == "Winter" else 0
        }])

        input_risk = row.reindex(columns=risk_cols, fill_value=0)
        input_disease = row.reindex(columns=disease_cols, fill_value=0)

        risk = risk_model.predict(input_risk)[0]
        disease = disease_model.predict(input_disease)[0]

        # Safety override
        if humidity < 55 and rainfall < 40:
            risk = "Low"
            disease = "Healthy"

        return jsonify({
            "risk": risk,
            "disease": disease
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# --------------------------------------------------
# LIVE WEATHER FETCH ONLY
# --------------------------------------------------
@app.route("/weather/<city>")
def get_weather(city):

    city = city.lower()

    if city not in CITY_COORDS:
        return jsonify({"error": "City not supported"})

    lat, lon = CITY_COORDS[city]

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"

    response = requests.get(url).json()

    if "current" not in response:
        return jsonify({"error": "Weather data unavailable"})

    current = response["current"]

    return jsonify({
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "rainfall": current["precipitation"],
        "wind_speed": current["wind_speed_10m"]
    })


# --------------------------------------------------
# AUTO PREDICTION USING LIVE WEATHER
# --------------------------------------------------
@app.route("/predict_city/<city>")
def predict_city(city):

    city = city.lower()

    if city not in CITY_COORDS:
        return jsonify({"error": "City not supported"})

    lat, lon = CITY_COORDS[city]

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"

    response = requests.get(url).json()

    if "current" not in response:
        return jsonify({"error": "Weather data unavailable"})

    current = response["current"]

    temperature = current["temperature_2m"]
    humidity = current["relative_humidity_2m"]
    rainfall = current["precipitation"]
    wind_speed = current["wind_speed_10m"]

    row = pd.DataFrame([{
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall,
        'wind_speed': wind_speed,
        'season_Monsoon': 0,
        'season_Summer': 1,
        'season_Winter': 0
    }])

    input_risk = row.reindex(columns=risk_cols, fill_value=0)
    input_disease = row.reindex(columns=disease_cols, fill_value=0)

    risk = risk_model.predict(input_risk)[0]
    disease = disease_model.predict(input_disease)[0]

    # Safety override
    if humidity < 55 and rainfall < 40:
        risk = "Low"
        disease = "Healthy"

    return jsonify({
        "city": city,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "wind_speed": wind_speed,
        "risk": risk,
        "disease": disease
    })


# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
