from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# =========================================================
# Create FastAPI app
# =========================================================

app = FastAPI(
    title="Air Quality Prediction API",
    description="FastAPI for Air Quality Prediction",
    version="1.0"
)

# =========================================================
# Load ML model
# =========================================================

model = joblib.load("gradient_boosting_air_quality.pkl")


# =========================================================
# Input model
# =========================================================

class AirQualityInput(BaseModel):
    country: str
    state: str
    city: str
    station: str
    latitude: float
    longitude: float
    pollutant_id: str
    last_update: str


# =========================================================
# GET 1 - Home
# =========================================================

@app.get("/")
def home():
    return {
        "message": "Air Quality Prediction API is running"
    }


# =========================================================
# GET 2 - API Information
# =========================================================

@app.get("/info")
def api_info():
    return {
        "project": "Air Quality Prediction",
        "model": "Gradient Boosting Regressor",
        "target": "pollutant_avg",
        "status": "Active"
    }


# =========================================================
# POST 1 - Prediction
# =========================================================

@app.post("/predict")
def predict(data: AirQualityInput):

    input_data = pd.DataFrame([{
        "country": data.country,
        "state": data.state,
        "city": data.city,
        "station": data.station,
        "latitude": data.latitude,
        "longitude": data.longitude,
        "pollutant_id": data.pollutant_id,
        "last_update": data.last_update
    }])

    # Convert date
    input_data["last_update"] = pd.to_datetime(
        input_data["last_update"],
        errors="coerce"
    )

    # Create date features
    input_data["year"] = input_data["last_update"].dt.year
    input_data["month"] = input_data["last_update"].dt.month
    input_data["day"] = input_data["last_update"].dt.day
    input_data["hour"] = input_data["last_update"].dt.hour
    input_data["day_of_week"] = input_data["last_update"].dt.dayofweek

    # Remove original date
    input_data.drop(
        columns=["last_update"],
        inplace=True
    )

    # Prediction
    prediction = model.predict(input_data)

    return {
        "predicted_pollutant_avg": round(
            float(prediction[0]), 2
        )
    }


# =========================================================
# POST 2 - Prediction Test
# =========================================================

@app.post("/predict-test")
def predict_test(data: AirQualityInput):

    return {
        "message": "Prediction request received successfully",
        "city": data.city,
        "state": data.state,
        "pollutant": data.pollutant_id,
        "status": "Ready for prediction"
    }