from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

import pandas as pd
import joblib


# ============================================================
# 1. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Air Quality Prediction API",
    description="Machine Learning API for Air Quality Prediction using XGBoost",
    version="1.0.0"
)


# ============================================================
# 2. MODEL PATH
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
APP_DIR = BACKEND_DIR.parent

MODEL_PATH = (
    APP_DIR
    / "pkl file"
    / "xgboost_air_quality.pkl"
)


# ============================================================
# 3. LOAD MODEL
# ============================================================

model = None

print("\n========================================")
print("AIR QUALITY PREDICTION API")
print("========================================")

print("Model path:")
print(MODEL_PATH)

try:

    model = joblib.load(MODEL_PATH)

    print("\nMODEL LOADED SUCCESSFULLY")

except Exception as e:

    print("\nMODEL LOADING FAILED")
    print("Error:", repr(e))


# ============================================================
# 4. INPUT DATA MODEL
# ============================================================

class AirQualityInput(BaseModel):

    country: str
    state: str
    city: str
    station: str

    latitude: float
    longitude: float

    pollutant_id: str

    last_update: str


# ============================================================
# 5. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Air Quality Prediction API is running",
        "status": "success"
    }


# ============================================================
# 6. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    if model is None:

        return {
            "status": "unhealthy",
            "model_loaded": False
        }

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# 7. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(data: AirQualityInput):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Machine learning model is not loaded"
        )


    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    input_data["last_update"] = pd.to_datetime(
        input_data["last_update"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Create same date features used during training
    # --------------------------------------------------------

    input_data["year"] = input_data["last_update"].dt.year

    input_data["month"] = input_data["last_update"].dt.month

    input_data["day"] = input_data["last_update"].dt.day

    input_data["hour"] = input_data["last_update"].dt.hour

    input_data["day_of_week"] = (
        input_data["last_update"].dt.dayofweek
    )


    # --------------------------------------------------------
    # Remove original date column
    # --------------------------------------------------------

    input_data.drop(
        columns=["last_update"],
        inplace=True
    )


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "message": "Prediction successful",

        "predicted_pollutant_avg": round(
            float(prediction),
            2
        ),

        "target": "pollutant_avg",

        "model": "XGBoost"

    }


# ============================================================
# 8. TEST ENDPOINT
# ============================================================

@app.post("/predict-test")
def predict_test(data: AirQualityInput):

    return {

        "message": "Prediction input received successfully",

        "country": data.country,

        "state": data.state,

        "city": data.city,

        "station": data.station,

        "pollutant_id": data.pollutant_id,

        "status": "valid"

    }