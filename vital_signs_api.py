
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = FastAPI(
    title="Human Digital Twin - Vital Signs Prediction API",
    description="Predicts Risk Category (Low, Medium, High) from vital signs data",
    version="1.0.0"
)


origins = [
  "https://human-digital-twin-frontend.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS","PUT","DELETE"],
    allow_headers=["*"],
)
MODEL_FILE = "hdt_vitals_model.pkl"
SCALER_FILE = "hdt_vitals_scaler.pkl"


class VitalSigns(BaseModel):
    HeartRate: float
    RespiratoryRate: float
    BodyTemperature: float
    OxygenSaturation: float
    SystolicBP: float
    DiastolicBP: float
    Derived_HRV: float
    Derived_MAP: float
    Derived_BMI: float



def train_model():
    print(" Training model on Human Vital Signs Dataset...")

    
    df = pd.read_csv("human_vital_signs_dataset_2024.csv")

    
    X = df[[
        "Heart Rate",
        "Respiratory Rate",
        "Body Temperature",
        "Oxygen Saturation",
        "Systolic Blood Pressure",
        "Diastolic Blood Pressure",
        "Derived_HRV",
        "Derived_MAP",
        "Derived_BMI"
    ]]
    y = df["Risk Category"]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train_scaled, y_train)

    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(" Model trained and saved successfully!")


def load_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        train_model()
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print(" Model loaded successfully!")
    return model, scaler


model, scaler = load_model()



@app.get("/")
def root():
    return {
        "message": "🩺 Human Digital Twin - Vital Signs ML API is running",
        "endpoint": "/predict",
        "method": "POST",
        "sample_input": {
            "HeartRate": 85,
            "RespiratoryRate": 16,
            "BodyTemperature": 36.8,
            "OxygenSaturation": 97,
            "SystolicBP": 120,
            "DiastolicBP": 80,
            "Derived_HRV": 0.12,
            "Derived_MAP": 95.5,
            "Derived_BMI": 24.3
        }
    }


@app.post("/predict")
def predict(vitals: VitalSigns):
    try:
        features = [
            vitals.HeartRate,
            vitals.RespiratoryRate,
            vitals.BodyTemperature,
            vitals.OxygenSaturation,
            vitals.SystolicBP,
            vitals.DiastolicBP,
            vitals.Derived_HRV,
            vitals.Derived_MAP,
            vitals.Derived_BMI
        ]

        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]

        return {
            "input": vitals.dict(),
            "Predicted_Risk_Category": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



