import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MLflow-registry scorer")

MODEL_NAME = "PenguinRF"
# MODEL_STAGE = "Production"     # or "Staging", "1", "2" …
MODEL_VERSION = "1"  # instead of "Production"
# model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
# model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
# model = joblib.load(os.getenv("MODEL_PATH", "penguin_auto.pkl"))
model = joblib.load(os.getenv("MODEL_PATH", "penguin_xgb_tuned.pkl"))
# model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
# model = mlflow.pyfunc.load_model("penguin_rf_tuned.pkl")   # plain file


class Measures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float


@app.post("/predict")
def predict(meas: Measures):
    X = pd.DataFrame([meas.dict()])
    pred, probs = model.predict(X)[0], model.predict_proba(X)[0]
    return {"species": pred, "confidence": float(probs.max())}
