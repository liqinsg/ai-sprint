from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

from pydantic import BaseModel, Field


class PenguinMeasures(BaseModel):
    bill_length_mm: float = Field(..., gt=0, example=39.5)
    bill_depth_mm: float = Field(..., gt=0, example=18.8)
    flipper_length_mm: float = Field(..., gt=0, example=196)
    body_mass_g: float = Field(..., gt=0, example=4100)


class PredictionOut(BaseModel):
    species: str
    confidence: float


app = FastAPI(title="Penguin Classifier API",
              description="Logistic-regression model trained on Day 2",
              version="0.1.0")

# load model once at start-up
# model = joblib.load("penguin_logreg_pipeline.pkl")
# load the model retrained on Day 5 which is trainned on github with auto workflow
model = joblib.load("penguin_auto.pkl")

class_names = ["Adelie", "Chinstrap", "Gentoo"]   # same order as sklearn


@app.get("/")
def root():
    return {"message": "send measurements to /predict"}


@app.post("/predict", response_model=PredictionOut)
def predict(measures: PenguinMeasures):
    X = pd.DataFrame([measures.dict()])
    pred_species = model.predict(X)[0]
    confidence = float(model.predict_proba(X).max())
    return PredictionOut(species=pred_species, confidence=confidence)

    """ Example curl command:
(ai-sprint) qili@NBK202500000057:~/ai-sprint$ curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "bill_length_mm": 39.5,
  "bill_depth_mm": 18.8,
  "flipper_length_mm": 196,
  "body_mass_g": 4100
}'
{"species":"Adelie","confidence":0.9936724776333471}
"""
