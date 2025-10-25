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
model = joblib.load("penguin_logreg_pipeline.pkl")

class_names = ["Adelie", "Chinstrap", "Gentoo"]   # same order as sklearn


@app.get("/")
def root():
    return {"message": "send measurements to /predict"}


# @app.post("/predict")
# def predict(bill_length_mm: float,
#             bill_depth_mm: float,
#             flipper_length_mm: float,
#             body_mass_g: float):
#     """
#     Predict penguin species from four numeric features.
#     """
#     X = pd.DataFrame([[bill_length_mm, bill_depth_mm,
#                        flipper_length_mm, body_mass_g]],
#                      columns=["bill_length_mm", "bill_depth_mm",
#                               "flipper_length_mm", "body_mass_g"])
#     # pred_idx = model.predict(X)[0]
#     # probs = model.predict_proba(X)[0]
#     # species = class_names[pred_idx]
#     # confidence = float(probs.max())
#     # return {"species": species, "confidence": confidence}
#     pred_species = model.predict(X)[0]          # already a string
#     probs = model.predict_proba(X)[0]
#     confidence = float(probs.max())
#     return {"species": pred_species, "confidence": confidence}

@app.post("/predict", response_model=PredictionOut)
def predict(measures: PenguinMeasures):
    X = pd.DataFrame([measures.dict()])
    pred_species = model.predict(X)[0]
    confidence = float(model.predict_proba(X).max())
    return PredictionOut(species=pred_species, confidence=confidence)