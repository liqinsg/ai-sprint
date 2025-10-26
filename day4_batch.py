from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd

app = FastAPI(title="Penguin Batch API", version="0.1.0")
model = joblib.load("penguin_logreg_pipeline.pkl")


class PenguinRow(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float


class BatchIn(BaseModel):
    rows: List[PenguinRow]


class BatchOut(BaseModel):
    # each dict = {"species": str, "confidence": float}
    predictions: List[dict]


@app.post("/predict_batch", response_model=BatchOut)
def predict_batch(batch: BatchIn):
    df = pd.DataFrame([r.dict() for r in batch.rows])
    preds = model.predict(df)
    probs = model.predict_proba(df)
    results = [{"species": pred, "confidence": float(prob.max())}
               for pred, prob in zip(preds, probs)]
    return BatchOut(predictions=results)

    """ POST /predict_batch example
curl -X 'POST' \
  'http://127.0.0.1:8000/predict_batch' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "rows": [
    {"bill_length_mm": 39.5, "bill_depth_mm": 18.8, "flipper_length_mm": 196, "body_mass_g": 4100},
    {"bill_length_mm": 48.5, "bill_depth_mm": 15.2, "flipper_length_mm": 220, "body_mass_g": 5600}
  ]
}'
Response:
{
  "predictions": [
    {
      "species": "Adelie",
      "confidence": 0.9872545901820577
    },
    {
      "species": "Gentoo",
      "confidence": 0.9949628473605735
    }
  ]
}
    """
