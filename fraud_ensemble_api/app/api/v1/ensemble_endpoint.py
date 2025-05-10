# app/api/v1/ensemble_endpoint.py
from fastapi import APIRouter
import pandas as pd
from app.models.schemas import InputData
from app.services.ensemble_service import predict_all_models

router = APIRouter(tags=["Ensemble Model"])

@router.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    results = predict_all_models(df)

    model_preds = {
        name: ("Fraudulent" if res["prediction"] == 1 else "Non-Fraudulent")
        for name, res in results.items()
    }

    summary_parts = []
    fraudulent = [name for name, pred in model_preds.items() if pred == "Fraudulent"]
    nonfraud = [name for name in model_preds if name not in fraudulent]
    if fraudulent:
        summary_parts.append(f"Fraudulent by {', '.join(fraudulent)}")
    if nonfraud:
        summary_parts.append(f"Non-Fraudulent by {', '.join(nonfraud)}")
    summary = "Transaction is classified as " + " but as ".join(summary_parts) + "."

    return {
        "Prediction Summary": summary,
        "Model Predictions": {
            name: {
                "Prediction": model_preds[name],
                "Fraud Probability": round(float(res["proba"]), 3),
                "Non-Fraud Probability": round(1 - float(res["proba"]), 3)
            }
            for name, res in results.items()
        }
    }
