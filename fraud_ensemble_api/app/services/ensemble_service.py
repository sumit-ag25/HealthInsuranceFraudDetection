# app/services/ensemble_service.py
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from tensorflow import keras
from app.utils.preprocessor import preprocess
from app.config import SCALER_PATH, RF_MODEL_PATH, DL_MODEL_PATH, XGB_MODEL_PATH

model_xgb = XGBClassifier()
model_xgb.load_model(XGB_MODEL_PATH)

model_rf = joblib.load(RF_MODEL_PATH)
model_dl = keras.models.load_model(DL_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_all_models(data: pd.DataFrame):
    features_scaled = preprocess(data)

    # XGBoost
    pred_xgb = model_xgb.predict(features_scaled)[0]
    proba_xgb = model_xgb.predict_proba(features_scaled)[0][1]

    # Random Forest
    pred_rf = model_rf.predict(features_scaled)[0]
    proba_rf = model_rf.predict_proba(features_scaled)[0][1]

    # Deep Learning
    proba_dl = float(model_dl.predict(features_scaled)[0][0])
    pred_dl = 1 if proba_dl > 0.5 else 0

    return {
        "XGBoost": {"prediction": pred_xgb, "proba": proba_xgb},
        "Random Forest": {"prediction": pred_rf, "proba": proba_rf},
        "Deep Learning": {"prediction": pred_dl, "proba": proba_dl}
    }