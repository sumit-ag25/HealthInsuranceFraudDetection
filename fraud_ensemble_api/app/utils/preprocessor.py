# app/utils/preprocessor.py
import joblib
import pandas as pd
from app.config import SCALER_PATH

scaler = joblib.load(SCALER_PATH)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return scaler.transform(df.to_numpy())