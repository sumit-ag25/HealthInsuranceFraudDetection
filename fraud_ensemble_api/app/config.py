# app/config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.joblib")
DL_MODEL_PATH = os.path.join(MODEL_DIR, "deep_learning_model.keras")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
