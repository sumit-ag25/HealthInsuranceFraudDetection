from fastapi import FastAPI
from app.api.v1.ensemble_endpoint import router as ensemble_router

app = FastAPI(
    title="Health Care Fraud Detection API",
    description="🌐 Predict health insurance fraud using 3 different models.\n\n"
                "📦 Models: XGBoost, Random Forest, Deep Learning \n\n"
                "🧠 Input the feature json and get prediction results. \n\n"
                "👤 Team Members: \n\n"
                "Sumit Kumar Agarwal (ID: 2023AIML512)\n"
                "Manu Saxena         (ID: 2023AIML506)\n"
                "Prabhujyot Singh    (ID: 2023AIML584)\n"
                "Vishal Gupta        (ID: 2023AIML518)",
    version="1.0",
    contact={
        "name": "BITS Pilani",
        "url": "https://www.bits-pilani.ac.in/"
    }
)

app.include_router(ensemble_router, prefix="/ensemble")