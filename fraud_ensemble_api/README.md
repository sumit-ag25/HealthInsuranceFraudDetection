## HealthCare Fraud Detection API ##

# Launch any prompt terminal where python is installed and navigate to fraud_ensemble_api folder. (Avoid spaces in path name)

# Create a virtual environment and activate (optional but recommended).
python -m venv venv

venv\Scripts\activate (Windows) or source venv/bin/activate (Mac-OS)

# Install dependencies.
pip install -r requirements.txt

# Run the app.
uvicorn app.main:app --reload --port 8000

Once the application is started successfully, it will launch API suite on http://127.0.0.1:8000/docs

# Click on ensemble predict model and Try it out.

# The model expects the json input structure as defined in `schemas.py`. 

# Example: Sample Fradulent Json

{ 
   "frequency": 3.0, 
   "total_claim_amount": 120.0, 
   "room_capping": 60.0, 
   "treatment_claim": 20.0, 
   "medicine_capping": 50.0, 
   "lockin_days": 30.0, 
   "treatment_capping": 20.0, 
   "room_claim": 50.0, 
   "claim_frequeny_history": 2.0, 
   "Hospital_trust_score": 4.2, 
   "cost": 100.0, 
   "sum_assured": 10000.0, 
   "sum_assured_left": 9000.0, 
   "medicine_claim": 40.0, 
   "diagnostic_capping": 40.0, 
   "proximity_score": 85.0, 
   "diagnostic_claim": 30.0, 
   "patient_score": 92.0, 
   "policy_claim_frequency": 2.0, 
   "Income_range": 120000.0, 
   "Age": 45, 
   "duration": 7.0, 
   "claim_type_encoded": 1, 
   "multiple_treatement_encoded": 0, 
   "integer_of_Beds": 150.0
}

# Response Format:

{
  "Prediction Summary": "Transaction is classified as Fraudulent by XGBoost, Random Forest, Deep Learning.",
  "Model Predictions": {
    "XGBoost": {
      "Prediction": "Fraudulent",
      "Fraud Probability": 0.844,
      "Non-Fraud Probability": 0.156
    },
    "Random Forest": {
      "Prediction": "Fraudulent",
      "Fraud Probability": 0.75,
      "Non-Fraud Probability": 0.25
    },
    "Deep Learning": {
      "Prediction": "Fraudulent",
      "Fraud Probability": 0.999,
      "Non-Fraud Probability": 0.001
    }
  }
}

# Example: Sample Non-Fradulent Json

{
	"frequency": 2.0, 
	"total_claim_amount": 300.0, 
	"room_capping": 200.0, 
	"treatment_claim": 100.0, 
	"medicine_capping": 150.0, 
	"lockin_days": 30.0, 
	"treatment_capping": 50.0, 
	"room_claim": 100.0, 
	"claim_frequeny_history": 3.0, 
	"Hospital_trust_score": 4.5, 
	"cost": 500.0, 
	"sum_assured": 2000.0, 
	"sum_assured_left": 1500.0, 
	"medicine_claim": 100.0, 
	"diagnostic_capping": 50.0, 
	"proximity_score": 0.8, 
	"diagnostic_claim": 30.0, 
	"patient_score": 7.0, 
	"policy_claim_frequency": 1.0, 
	"Income_range": 50000.0, 
	"Age": 35.0, 
	"duration": 12.0, 
	"claim_type_encoded": 1.0, 
	"multiple_treatement_encoded": 0.0, 
	"integer_of_Beds": 100.0
}

# Response Format:

{
  "Prediction Summary": "Transaction is classified as Non-Fraudulent by XGBoost, Random Forest, Deep Learning.",
  "Model Predictions": {
    "XGBoost": {
      "Prediction": "Non-Fraudulent",
      "Fraud Probability": 0.251,
      "Non-Fraud Probability": 0.749
    },
    "Random Forest": {
      "Prediction": "Non-Fraudulent",
      "Fraud Probability": 0.16,
      "Non-Fraud Probability": 0.84
    },
    "Deep Learning": {
      "Prediction": "Non-Fraudulent",
      "Fraud Probability": 0,
      "Non-Fraud Probability": 1
    }
  }
}

👤 Author(s)
Sumit Kumar Agarwal
Manu Saxena
Prabhujyot Singh
Vishal Gupta