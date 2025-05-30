# Steps to Reproduce:

1. Run Data_Generation_ollama.ipynb present inside 'notebooks' folder to create realisitic dataset.

2. Open Health_Insurance_Dataset.csv present inside 'dataset' folder to validate the data.

3. Run EDA_ModelBuilding.ipynb present inside 'notebooks' folder for EDA, preprocessing, model building & evaluation, saving the model.

4. The final API is in the 'fraud_ensemble_api' folder. Use the README.md inside the folder for deployment steps.


Datasets/Dictionaries Prepared:
-------------------------------
- dataset/Health_Insurance_Dataset.csv (Generated using Data_generation_ollama.ipynb)

Code:
-----
- notebooks/Data_Generation_ollama.ipynb
- notebooks/Data_Generation_ollama.py

- notebooks/EDA_ModelBuilding.ipynb
- notebooks/EDA_ModelBuilding.py


Summary:
-----------
Built a supervised ML model (Scikit-learn) to classify fraudulent health claims, using SMOTE for class imbalance andprecision-recall for evaluation.
Developed a RESTful API using FastAPI and deployed the model with Joblib for real-time prediction.
Achieved 99% accuracy across the 3 models


👤 Author(s)
Sumit Kumar Agarwal


