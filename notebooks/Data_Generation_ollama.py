#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install colab-xterm')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'colabxterm')


# In[ ]:


get_ipython().run_line_magic('xterm', '')


# In[ ]:


get_ipython().system('pip install ollama')


# Generates synthetic health insurance claims using Ollama's LLM.
# 
# Ensures valid JSON responses using regex.
# 
# Retries up to 3 times if errors occur.
# 
# Processes data in batches (1000 records in total).
# 
# Saves the dataset incrementally in a CSV file.

# In[ ]:


import json
import pandas as pd
import ollama
import time
import re

def extract_json(text):
    """Extracts and returns only the JSON array from the response using regex."""
    json_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    return json_match.group(0) if json_match else None

def generate_claims(batch_size=10, retries=3):
    prompt = f"""
    Generate exactly {batch_size} **synthetic health insurance claim records** in JSON format.
    Each record must have:
    - Claim_type (string): Daycare/Domiciliary/Hospitalization/OPD
    - total_Claim_amount (float)
    - DOA (date)
    - DOD (date)
    - medicine_claim (float)
    - room_claim (float)
    - diagnostic_claim (float)
    - treatement_claim (float)
    - claim_status (string): Approved/Rejected/Pending
    - claim_date (date)
    - payment_mode (string): Credit/Debit/UPI/Cash
    - approval_date (date, nullable)
    - rejection_reason (string, nullable)
    - proximity_score (float)
    - Hospital_Name (string)
    - Hospital_Geo_Location (string) ex: latitude, longitude
    - Location (string)
    - integer_of_Beds (integer)
    - Hospital_Type (string): Network/Non-Network
    - Hospital_trust_score (float)
    - specialization (string)
    - Patient_unique_id (string)
    - Age (integer)
    - Income_range (float)
    - pre_existing_conditions (string):
    - patient_physical_disability (string): Yes/No
    - claim_frequeny_history (integer)
    - patient_geo_location (string) ex: latitude, longitude
    - patient_score (float)
    - gender (string): Male/Female
    - policy_type (string): Floater/Individual/Family/Group/Corporate
    - sum_assured (float)
    - sum_assured_left (float)
    - initial_startdate (date)
    - start_date (date)
    - end_date (date)
    - lockin_days (integer)
    - medicine_capping (float)
    - room_capping (float)
    - diagnostic_capping (float)
    - treatement_capping (float)
    - policy_claim_frequency (integer)
    - Policy_benefits (string) ex: Doctor Visits,Generic $10 co-pay,Annual Wellness Check-up,Mental Health Counseling,Emergency Room Visit, Ambulance coverage, Maternity and Newborn Care, Outpatient Surgery,Dental Coverage,Chronic Disease Management,Telemedicine Visits,Specialist Consultations,Rehabilitation Services
    - disease_type (string)
    - proximity_score (numerical)
    - cost (float)
    - duration (integer)
    - frequency (integer)
    - multiple_treatement (boolean)
    - excessive_prescriptions (boolean)
    - fraud_flag (integer): 1 for fraudulent claims, 0 for genuine claims

    **Output Instructions:**
    - ✅ Return ONLY a **valid JSON array** (e.g., `[{{...}}, {{...}}, ...]`).
    - ❌ Do NOT include explanations, comments, or Python code.
    """

    for attempt in range(retries):
        try:

            print(f"test")
            response = ollama.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                options={"format": "json"}  # 🔥 Force JSON output
            )

            content = response.get("message", {}).get("content", "").strip()

            # 🔍 Debug: Print first 500 characters of the response
            print(f"\n📢 Raw Response (First 500 chars):\n{content[:500]}\n")

            # 🔹 Extract JSON using regex
            extracted_json = extract_json(content)
            if not extracted_json:
                print(f"⚠️ Attempt {attempt+1}: No valid JSON found. Retrying...")
                time.sleep(1)
                continue

            # 🔹 Try parsing extracted JSON
            data = json.loads(extracted_json)

            # ✅ Ensure it's a list of dictionaries
            if isinstance(data, list) and all(isinstance(record, dict) for record in data):
                return data

            print(f"⚠️ Attempt {attempt+1}: Response is not a valid JSON list. Retrying...")
            time.sleep(1)

        except json.JSONDecodeError as e:
            print(f"⚠️ JSONDecodeError on attempt {attempt+1}: {e}\nExtracted content:\n{extracted_json[:500] if extracted_json else 'None'}")
            time.sleep(1)

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return []

    print("⚠️ Failed to generate data after retries.")
    return []

# === Generate Dataset in Batches ===
all_data = []
batch_size = 1  # 🔹 Start small to check stability
num_batches = 1 // batch_size

for i in range(num_batches):
    batch_data = generate_claims(batch_size)

    if batch_data:
        all_data.extend(batch_data)
        print(f"✅ Batch {i+1}/{num_batches} completed. Total records so far: {len(all_data)}")

        # Save partial data every batch
        pd.DataFrame(all_data).to_csv("health_insurance_fraud_dataset.csv", index=False)

    else:
        print(f"⚠️ Batch {i+1} failed. Skipping...")

# Final Check
if all_data:
    print("🎉 Dataset generated successfully!")
else:
    print("⚠️ No data generated. Check Ollama responses.")


# In[ ]:




