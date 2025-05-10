#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Import necessary libraries.
import shap
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.stats import chi2_contingency
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
df = pd.read_csv('Health_Insurance_Dataset.csv')

# Display basic info
df.info()


# In[2]:


# display the first five entries
df.head()


# In[3]:


# Summary statistics
df.describe()


# In[4]:


# Check for missing values
df.isnull().sum()


# In[5]:


# Check for datatypes
df.dtypes


# In[6]:


# Check distribution of each categorical features
categorical_features = df.select_dtypes(include=['object']).columns
for col in categorical_features:
    print(f'\n{col} Distribution:')
    print(df[col].value_counts())
    print()


# In[7]:


# Plot histograms for numeric columns to check distibution
numeric_features = df.select_dtypes(include=[np.number]).columns
df[numeric_features].hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()


# In[8]:


# Distribution of Total Claim Amount
plt.figure(figsize=(40, 6))
sns.histplot(df['total_claim_amount'], bins=30, kde=True)
plt.title('Distribution of Total Claim Amount')
plt.xlabel('Total Claim Amount')
plt.ylabel('Frequency')
plt.show()



# In[9]:


#Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_features].corr(), annot=True, cmap='coolwarm', linewidths=0.5,fmt=".1f")
plt.title('Correlation Heatmap')
plt.show()


# In[10]:


# Pairplot for key numerical features to see potential correlations
sns.pairplot(df[['total_claim_amount', 'medicine_claim', 'cost', 'sum_assured', 'Age']])
plt.suptitle('Pairplot of Key Features', y=1.02)
plt.show()


# In[11]:


# BoxPlot to understand the different policy type with claim amount
plt.figure(figsize=(10, 6))
sns.boxplot(x='policy_type', y='total_claim_amount', data=df)
plt.title('Claim Amount vs. Policy Type')
plt.xlabel('Policy Type')
plt.ylabel('Total Claim Amount')
plt.show()


# In[12]:


# Plot to understand the claim type and their status
plt.figure(figsize=(10, 6))
sns.countplot(x='claim_type', hue='claim_status', data=df)
plt.title('Claim Status by Claim Type')
plt.xlabel('Claim Type')
plt.ylabel('Count')
plt.show()


# In[13]:


# Boxplot to visualize the distribution of treatment claim amounts
plt.figure(figsize=(12, 8))
sns.boxplot(x='disease_type', y='treatment_claim', data=df)
plt.title('Treatment Claim by Disease Type')
plt.xlabel('Disease Type')
plt.ylabel('Treatment Claim Amount')
plt.xticks(rotation=45)
plt.show()


# In[14]:


# Ensure 'total_Claim_amount' is numeric, coercing errors to NaN
df['total_claim_amount'] = pd.to_numeric(df['total_claim_amount'], errors='coerce')

# Box Plot - Total Claim Amount by Claim Status
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='claim_status', y='total_claim_amount')
plt.title('Total Claim Amount Distribution by Claim Status')
plt.show()


# In[15]:


# Function to perform Chi-Square test to check relationship between few features and the claim status
def chi_square_test(df, feature):
    contingency_table = pd.crosstab(df[feature], df['claim_status'])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Square Test for '{feature}' and 'claim_status'")
    print(f"Chi-Square Statistic: {chi2}")
    print(f"P-value: {p_value}")
    print("Result: Significant relationship found" if p_value < 0.05 else "Result: No significant relationship found")
    print("-" * 60)

# Running Chi-Square Test on claim-related features
chi_square_test(df, 'claim_type')
chi_square_test(df, 'payment_mode')
chi_square_test(df, 'rejection_reason')
chi_square_test(df, 'Hospital_Type')


# Feature selection using algorithms:
# 1) SelectKBest
# SelectKBest(score_func=f_classif) uses an ANOVA F-test to check how well each independent variable explains the variance in the target variable (claim_status).
# The higher the score, the more statistically significant the feature is in predicting the target.
# 
# Random forest :
# 
# Measures how often a feature was used to split the data in the decision trees, weighted by how much it improves classification.
# Values sum to 1.
# Zero means the feature was never used by the model.
# 
# Lasso applies L1 regularization, which shrinks coefficients and sets some to exactly zero.
# The closer to zero, the less important the feature is.
# Features with 0 importance are essentially dropped by the model.

# In[17]:


# Drop these columns from the original DataFrame
columns_to_drop = ['Hospital_Name', 'Location', 'Patient_unique_id']
df = df.drop(columns=columns_to_drop)

# List of categorical columns to encode
columns_to_encode = ["claim_type", "claim_status", "payment_mode", 
                     "Hospital_Type", "gender", "patient_physical_disability", "multiple_treatement", "rejection_reason", "excessive_prescriptions"]
# Initialize LabelEncoders for each column
label_encoders = {col: LabelEncoder() for col in columns_to_encode}

# Apply Label Encoding and create new columns
for col in columns_to_encode:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col])

# Select only numerical columns but EXCLUDE 'Fraud_Flag'
numerical_cols = [col for col in df.select_dtypes(include=['int64', 'float64', 'int32']).columns if col != 'fraud_flag']


# In[18]:


# Split the data into training and testing sets first
X = df[numerical_cols]
y = df['fraud_flag']

# First, split into train and test sets (90% train, 10% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

# Then, split the remaining data into train and validation sets (70% train, 20% validation of the 90% remaining)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20/0.90, random_state=42, stratify=y_train_val)

# Print the sizes to confirm the split
print(f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}, Test set size: {len(X_test)}")

# Initialize the imputer (Using Mean Strategy)
imputer = SimpleImputer(strategy="mean") 

# Fit the imputer on the training set only
X_train_imputed = imputer.fit_transform(X_train)

# Apply the imputer to the validation and test sets (using the fitted imputer)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)


# Ensure y_train has valid integer labels
y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)


# In[19]:


# Feature Selection using SelectKBest
k_best = SelectKBest(score_func=f_classif, k='all')  
k_best.fit(X_train_imputed, y_train)  # Fit on train data only
kbest_feature_importance = pd.Series(
    k_best.scores_, index=X_train.columns  # Ensure correct feature indexing
).sort_values(ascending=False)

# Feature Selection using Random Forest
#rf = RandomForestClassifier(class_weight='balanced')
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_imputed, y_train)
rf_feature_importance = pd.Series(
    rf.feature_importances_, index=X_train.columns  # No dependency on KBest
).sort_values(ascending=False)

# Feature Selection using Lasso (L1 Regularization)
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_imputed, y_train)
lasso_feature_importance = pd.Series(
    np.abs(lasso.coef_), index=X_train.columns  # No dependency on KBest
).sort_values(ascending=False)

# Display SelectKBest Feature Importance
print("SelectKBest Feature Importance:\n")
print(kbest_feature_importance)

# Display Random Forest Feature Importance
print("\nRandom Forest Feature Importance:\n"), 
print(rf_feature_importance)

# Display Lasso Feature Importance
print("\nLasso (L1) Feature Importance:\n")
print(lasso_feature_importance)


# In[20]:


# SHAP Explainer for Tree-Based Model to check feature importance 
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train_imputed)  # SHAP values for all classes

# Compute mean absolute SHAP values for each feature
shap_importance = np.abs(shap_values[:, :, 1]).mean(axis=0)  # Taking mean across all samples

# Create a sorted list of feature importance
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'SHAP Importance': shap_importance})
feature_importance = feature_importance.sort_values(by='SHAP Importance', ascending=False)

# Print top features
print("\n SHAP Feature Importance:\n")
print(feature_importance)

# 🔹 Convert SHAP values to 2D for visualization 
shap_values_class_1 = shap_values[:, :, 1]  # Extract values for Fraud (Class 1)

# 🔹 SHAP Summary Plot for Class 1 (Fraud)
shap.summary_plot(shap_values_class_1, X_train_imputed)

# Extract SHAP values for Class 0 (Non-fraud)
shap_values_class_0 = shap_values[:, :, 0]  # Extract values for Class 0 (Non-fraud)

# SHAP Summary Plot for Class 0
shap.summary_plot(shap_values_class_0, X_train_imputed)


# In[21]:


#Calculating PCA
pca = PCA(0.80)

pct = pca.fit_transform(X_train_imputed)

print("Eigenvalues")
print(pca.explained_variance_)
#print()
#print('eigen vectors')
#print(pca.components_)

print('\n Explained Variance Ratio')
print(pca.explained_variance_ratio_)
print('---------------------')

# Plotting the cumulative explained variance
cumulative_variance = pca.explained_variance_ratio_.cumsum()
plt.plot(cumulative_variance, marker='o')
plt.xlabel("Number of Components")
plt.ylabel('Cumulative Explained Variance')
plt.title("Cumulative Explained Variance Plot")
plt.show()


#print(pct.shape)

#X_train_pca, X_test_pca, y_train, y_test = train_test_split(pct, y, test_size=0.2, random_state=42)


# In[22]:


# Apply Truncated SVD (keeping top 5 components based on pca analysis)
k = min(5, X_train.shape[1])
svd = TruncatedSVD(n_components=k)
svd_transformed = svd.fit_transform(X_train_imputed)

# Explained variance ratio
explained_variance = svd.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

# Plot explained variance
plt.figure(figsize=(8,5))
plt.plot(range(1, k+1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('SVD - Explained Variance')
plt.show()

# Create a separate DataFrame for the SVD components
svd_df = pd.DataFrame(svd_transformed, columns=[f'SVD_Component_{i+1}' for i in range(k)])

# check first five entries
svd_df.head()



# In[23]:


# Print the first few rows of the SVD transformed values
svd_transformed


# In[24]:


# Select top 25 features for model building based on importance seen through each algorithm.
selected_features = kbest_feature_importance.head(25).index.tolist()

# Get the indices of the selected features
selected_features_numeric = [X_train.columns.get_loc(feature) for feature in selected_features]

# Select the columns corresponding to the selected features from the imputed data
X_train_selected = X_train_imputed[:, selected_features_numeric]
X_test_selected = X_test_imputed[:, selected_features_numeric]
X_val_selected = X_val_imputed[:, selected_features_numeric]


# Standardization
scaler = StandardScaler()

# Fit the scaler on the training set only
X_train_scaled = scaler.fit_transform(X_train_selected)

# Apply the same scaler to the validation and test sets
X_val_scaled = scaler.transform(X_val_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Save the scaler to use it during inference
joblib.dump(scaler, "scaler.joblib")    


# In[25]:


# Initialize and train Logistic Regression
log_reg = LogisticRegression(solver='lbfgs')  # Standard solver for binary classification
log_reg.fit(X_train_scaled, y_train)

# Make predictions on validation data
y_pred_LR = log_reg.predict(X_val_scaled)

# Evaluate model
print("Accuracy:", accuracy_score(y_val, y_pred_LR))
print(classification_report(y_val, y_pred_LR))

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_pred_LR)
print("Confusion Matrix:\n", conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-FRAUD', 'FRAUD'], yticklabels=['Non-FRAUD', 'FRAUD'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC-AUC score (Receiver Operating Characteristic - Area Under Curve)
roc_auc = roc_auc_score(y_val, log_reg.predict_proba(X_val_scaled)[:, 1])  # Predict probabilities for class 1
print("ROC-AUC Score:", roc_auc)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, log_reg.predict_proba(X_val_scaled)[:, 1])
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='b', label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[26]:


# Make predictions
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_scaled, y_train)
y_pred_RF = rf_selected.predict(X_val_scaled)

# Evaluate model
print("Accuracy:", accuracy_score(y_val, y_pred_RF))
print("Classification Report:\n", classification_report(y_val, y_pred_RF))

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_pred_RF)
print("Confusion Matrix:\n", conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-FRAUD', 'FRAUD'], yticklabels=['Non-FRAUD', 'FRAUD'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC-AUC score (Receiver Operating Characteristic - Area Under Curve)
roc_auc = roc_auc_score(y_val, rf_selected.predict_proba(X_val_scaled)[:, 1])  # Predict probabilities for class 1
print("ROC-AUC Score:", roc_auc)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, rf_selected.predict_proba(X_val_scaled)[:, 1])
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='b', label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[27]:


# Create the XGBoost model (Classifier for classification task)
model_xgb = xgb.XGBClassifier(
    objective='binary:logistic',   # 'binary:logistic' for binary classification; use 'multi:softmax' for multi-class
    eval_metric='logloss',         # Evaluation metric for binary classification
    use_label_encoder=False,       # Avoid deprecated label encoding warnings
    n_estimators=100,              # Number of boosting rounds (trees)
    learning_rate=0.1,             # Learning rate (step size)
    max_depth=6,                   # Maximum depth of each tree
    colsample_bytree=0.8,          # Fraction of features to use for each tree
    subsample=0.8                  # Fraction of samples to use for each tree
)

# Fit the model on the training data
model_xgb.fit(X_train_scaled, y_train)

# Make predictions on the validation set
y_pred_xgb = model_xgb.predict(X_val_scaled)
y_pred_proba_xgb = model_xgb.predict_proba(X_val_scaled)[:, 1]  # For ROC AUC, get the probabilities for the positive class

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred_xgb)
print(f"Validation Accuracy: {accuracy:.4f}")

# Print confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_xgb))

print("Classification Report:")
print(classification_report(y_val, y_pred_xgb))

# ROC AUC Score
roc_auc = roc_auc_score(y_val, y_pred_proba_xgb)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba_xgb)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[28]:


# Define the model
model_DL = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),

    Dense(8, activation='relu'),
    
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model_DL.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC', 'Precision', 'Recall', 'accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model_DL.fit(X_train_scaled, y_train, 
                    validation_data=(X_val_scaled, y_val), 
                    epochs=100, 
                    batch_size=64, 
                    callbacks=[early_stopping],
                    verbose=1)

# Evaluate the model
results = model_DL.evaluate(X_val_scaled, y_val, verbose=0)

# Optionally, print the evaluation results
print(f"Validation Loss: {results[0]}")
print(f"Validation AUC: {results[1]}")
print(f"Validation Precision: {results[2]}")
print(f"Validation Recall: {results[3]}")
print(f"Validation Accuracy: {results[4]}")


# Make predictions on the validation or test data
y_pred_dl = model_DL.predict(X_val_scaled)  # For validation data
y_pred_dl_class = (y_pred_dl > 0.5).astype(int)  # Convert probabilities to class labels

# Calculate the ROC AUC score
roc_auc_dl = roc_auc_score(y_val, y_pred_dl)
print(f"ROC AUC Score for Deep Learning Model: {roc_auc_dl:.4f}")

# Optionally, plot confusion matrix or classification report
print("\nConfusion Matrix:")
cm_dl_val = confusion_matrix(y_val, y_pred_dl_class)
sns.heatmap(cm_dl_val, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Data)')
plt.show()

print("Classification Report:")
print(classification_report(y_val, y_pred_dl_class))


# Check each model accuracy on Test Data which was not exposed till now.

# Logistic Regression 

# In[31]:


y_test_pred_LR = log_reg.predict(X_test_scaled)
y_test_pred_proba_LR = log_reg.predict_proba(X_test_scaled)[:, 1]

test_accuracy_LR = accuracy_score(y_test, y_test_pred_LR)
print(f"Test Accuracy: {test_accuracy_LR:.4f}")

# Test ROC AUC Score
test_roc_auc_LR = roc_auc_score(y_test, y_test_pred_proba_LR)
print(f"Test ROC AUC Score: {test_roc_auc_LR:.4f}")

# Optionally, plot confusion matrix or classification report
print("Confusion Matrix:")
cm_dl_test = confusion_matrix(y_test, y_test_pred_LR)
sns.heatmap(cm_dl_test, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Data)')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_test_pred_LR))


# Random Forest 

# In[33]:


y_test_pred_RF = rf_selected.predict(X_test_scaled)
y_test_pred_proba_RF = rf_selected.predict_proba(X_test_scaled)[:, 1]

test_accuracy_RF = accuracy_score(y_test, y_test_pred_RF)
print(f"Test Accuracy: {test_accuracy_RF:.4f}")

# Test ROC AUC Score
test_roc_auc_RF = roc_auc_score(y_test, y_test_pred_proba_RF)
print(f"Test ROC AUC Score: {test_roc_auc_RF:.4f}")

# Optionally, plot confusion matrix or classification report
print("Confusion Matrix:")
cm_dl_test = confusion_matrix(y_test, y_test_pred_RF)
sns.heatmap(cm_dl_test, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Data)')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_test_pred_RF))


# XGBoost

# In[35]:


# Optionally, you can evaluate on the test set as well
y_test_pred_xgb = model_xgb.predict(X_test_scaled)
y_test_pred_proba_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1]  # For ROC AUC on test set

test_accuracy = accuracy_score(y_test, y_test_pred_xgb)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Test ROC AUC Score
test_roc_auc = roc_auc_score(y_test, y_test_pred_proba_xgb)
print(f"Test ROC AUC Score: {test_roc_auc:.4f}")

# Optionally, plot confusion matrix or classification report
print("Confusion Matrix:")
cm_dl_test = confusion_matrix(y_test, y_test_pred_xgb)
sns.heatmap(cm_dl_test, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Data)')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_test_pred_xgb))


# DeepLearning

# In[37]:


# 1. Make predictions using the trained deep learning model
y_pred_dl_test = model_DL.predict(X_test_scaled)  # Predict the probabilities for the test data
# 2. Convert probabilities to class labels (for binary classification)
y_pred_dl_test = (y_pred_dl_test > 0.5).astype(int)  # Convert probabilities to class labels (0 or 1)

# 3. Evaluate the model on the test data
accuracy_test_dl = accuracy_score(y_test, y_pred_dl_test)
# Print evaluation metrics
print(f"Test Accuracy (Deep Learning): {accuracy_test_dl:.4f}")

# Calculate the ROC AUC score
roc_auc_dl = roc_auc_score(y_test, y_pred_dl_test)
print(f"ROC AUC Score for Deep Learning Model: {roc_auc_dl:.4f}")

# Optionally, plot confusion matrix or classification report
print("\nConfusion Matrix:")
cm_dl_test = confusion_matrix(y_test, y_pred_dl_test)
sns.heatmap(cm_dl_test, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Data)')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred_dl_test))


# Based on above model accuracy, classification report & confusion Matrix, we select 3 models for API building i.e. XGboost, RandomForest, DeepLearning.

# In[39]:


# Save the models for API building.
joblib.dump(rf_selected, 'random_forest_model.joblib')
model_xgb.save_model("xgb_model.json")
model_DL.save("deep_learning_model.keras")

