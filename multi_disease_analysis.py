import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import pickle
import os

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Create folders if they don't exist
if not os.path.exists('models'):
    os.makedirs('models')

def evaluate_model(model, X_test, y_test, disease_name):
    """Utility function to evaluate and print metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"--- Model Evaluation for {disease_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 40 + "\n")

# =============================================================================
# 1. PARKINSON'S DISEASE PREDICTION
# =============================================================================
print("### 1. PARKINSON'S DISEASE PREDICTION ###")

# Load Dataset
parkinsons_df = pd.read_csv(r'Dataset/parkinsons.csv')

# Data Preprocessing
# Drop the 'name' column as it is irrelevant for prediction
X_parkinsons = parkinsons_df.drop(columns=['name', 'status'], axis=1)
y_parkinsons = parkinsons_df['status']

# Split into Training and Testing sets
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_parkinsons, y_parkinsons, test_size=0.2, random_state=42)

# Feature Scaling
scaler_p = StandardScaler()
X_train_p = scaler_p.fit_transform(X_train_p)
X_test_p = scaler_p.transform(X_test_p)

# Train Random Forest Classifier
model_p = RandomForestClassifier(n_estimators=100, random_state=42)
model_p.fit(X_train_p, y_train_p)

# Evaluation
evaluate_model(model_p, X_test_p, y_test_p, "Parkinson's Disease")

# Save Models
with open('models/parkinsons_model.pkl', 'wb') as f:
    pickle.dump(model_p, f)
with open('models/parkinsons_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_p, f)


# =============================================================================
# 2. HEPATITIS PREDICTION
# =============================================================================
print("### 2. HEPATITIS PREDICTION ###")

# Load Dataset
# Note: Preview showed missing values and mixed types
hepatitis_df = pd.read_csv(r'Dataset/hepatitis_csv.csv')

# Data Preprocessing
# Handling missing values (replacing empty strings or nulls)
# Mode for categorical/boolean, mean for numerical
for column in hepatitis_df.columns:
    if hepatitis_df[column].dtype == 'object':
        hepatitis_df[column] = hepatitis_df[column].fillna(hepatitis_df[column].mode()[0])
    else:
        hepatitis_df[column] = hepatitis_df[column].fillna(hepatitis_df[column].mean())

# Encoding Categorical Features
le = LabelEncoder()
categorical_cols = ['sex', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia', 
                    'liver_big', 'liver_firm', 'spleen_palpable', 'spiders', 'ascites', 
                    'varices', 'histology', 'class']

for col in categorical_cols:
    hepatitis_df[col] = le.fit_transform(hepatitis_df[col].astype(str))

X_hepatitis = hepatitis_df.drop(columns=['class'], axis=1)
y_hepatitis = hepatitis_df['class']

# Split into Training and Testing sets
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_hepatitis, y_hepatitis, test_size=0.2, random_state=42)

# Feature Scaling
scaler_h = StandardScaler()
X_train_h = scaler_h.fit_transform(X_train_h)
X_test_h = scaler_h.transform(X_test_h)

# Train Random Forest Classifier
model_h = RandomForestClassifier(n_estimators=100, random_state=42)
model_h.fit(X_train_h, y_train_h)

# Evaluation
evaluate_model(model_h, X_test_h, y_test_h, "Hepatitis")

# Save Models
with open('models/hepatitis_model.pkl', 'wb') as f:
    pickle.dump(model_h, f)
with open('models/hepatitis_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_h, f)


# =============================================================================
# 3. KIDNEY DISEASE PREDICTION
# =============================================================================
print("### 3. KIDNEY DISEASE PREDICTION ###")

# Load Dataset
kidney_df = pd.read_csv(r'Dataset/kidney_disease_dataset.csv')

# Data Preprocessing
# Handling missing values
for column in kidney_df.columns:
    if kidney_df[column].dtype == 'object':
        kidney_df[column] = kidney_df[column].fillna(kidney_df[column].mode()[0])
    else:
        kidney_df[column] = kidney_df[column].fillna(kidney_df[column].mean())

# Encoding Categorical Features
# Target is 'Target'
target_le = LabelEncoder()
kidney_df['Target'] = target_le.fit_transform(kidney_df['Target'])

# Encode other categorical columns
obj_cols = kidney_df.select_dtypes(include=['object']).columns
for col in obj_cols:
    kidney_df[col] = le.fit_transform(kidney_df[col].astype(str))

X_kidney = kidney_df.drop(columns=['Target'], axis=1)
y_kidney = kidney_df['Target']

# Split into Training and Testing sets
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X_kidney, y_kidney, test_size=0.2, random_state=42)

# Feature Scaling
scaler_k = StandardScaler()
X_train_k = scaler_k.fit_transform(X_train_k)
X_test_k = scaler_k.transform(X_test_k)

# Train Random Forest Classifier
model_k = RandomForestClassifier(n_estimators=100, random_state=42)
model_k.fit(X_train_k, y_train_k)

# Evaluation
evaluate_model(model_k, X_test_k, y_test_k, "Kidney Disease")

# Save Models
with open('models/kidney_model.pkl', 'wb') as f:
    pickle.dump(model_k, f)
with open('models/kidney_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_k, f)


print("All models and scalers (Parkinson's, Hepatitis, Kidney) saved successfully to the 'models' directory.")
