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


# =============================================================================
# 4. HEART DISEASE PREDICTION
# =============================================================================
print("### 4. HEART DISEASE PREDICTION ###")

# Load Dataset
heart_df = pd.read_csv(r'Dataset/Heart_Disease_Prediction.csv')

# Data Preprocessing
# Encode Target 'Presence'/'Absence'
le_heart = LabelEncoder()
heart_df['Heart Disease'] = le_heart.fit_transform(heart_df['Heart Disease'])

X_heart = heart_df.drop(columns=['Heart Disease'], axis=1)
y_heart = heart_df['Heart Disease']

# Split into Training and Testing sets
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

# Feature Scaling
scaler_heart = StandardScaler()
X_train_heart = scaler_heart.fit_transform(X_train_heart)
X_test_heart = scaler_heart.transform(X_test_heart)

# Train Random Forest Classifier
model_heart = RandomForestClassifier(n_estimators=100, random_state=42)
model_heart.fit(X_train_heart, y_train_heart)

# Evaluation
evaluate_model(model_heart, X_test_heart, y_test_heart, "Heart Disease")

# Save Models
with open('models/heart_model.pkl', 'wb') as f:
    pickle.dump(model_heart, f)
with open('models/heart_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_heart, f)


# =============================================================================
# 5. THYROID DISEASE PREDICTION
# =============================================================================
print("### 5. THYROID DISEASE PREDICTION ###")

# Load Dataset
thyroid_df = pd.read_csv(r'Dataset/Thyroid_Dataset.csv')

# Data Preprocessing
# thyroid_df needs cleaning as it contains '?' or missing values and categorical data
thyroid_df.replace('?', np.nan, inplace=True)

# Filling missing values
for column in thyroid_df.columns:
    if thyroid_df[column].dtype == 'object':
        thyroid_df[column] = thyroid_df[column].fillna(thyroid_df[column].mode()[0])
    else:
        thyroid_df[column] = thyroid_df[column].fillna(thyroid_df[column].mean())

# Encoding Categorical Features
le_thyroid = LabelEncoder()
for col in thyroid_df.select_dtypes(include=['object']).columns:
    thyroid_df[col] = le_thyroid.fit_transform(thyroid_df[col].astype(str))

X_thyroid = thyroid_df.drop(columns=['Class'], axis=1)
y_thyroid = thyroid_df['Class']

# Split into Training and Testing sets
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_thyroid, y_thyroid, test_size=0.2, random_state=42)

# Feature Scaling
scaler_t = StandardScaler()
X_train_t = scaler_t.fit_transform(X_train_t)
X_test_t = scaler_t.transform(X_test_t)

# Train Random Forest Classifier
model_t = RandomForestClassifier(n_estimators=100, random_state=42)
model_t.fit(X_train_t, y_train_t)

# Evaluation
evaluate_model(model_t, X_test_t, y_test_t, "Thyroid Disease")

# Save Models
with open('models/thyroid_model.pkl', 'wb') as f:
    pickle.dump(model_t, f)
with open('models/thyroid_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_t, f)


# =============================================================================
# 6. LIVER DISEASE PREDICTION
# =============================================================================
print("### 6. LIVER DISEASE PREDICTION ###")

# Load Dataset
liver_df = pd.read_csv(r'Dataset/indian_liver_patient.csv')

# Data Preprocessing
# Handling missing values
liver_df['Albumin_and_Globulin_Ratio'] = liver_df['Albumin_and_Globulin_Ratio'].fillna(liver_df['Albumin_and_Globulin_Ratio'].mean())

# Encoding Gender
le_liver = LabelEncoder()
liver_df['Gender'] = le_liver.fit_transform(liver_df['Gender'])

# Target is 'Dataset' (1: disease, 2: no disease)
# Standardize it to 0 and 1 (0: No disease, 1: Disease)
liver_df['Dataset'] = liver_df['Dataset'].map({1: 1, 2: 0})

X_liver = liver_df.drop(columns=['Dataset'], axis=1)
y_liver = liver_df['Dataset']

# Split into Training and Testing sets
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_liver, y_liver, test_size=0.2, random_state=42)

# Feature Scaling
scaler_l = StandardScaler()
X_train_l = scaler_l.fit_transform(X_train_l)
X_test_l = scaler_l.transform(X_test_l)

# Train Random Forest Classifier
model_l = RandomForestClassifier(n_estimators=100, random_state=42)
model_l.fit(X_train_l, y_train_l)

# Evaluation
evaluate_model(model_l, X_test_l, y_test_l, "Liver Disease")

# Save Models
with open('models/liver_model.pkl', 'wb') as f:
    pickle.dump(model_l, f)
with open('models/liver_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_l, f)

# =============================================================================
# 7. LIFESTYLE DISEASE MODELS (RUL-BASED/THRESHOLD PLACEHOLDERS)
# =============================================================================
# For these, we will save simple DecisionTree placeholders so they can be 
# loaded in app.py similarly to other models, or we can use custom logic in app.py.
# The user said "set the values", so we'll implement rule-based logic in app.py 
# for maximum flexibility, but we'll print a confirmation here.

print("\n### LIFESTYLE DISEASE MODELS (Diabetes, Obesity, Hypertension, Anemia, Fatty Liver, High Cholesterol) ###")
print("These will be handled via clinical threshold logic in the application portal.")

print("\nSummary: All models (Parkinson's, Hepatitis, Kidney, Heart, Thyroid, Liver) have been trained and saved successfully.")
