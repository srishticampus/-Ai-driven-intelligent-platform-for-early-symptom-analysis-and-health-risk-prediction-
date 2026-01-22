import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

# Create folders
if not os.path.exists('models'):
    os.makedirs('models')

def train_and_save(df, target_col, model_name, drop_cols=None):
    print(f"\n--- Training {model_name} ---")
    
    # 1. Cleaning
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col:
            df[col] = le.fit_transform(df[col])
    
    # If target is object, encode it
    if df[target_col].dtype == 'object':
        df[target_col] = le.fit_transform(df[target_col])
        target_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Target Mapping: {target_mapping}")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # 2. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Model Training (Random Forest)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Accuracy: {model.score(X_test, y_test):.4f}")
    
    # 4. Save
    with open(f'models/{model_name.lower()}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'models/{model_name.lower()}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return list(X.columns)

# --- DIABETES ---
df_dia = pd.read_csv('Dataset/diabetes.csv')
# Handle logically missing zeros for diabetes specifically
cols_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for c in cols_fix: df_dia[c] = df_dia[c].replace(0, df_dia[c].median())
dia_cols = train_and_save(df_dia, 'Outcome', 'Diabetes')

# --- HEART ---
df_heart = pd.read_csv('Dataset/Heart_Disease_Prediction.csv')
heart_cols = train_and_save(df_heart, 'Heart Disease', 'Heart')

# --- LIVER ---
df_liver = pd.read_csv('Dataset/indian_liver_patient.csv')
# Gender needs encoding before train_and_save handles it generally
liver_cols = train_and_save(df_liver, 'Dataset', 'Liver')

# --- KIDNEY ---
df_kidney = pd.read_csv('Dataset/kidney_disease_dataset.csv')
kidney_cols = train_and_save(df_kidney, 'Target', 'Kidney')

# Save Column metadata for the app to know what inputs to show
metadata = {
    "Diabetes": dia_cols,
    "Heart": heart_cols,
    "Liver": liver_cols,
    "Kidney": kidney_cols
}
with open('models/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\n🚀 All models trained and meta-data saved!")
