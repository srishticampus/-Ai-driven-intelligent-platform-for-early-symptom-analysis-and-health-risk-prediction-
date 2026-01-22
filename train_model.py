import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle
import os

# Create folders if they don't exist
if not os.path.exists('models'):
    os.makedirs('models')

# 1. Load the dataset
data_path = r'Dataset/diabetes.csv'
df = pd.read_csv(data_path)

print("Original Data Shape:", df.shape)

# 2. Preprocessing - Data Cleaning
# Handle logical zeros in specific columns
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_fix:
    df[col] = df[col].replace(0, df[col].median())

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Hyperparameter Tuning for Random Forest
print("\n--- Tuning Random Forest ---")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
print(f"Best RF Params: {rf_grid.best_params_}")
print(f"RF CV Accuracy: {rf_grid.best_score_:.4f}")

# 6. Trying Gradient Boosting for comparison
print("\n--- Tuning Gradient Boosting ---")
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=5, scoring='accuracy', n_jobs=-1)
gb_grid.fit(X_train, y_train)
gb_best = gb_grid.best_estimator_
print(f"Best GB Params: {gb_grid.best_params_}")
print(f"GB CV Accuracy: {gb_grid.best_score_:.4f}")

# 7. Select and Evaluate the Best Model
if gb_grid.best_score_ > rf_grid.best_score_:
    best_model = gb_best
    model_name = "Gradient Boosting"
else:
    best_model = rf_best
    model_name = "Random Forest"

print(f"\n🚀 Selected Best Model: {model_name}")

y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Final Model Evaluation ({model_name}) ---")
print(f"Test Accuracy: {final_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save the best model and the scaler
with open('models/diabetes_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"\nOptimized {model_name} and Scaler saved successfully to 'models/'.")
