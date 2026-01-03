"""
Quick Model Retraining Script
Trains a Random Forest model with the new preprocessed data
"""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import sys
sys.path.append('.')
import config

# Load data
print("Loading data...")
train_df = pd.read_csv(config.TRAIN_DATA_PATH)
test_df = pd.read_csv(config.TEST_DATA_PATH)

X_train = train_df.drop(columns=['Target'])
y_train = train_df['Target']
X_test = test_df.drop(columns=['Target'])
y_test = test_df['Target']

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train Random Forest
print("\nTraining Random Forest...")
model = RandomForestClassifier(**config.RF_PARAMS)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✓ Model trained successfully")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save
model_path = config.MODEL_DIR / "best_model.pkl"
joblib.dump(model, model_path)
print(f"\n✓ Model saved to: {model_path}")
