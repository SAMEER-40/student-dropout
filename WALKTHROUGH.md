# 📖 Project Walkthrough - Student Dropout Prediction System

This document guides you through the complete project flow, from raw data to predictions.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Raw CSVs (5)  →  Merge  →  Preprocess  →  SMOTE  →  Train/Test     │
│  data/raw/         ↓          ↓            ↓          ↓             │
│                notebooks/  LabelEncode   Balance   data/processed/  │
│                            StandardScale  Classes                    │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Train Data  →  Random Forest  →  XGBoost  →  Logistic Regression   │
│                      ↓                                               │
│              Hyperparameter Tuning (RandomizedSearchCV)              │
│                      ↓                                               │
│              Best Model: Random Forest (77% accuracy)                │
│                      ↓                                               │
│              Save: best_model.pkl, preprocessor.pkl, encoder.pkl    │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      WEB APPLICATION                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   React Frontend (localhost:5173)                                    │
│        ↓                                                             │
│   User fills form with student data                                  │
│        ↓                                                             │
│   POST /api/predict                                                  │
│        ↓                                                             │
│   FastAPI Backend (localhost:8000)                                   │
│        ↓                                                             │
│   PredictionService.predict()                                        │
│        ↓                                                             │
│   preprocessor.transform() → model.predict() → SHAP explain         │
│        ↓                                                             │
│   Return: { prediction, probabilities, shap_explanation }           │
│        ↓                                                             │
│   Frontend displays result with charts                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Folder-by-Folder Explanation

### 1. `data/raw/` - Original Datasets
Contains 5 CSV files from different sources:
- `dataset.csv` - Primary higher education data
- `student-mat.csv` - Portuguese student math performance
- `Dropout_Academic Success - Sheet1.csv` - Academic success indicators
- `student_mental_health.csv` - Mental health factors
- `predict+students+dropout+and+academic+success.csv` - Composite dataset

### 2. `data/processed/` - Cleaned Data
After running the preprocessing notebooks:
- `merged_datasets.csv` - All 5 datasets combined (~9,000 rows)
- `train_data.csv` - 80% for training (with SMOTE balancing)
- `test_data.csv` - 20% for testing

### 3. `notebooks/` - Development Journey
Run these in order to understand the ML pipeline:

| Notebook | Purpose | Key Output |
|----------|---------|------------|
| `00_merge_datasets_v2.ipynb` | Combine all 5 datasets | `merged_datasets.csv` |
| `01_data_exploration.ipynb` | Visualize distributions, correlations | Understanding of data |
| `02_data_preprocessing.ipynb` | Handle missing values, encode, scale, SMOTE | `preprocessor.pkl`, `train_data.csv` |
| `03_model_training.ipynb` | Train RF, XGBoost, LR, DT, SVM | Model comparison |
| `04_model_optimization.ipynb` | Hyperparameter tuning | `best_model.pkl` |
| `05_model_interpretability.ipynb` | SHAP analysis | Feature importance charts |

### 4. `models/` - Trained Artifacts
- `best_model.pkl` - Trained Random Forest classifier (35 MB)
- `best_model_xgboost.pkl` - Alternative XGBoost model (2 MB)
- `preprocessor.pkl` - ColumnTransformer with scaler
- `target_encoder.pkl` - LabelEncoder for Dropout/Enrolled/Graduate
- `manifest.json` - Version tracking

### 5. `src/` - Core Python Code
- `utils.py` - Model loading, validation, prediction functions
- `explainability.py` - SHAP and LIME wrapper classes

### 6. `api/` - FastAPI Backend
- `main.py` - API endpoints (/health, /schema, /predict)
- `schemas.py` - Pydantic request/response validation
- `services/prediction.py` - Pure prediction logic (no web dependencies)

### 7. `frontend/` - React UI
- `src/App.jsx` - Main app with schema loading
- `src/api.js` - API client functions
- `src/components/` - StudentForm, ResultCard, ShapChart

---

## 🔄 Data Flow Example

**User Action**: Fill form with Age=20, Scholarship=Yes, Fees Paid=Yes

```
1. Frontend (React)
   └── StudentForm collects input
   └── Sends POST /api/predict { features: {...}, explain: true }

2. Backend (FastAPI)
   └── api/main.py receives request
   └── Validates with Pydantic schemas
   └── Calls PredictionService.predict()

3. Prediction Service
   └── Converts features dict → DataFrame
   └── validate_input() checks all 15 features present
   └── preprocessor.transform() → scales/encodes
   └── model.predict() → returns class index (0, 1, or 2)
   └── model.predict_proba() → returns [0.15, 0.25, 0.60]
   └── SHAP TreeExplainer → computes feature importances

4. Response
   └── {
         prediction: 2,
         prediction_label: "Graduate",
         probabilities: { Dropout: 0.15, Enrolled: 0.25, Graduate: 0.60 },
         confidence: 0.60,
         explanation: [{ feature: "Scholarship_Holder", shap_value: 0.12, ... }]
       }

5. Frontend Display
   └── ResultCard shows "🟢 Graduate" with 60% confidence
   └── ShapChart shows why (scholarship positive, fees paid positive)
```

---

## 🎯 Key Design Decisions

1. **Multi-Source Data**: 5 datasets merged for robust generalization
2. **SMOTE Balancing**: Synthetic oversampling for minority classes
3. **Random Forest**: Chosen over XGBoost for better interpretability
4. **FastAPI + React**: Modern separation of concerns
5. **Schema-Driven Forms**: Frontend fetches schema from backend, never hardcodes
6. **Optional SHAP**: Explanation is opt-in to keep predictions fast

---

## 🧪 Testing the Pipeline

```bash
# Run all tests
python -m pytest tests/ -v

# Expected output:
# test_golden.py::TestGoldenInvariants::test_model_loads_successfully PASSED
# test_golden.py::TestGoldenInvariants::test_feature_count_consistency PASSED
# test_golden.py::TestGoldenInvariants::test_target_classes_complete PASSED
# ... 8 tests passed
```

---

## 🚀 Demonstrating to Others

1. **Show the raw data**: Open `data/raw/` CSVs in Excel
2. **Run a notebook**: Open `03_model_training.ipynb` to show training
3. **Start the app**: Backend + Frontend as in README
4. **Make a prediction**: Fill the form, explain the SHAP chart
5. **Show the code**: Walk through `api/services/prediction.py`

---

*This project demonstrates end-to-end ML: data collection, preprocessing, model training, explainability, and production deployment.*
