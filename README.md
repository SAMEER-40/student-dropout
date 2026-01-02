# 🎓 Student Dropout Prediction System

AI-powered machine learning system to predict student dropout risk in higher education.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![React](https://img.shields.io/badge/React-Frontend-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-77%25-orange)

## 🚀 Quick Start (After Cloning)

### Prerequisites
- **Python 3.9+**
- **Node.js 18+** (for frontend)

### Step 1: Install Python Dependencies
```bash
cd student-dropout
pip install -r requirements.txt
```

### Step 2: Start the Backend (Terminal 1)
```bash
python -m uvicorn api.main:app --reload --port 8000
```
You should see:
```
✓ Model loaded: 15 features, 3 classes
INFO: Uvicorn running on http://127.0.0.1:8000
```

### Step 3: Start the Frontend (Terminal 2)
```bash
cd frontend
npm run dev
```
> **Note**: `node_modules` is included, so no `npm install` needed!

### Step 4: Open the App
Navigate to **http://localhost:5173** in your browser.

---

## 📊 Features

| Feature | Description |
|---------|-------------|
| **Multi-Class Prediction** | Classifies students as `Dropout`, `Enrolled`, or `Graduate` |
| **77% Accuracy** | Tuned Random Forest on 9,000+ student records |
| **Explainable AI** | SHAP explanations show why predictions are made |
| **Modern UI** | React-based responsive dashboard |
| **REST API** | FastAPI backend with validation |

---

## 📂 Project Structure

```
student-dropout/
├── api/                    # FastAPI Backend
│   ├── main.py            # API endpoints
│   ├── schemas.py         # Pydantic models
│   └── services/          # Prediction service
├── frontend/              # React Frontend
│   └── src/
│       ├── App.jsx        # Main component
│       └── components/    # UI components
├── src/                   # Core ML utilities
│   ├── utils.py           # Model loading, validation
│   └── explainability.py  # SHAP/LIME
├── data/
│   ├── raw/               # 5 original datasets
│   └── processed/         # Merged & split data
├── models/                # Trained models
│   ├── best_model.pkl     # Random Forest (77%)
│   ├── preprocessor.pkl   # Feature transformer
│   └── target_encoder.pkl # Label encoder
├── notebooks/             # Jupyter notebooks
│   ├── 00_merge_datasets.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_optimization.ipynb
│   └── 05_model_interpretability.ipynb
└── tests/                 # Test suite
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check API status |
| `/schema` | GET | Get feature schema |
| `/predict` | POST | Make prediction |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Age": 20,
      "Gender": 1,
      "Scholarship_Holder": 1,
      "Tuition_Fees_Up_To_Date": 1,
      "Debtor": 0
    },
    "explain": true
  }'
```

---

## 🧪 Running Tests
```bash
python -m pytest tests/ -v
```

---

## 📓 Reproducing the Model

To retrain the model from scratch, run the notebooks in order:
1. `notebooks/00_merge_datasets_v2.ipynb` - Merge raw datasets
2. `notebooks/02_data_preprocessing.ipynb` - Preprocess & split
3. `notebooks/03_model_training.ipynb` - Train models
4. `notebooks/04_model_optimization.ipynb` - Hyperparameter tuning

---

## 📈 Model Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **Random Forest (Tuned)** | **77%** | 0.77 |
| XGBoost | 76% | 0.76 |
| Logistic Regression | 72% | 0.72 |

### Top Predictors (SHAP)
1. Curricular Units 2nd Sem Grade
2. Tuition Fees Up to Date
3. Age at Enrollment
4. Scholarship Holder

---

## 👤 Author
**Santosh**

---

*Built with Python, FastAPI, React, Scikit-Learn, and SHAP*
