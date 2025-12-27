# 🎓 Student Dropout Prediction System v2.0

## 📋 Project Overview
AI-powered machine learning system to predict student dropout risk in higher education. Built with **FastAPI** backend and **React** frontend.

### Key Features
- **Multi-Class Prediction**: Classifies students as `Dropout`, `Enrolled`, or `Graduate`
- **Real-time Analysis**: Instant predictions with confidence scores
- **Explainable AI**: SHAP explanations showing why predictions are made
- **Modern UI**: React-based dashboard with responsive design

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
cd frontend && npm install
```

### 2. Start the Backend (FastAPI)
```bash
uvicorn api.main:app --reload --port 8000
```

### 3. Start the Frontend (React)
```bash
cd frontend
npm run dev
```

### 4. Open the App
Navigate to **http://localhost:5173** in your browser.

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check API status |
| `/schema` | GET | Get feature schema for form rendering |
| `/predict` | POST | Make a dropout prediction |

### Example Prediction Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {"Age": 20, "Scholarship_Holder": 1, "Tuition_Fees_Up_To_Date": 1},
    "explain": true,
    "top_k_features": 5
  }'
```

## 📂 Project Structure
```
Santosh_minor/
├── api/                    # FastAPI Backend
│   ├── main.py            # API endpoints
│   ├── schemas.py         # Pydantic models
│   └── services/          # Business logic
├── frontend/              # React Frontend
│   └── src/
│       ├── App.jsx        # Main component
│       └── components/    # UI components
├── src/                   # Core ML utilities
├── models/                # Trained models (.pkl)
├── notebooks/             # Jupyter notebooks
├── tests/                 # Test suite
└── app.py                 # [DEPRECATED] Streamlit app
```

## 🧪 Testing
```bash
pytest tests/ -v
```

## 📈 Model Performance
- **Accuracy**: 77%
- **Algorithm**: Random Forest (tuned)
- **Features**: 15 predictors

---
*Built with FastAPI, React, Scikit-Learn, and SHAP*
