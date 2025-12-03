"""
Configuration file for Student Dropout Prediction Project
Multi-Dataset Support for Enhanced Accuracy
"""
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
NOTEBOOK_DIR = BASE_DIR / "notebooks"

# Multiple Dataset Paths
DATASET_1_PATH = RAW_DATA_DIR / "dataset.csv"  # Higher Education Predictors
DATASET_2_PATH = RAW_DATA_DIR / "student-mat.csv"  # Student Performance (Math)
DATASET_3_PATH = RAW_DATA_DIR / "Dropout_Academic Success - Sheet1.csv"  # Academic Success
DATASET_4_PATH = RAW_DATA_DIR / "student_mental_health.csv"  # Mental Health
DATASET_5_PATH = RAW_DATA_DIR / "predict+students+dropout+and+academic+success.csv"  # Predict Dropout

# Dataset names for reference
DATASET_NAMES = {
    'dataset1': 'Higher Education Predictors',
    'dataset2': 'Student Performance',
    'dataset3': 'Academic Success',
    'dataset4': 'Mental Health',
    'dataset5': 'Predict Dropout & Success'
}

# All datasets list
ALL_DATASETS = [
    DATASET_1_PATH,
    DATASET_2_PATH,
    DATASET_3_PATH,
    DATASET_4_PATH,
    DATASET_5_PATH
]

# Merged and processed data paths
MERGED_DATASET_PATH = PROCESSED_DATA_DIR / "merged_datasets.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_data.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.csv"

# Model paths
LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_regression.pkl"
RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "random_forest.pkl"
XGBOOST_MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
DECISION_TREE_MODEL_PATH = MODEL_DIR / "decision_tree.pkl"
SVM_MODEL_PATH = MODEL_DIR / "svm.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# XGBoost parameters
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'objective': 'multi:softmax',  # Changed to multi-class
    'num_class': 3,  # Dropout, Graduate, Enrolled
    'random_state': RANDOM_STATE,
    'eval_metric': 'mlogloss'
}

# Random Forest parameters
RF_PARAMS = {
    'n_estimators':200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Decision Tree parameters
DT_PARAMS = {
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE
}

# Logistic Regression parameters
LR_PARAMS = {
    'max_iter': 2000,
    'random_state': RANDOM_STATE,
    'solver': 'lbfgs',
    'multi_class': 'multinomial'
}

# SVM parameters
SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'random_state': RANDOM_STATE
}

# Feature engineering
CATEGORICAL_FEATURES = []  # To be updated after data exploration
NUMERICAL_FEATURES = []    # To be updated after data exploration

# Target variable mapping
TARGET_MAPPING = {
    'Dropout': 0,
    'Graduate': 1,
    'Enrolled': 2
}

# Streamlit app configuration
APP_TITLE = "Student Dropout Prediction System"
APP_ICON = "🎓"
PAGE_CONFIG = {
    'page_title': APP_TITLE,
    'page_icon': APP_ICON,
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Model performance thresholds
MIN_ACCURACY = 0.83  # Based on reference project
MIN_PRECISION = 0.80
MIN_RECALL = 0.78
MIN_F1_SCORE = 0.80
MIN_ROC_AUC = 0.85

# Data merging strategy
MERGE_STRATEGY = 'union'  # 'union' or 'intersection' of features
HANDLE_MISSING = 'impute'  # 'impute' or 'drop'
FEATURE_SELECTION = True
MIN_FEATURE_IMPORTANCE = 0.01
