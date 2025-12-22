"""
Student Dropout Prediction - Utility Functions
Handles model loading, schema validation, and predictions with strict guarantees.
"""
import pandas as pd
import numpy as np
import joblib
import json
import logging
import hashlib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path configuration
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_PATH = MODEL_DIR / "app_metadata.json"
MANIFEST_PATH = MODEL_DIR / "manifest.json"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.csv"


class ModelArtifacts:
    """Container for loaded model artifacts with validation."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.target_encoder = None
        self.feature_names: List[str] = []
        self.n_features: int = 0
        self.target_classes: List[str] = []
        self.is_loaded: bool = False
        self.manifest: Dict[str, Any] = {}
    
    def validate_consistency(self) -> bool:
        """Validate that all artifacts are consistent with each other."""
        errors = []
        
        # Check preprocessor output features
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            prep_features = len(self.preprocessor.get_feature_names_out())
        elif hasattr(self.preprocessor, 'transformers_'):
            # Count features from transformers
            prep_features = sum(
                len(cols) for _, _, cols in self.preprocessor.transformers_
                if cols != 'drop'
            )
        else:
            prep_features = self.n_features
        
        # Check model expected features
        if hasattr(self.model, 'n_features_in_'):
            model_features = self.model.n_features_in_
        elif hasattr(self.model, 'feature_importances_'):
            model_features = len(self.model.feature_importances_)
        else:
            model_features = self.n_features
        
        # Validate
        if prep_features != model_features:
            errors.append(
                f"Feature count mismatch: preprocessor outputs {prep_features}, "
                f"model expects {model_features}"
            )
        
        if self.n_features != model_features:
            errors.append(
                f"Schema mismatch: metadata has {self.n_features} features, "
                f"model expects {model_features}"
            )
        
        if len(self.target_encoder.classes_) != len(self.target_classes):
            errors.append(
                f"Target class mismatch: encoder has {len(self.target_encoder.classes_)}, "
                f"metadata has {len(self.target_classes)}"
            )
        
        if errors:
            for error in errors:
                logger.error(f"Validation error: {error}")
            raise ValueError(
                "Model artifacts are inconsistent. This usually means artifacts "
                "were generated from different training runs. Please retrain.\n"
                + "\n".join(errors)
            )
        
        logger.info(f"✓ Validation passed: {model_features} features, "
                    f"{len(self.target_classes)} classes")
        return True


# Global artifacts container (loaded once at startup)
_artifacts: Optional[ModelArtifacts] = None


def _compute_file_hash(path: Path) -> str:
    """Compute MD5 hash of a file."""
    if not path.exists():
        return ""
    return hashlib.md5(open(path, 'rb').read()).hexdigest()[:12]


def load_model_and_preprocessor() -> Tuple[Any, Any, Any]:
    """
    Load the trained model, preprocessor, and target encoder.
    Performs strict validation to ensure all artifacts are consistent.
    
    Returns:
        Tuple of (model, preprocessor, target_encoder)
    
    Raises:
        FileNotFoundError: If required model files are missing
        ValueError: If artifacts are inconsistent
    """
    global _artifacts
    
    # Return cached if already loaded
    if _artifacts is not None and _artifacts.is_loaded:
        return _artifacts.model, _artifacts.preprocessor, _artifacts.target_encoder
    
    _artifacts = ModelArtifacts()
    
    # Define model paths in priority order
    model_candidates = [
        MODEL_DIR / "best_model_tuned.pkl",
        MODEL_DIR / "best_model.pkl",
        MODEL_DIR / "best_model_xgboost.pkl"
    ]
    
    # Find first existing model
    model_path = next((p for p in model_candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            f"No model file found. Searched for:\n"
            + "\n".join(f"  - {p}" for p in model_candidates) +
            "\n\nPlease run the training notebooks first:\n"
            "  1. notebooks/02_data_preprocessing.ipynb\n"
            "  2. notebooks/03_model_training.ipynb"
        )
    
    # Check other required files
    preprocessor_path = MODEL_DIR / "preprocessor.pkl"
    encoder_path = MODEL_DIR / "target_encoder.pkl"
    
    missing_files = []
    if not preprocessor_path.exists():
        missing_files.append(f"preprocessor.pkl")
    if not encoder_path.exists():
        missing_files.append(f"target_encoder.pkl")
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required artifacts: {', '.join(missing_files)}\n"
            "Please run notebooks/02_data_preprocessing.ipynb first."
        )
    
    # Load artifacts
    logger.info(f"Loading model from: {model_path.name}")
    _artifacts.model = joblib.load(model_path)
    _artifacts.preprocessor = joblib.load(preprocessor_path)
    _artifacts.target_encoder = joblib.load(encoder_path)
    
    # Load or infer feature schema
    _artifacts.feature_names, _, _ = get_feature_schema()
    _artifacts.n_features = len(_artifacts.feature_names)
    _artifacts.target_classes = list(_artifacts.target_encoder.classes_)
    
    # Load manifest if exists
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, 'r') as f:
            _artifacts.manifest = json.load(f)
        logger.info(f"Loaded manifest from: {MANIFEST_PATH}")
    
    # Validate consistency
    _artifacts.validate_consistency()
    _artifacts.is_loaded = True
    
    logger.info(f"✓ All artifacts loaded successfully")
    return _artifacts.model, _artifacts.preprocessor, _artifacts.target_encoder


def get_feature_schema() -> Tuple[List[str], List[str], List[str]]:
    """
    Get the list of numerical and categorical features.
    Uses cached metadata if available, otherwise infers from processed training data.
    
    Returns:
        Tuple of (numerical_cols, categorical_cols, all_cols)
    """
    # Try loading from metadata cache first
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            return (
                metadata.get('numerical_cols', []),
                metadata.get('categorical_cols', []),
                metadata.get('all_cols', [])
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load metadata: {e}, falling back to inference")
    
    # Fallback: Read from PROCESSED train data (not raw merged data!)
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at: {TRAIN_DATA_PATH}\n"
            "Please run notebooks/02_data_preprocessing.ipynb first."
        )
    
    logger.info(f"Inferring schema from: {TRAIN_DATA_PATH}")
    df_sample = pd.read_csv(TRAIN_DATA_PATH, nrows=100)
    
    # Target is the only column to exclude in processed data
    cols_to_drop = ['Target']
    X = df_sample.drop(columns=[c for c in cols_to_drop if c in df_sample.columns])
    
    all_cols = X.columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Cache for next time
    save_app_metadata(numerical_cols, categorical_cols, all_cols)
    
    return numerical_cols, categorical_cols, all_cols


def get_feature_info() -> Dict[str, Dict[str, Any]]:
    """
    Get feature metadata including min, max, default values, and descriptions.
    Used by frontend to render form dynamically.
    
    Returns:
        Dict mapping feature names to their metadata
    """
    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        if 'feature_info' in metadata:
            return metadata['feature_info']
    
    # Compute from training data
    if not TRAIN_DATA_PATH.exists():
        return {}
    
    df = pd.read_csv(TRAIN_DATA_PATH, nrows=1000)
    X = df.drop(columns=['Target'], errors='ignore')
    
    feature_info = {}
    for col in X.columns:
        feature_info[col] = {
            'min': float(X[col].min()),
            'max': float(X[col].max()),
            'mean': float(X[col].mean()),
            'default': float(X[col].median()),
            'dtype': str(X[col].dtype),
            'description': _get_feature_description(col)
        }
    
    return feature_info


def _get_feature_description(col: str) -> str:
    """Get human-readable description for a feature."""
    descriptions = {
        'Age': 'Age at enrollment (years)',
        'Gender': '0 = Female, 1 = Male',
        'Marital_Status': 'Marital status code',
        'Admission_Grade': 'Grade at admission (0-200)',
        'Displaced': 'Student displaced from home (0/1)',
        'Debtor': 'Has unpaid fees (0/1)',
        'Tuition_Fees_Up_To_Date': 'Fees paid up to date (0/1)',
        'Scholarship_Holder': 'Has scholarship (0/1)',
        'Unemployment_Rate': 'Regional unemployment rate %',
        'Inflation_Rate': 'Inflation rate %',
        'GDP': 'GDP growth rate %',
    }
    return descriptions.get(col, f'{col.replace("_", " ")}')


def get_categorical_options(feature_name: str) -> List[Any]:
    """
    Get unique values for a categorical feature to populate dropdowns.
    
    Args:
        feature_name: Name of the categorical feature
        
    Returns:
        List of unique values for the feature
    """
    # Try loading from metadata cache first
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            if 'categorical_options' in metadata and feature_name in metadata['categorical_options']:
                return metadata['categorical_options'][feature_name]
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Fallback: Read from training data
    if not TRAIN_DATA_PATH.exists():
        return []
    
    try:
        df = pd.read_csv(TRAIN_DATA_PATH, usecols=[feature_name])
        return sorted(df[feature_name].dropna().unique().tolist())
    except Exception as e:
        logger.warning(f"Failed to get options for {feature_name}: {e}")
        return []


def validate_input(input_data: pd.DataFrame, all_cols: List[str]) -> pd.DataFrame:
    """
    Validate and reorder input data to match expected schema.
    
    Args:
        input_data: DataFrame with input features
        all_cols: Expected column order
        
    Returns:
        Validated and reordered DataFrame
        
    Raises:
        ValueError: If input is invalid
    """
    # Check for missing columns
    missing_cols = set(all_cols) - set(input_data.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required features: {missing_cols}\n"
            f"Expected {len(all_cols)} features, got {len(input_data.columns)}"
        )
    
    # Check for extra columns (warning only)
    extra_cols = set(input_data.columns) - set(all_cols)
    if extra_cols:
        logger.warning(f"Ignoring extra columns: {extra_cols}")
    
    # Reorder to match expected schema
    return input_data[all_cols]


def make_prediction(
    model: Any,
    preprocessor: Any,
    input_data: pd.DataFrame
) -> Tuple[int, np.ndarray]:
    """
    Make a prediction using the model and preprocessor.
    
    Args:
        model: Trained model
        preprocessor: Fitted ColumnTransformer
        input_data: Input DataFrame with original columns (already validated)
        
    Returns:
        Tuple of (predicted_class_index, probability_array)
        
    Raises:
        RuntimeError: If prediction fails
    """
    try:
        # Preprocess the input
        X_processed = preprocessor.transform(input_data)
        
        # Validate feature count
        expected = model.n_features_in_ if hasattr(model, 'n_features_in_') else X_processed.shape[1]
        if X_processed.shape[1] != expected:
            raise ValueError(
                f"Feature count mismatch after preprocessing: "
                f"got {X_processed.shape[1]}, model expects {expected}"
            )
        
        # Predict
        prediction = model.predict(X_processed)[0]
        probabilities = model.predict_proba(X_processed)[0]
        
        return int(prediction), probabilities
        
    except Exception as e:
        logger.exception("Prediction failed")
        raise RuntimeError(f"Prediction failed: {e}")


def save_app_metadata(
    numerical_cols: List[str],
    categorical_cols: List[str],
    all_cols: List[str],
    categorical_options: Optional[Dict[str, List]] = None,
    feature_info: Optional[Dict[str, Dict]] = None
) -> None:
    """
    Save metadata to JSON for fast loading by the app.
    Call this at the end of preprocessing notebook.
    
    Args:
        numerical_cols: List of numerical feature names
        categorical_cols: List of categorical feature names
        all_cols: All feature names in order
        categorical_options: Optional dict of categorical feature options
        feature_info: Optional dict of feature metadata
    """
    metadata = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'all_cols': all_cols,
        'n_features': len(all_cols),
        'categorical_options': categorical_options or {},
        'feature_info': feature_info or {},
        'created_at': datetime.now().isoformat()
    }
    
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ App metadata saved to: {METADATA_PATH}")


def save_manifest(
    model_path: Path,
    preprocessor_path: Path,
    encoder_path: Path,
    accuracy: float = 0.0,
    extra_info: Optional[Dict] = None
) -> None:
    """
    Save a manifest file tracking all artifact versions.
    
    Args:
        model_path: Path to model file
        preprocessor_path: Path to preprocessor file
        encoder_path: Path to encoder file
        accuracy: Model accuracy on test set
        extra_info: Additional metadata to include
    """
    manifest = {
        'created_at': datetime.now().isoformat(),
        'model': {
            'path': str(model_path.name),
            'hash': _compute_file_hash(model_path)
        },
        'preprocessor': {
            'path': str(preprocessor_path.name),
            'hash': _compute_file_hash(preprocessor_path)
        },
        'encoder': {
            'path': str(encoder_path.name),
            'hash': _compute_file_hash(encoder_path)
        },
        'accuracy': accuracy,
        **(extra_info or {})
    }
    
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"✓ Manifest saved to: {MANIFEST_PATH}")


def get_loaded_artifacts() -> Optional[ModelArtifacts]:
    """Get the currently loaded model artifacts container."""
    return _artifacts
