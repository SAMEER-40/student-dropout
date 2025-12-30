"""
Pure Prediction Service Layer
No FastAPI imports, no request objects, no globals.
This layer is boringly pure - easy to test.
"""
import numpy as np
import pandas as pd
import shap
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Use absolute import from src
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import (
    load_model_and_preprocessor,
    get_feature_schema,
    get_feature_info,
    validate_input,
    get_loaded_artifacts
)

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Pure prediction service with no web framework dependencies.
    Handles model loading, predictions, and SHAP explanations.
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.target_encoder = None
        self.feature_names: List[str] = []
        self.target_classes: List[str] = []
        self._shap_explainer = None
        self._loaded = False
    
    def load_artifacts(self) -> None:
        """Load model artifacts from disk."""
        self.model, self.preprocessor, self.target_encoder = load_model_and_preprocessor()
        
        # Get feature schema
        _, _, self.feature_names = get_feature_schema()
        self.target_classes = list(self.target_encoder.classes_)
        
        self._loaded = True
        logger.info(f"Loaded model with {len(self.feature_names)} features, "
                    f"{len(self.target_classes)} classes")
    
    def is_ready(self) -> bool:
        """Check if service is ready for predictions."""
        return self._loaded and self.model is not None
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of expected feature names."""
        return self.feature_names
    
    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for each feature (for form rendering)."""
        return get_feature_info()
    
    def get_target_classes(self) -> List[str]:
        """Get list of target class labels."""
        return self.target_classes
    
    def predict(
        self,
        features: Dict[str, float],
        explain: bool = False,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Make a prediction for the given features.
        
        Args:
            features: Dictionary mapping feature names to values
            explain: Whether to include SHAP explanations
            top_k: Number of top features to include in explanation
            
        Returns:
            Dictionary containing prediction, probabilities, and optional explanation
            
        Raises:
            ValueError: If features are invalid
        """
        if not self.is_ready():
            raise RuntimeError("Service not initialized. Call load_artifacts() first.")
        
        # Convert features dict to DataFrame
        input_df = self._features_to_dataframe(features)
        
        # Validate and reorder columns
        input_df = validate_input(input_df, self.feature_names)
        
        # Preprocess
        X_processed = self.preprocessor.transform(input_df)
        
        # Predict
        prediction_idx = int(self.model.predict(X_processed)[0])
        probabilities = self.model.predict_proba(X_processed)[0]
        
        # Build result
        result = {
            "prediction": prediction_idx,
            "prediction_label": self.target_classes[prediction_idx],
            "probabilities": {
                cls: float(prob) 
                for cls, prob in zip(self.target_classes, probabilities)
            },
            "confidence": float(max(probabilities))
        }
        
        # Add SHAP explanation if requested
        if explain:
            result["explanation"] = self._compute_shap_explanation(
                X_processed, 
                features,
                top_k
            )
        
        return result
    
    def _features_to_dataframe(self, features: Dict[str, float]) -> pd.DataFrame:
        """Convert features dictionary to DataFrame."""
        # Ensure all expected features are present
        missing = set(self.feature_names) - set(features.keys())
        if missing:
            raise ValueError(
                f"Missing required features: {missing}\n"
                f"Expected features: {self.feature_names}"
            )
        
        # Build DataFrame with features in expected order
        data = {name: [features[name]] for name in self.feature_names}
        return pd.DataFrame(data)
    
    def _compute_shap_explanation(
        self,
        X_processed: np.ndarray,
        original_features: Dict[str, float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Compute SHAP explanation for a prediction.
        
        Args:
            X_processed: Preprocessed input array
            original_features: Original feature values
            top_k: Number of top features to return
            
        Returns:
            List of feature explanations sorted by absolute SHAP value
        """
        try:
            # Lazy initialize SHAP explainer
            if self._shap_explainer is None:
                logger.info("Initializing SHAP TreeExplainer...")
                self._shap_explainer = shap.TreeExplainer(self.model)
            
            # Get SHAP values
            shap_values = self._shap_explainer.shap_values(X_processed)
            
            # For multi-class, shap_values is a list per class
            # We'll explain the predicted class (class with highest prob)
            if isinstance(shap_values, list):
                # Use first class (Dropout) for dropout prediction
                sv = shap_values[0][0]
            else:
                sv = shap_values[0]
            
            # Build explanation
            explanations = []
            for i, (name, shap_val) in enumerate(zip(self.feature_names, sv)):
                explanations.append({
                    "feature": name,
                    "value": float(original_features.get(name, 0)),
                    "shap_value": float(shap_val),
                    "impact": "positive" if shap_val > 0 else "negative"
                })
            
            # Sort by absolute SHAP value and take top_k
            explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            return explanations[:top_k]
            
        except Exception as e:
            logger.exception("SHAP explanation failed")
            return []
    
    def predict_batch(
        self,
        batch_features: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple samples.
        More efficient than calling predict() multiple times.
        
        Args:
            batch_features: List of feature dictionaries
            
        Returns:
            List of prediction results (without explanations for efficiency)
        """
        if not self.is_ready():
            raise RuntimeError("Service not initialized. Call load_artifacts() first.")
        
        # Convert all to DataFrame
        dfs = [self._features_to_dataframe(f) for f in batch_features]
        input_df = pd.concat(dfs, ignore_index=True)
        
        # Validate
        input_df = validate_input(input_df, self.feature_names)
        
        # Preprocess and predict
        X_processed = self.preprocessor.transform(input_df)
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        # Build results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            results.append({
                "prediction": int(pred),
                "prediction_label": self.target_classes[int(pred)],
                "probabilities": {
                    cls: float(p) 
                    for cls, p in zip(self.target_classes, probs)
                },
                "confidence": float(max(probs))
            })
        
        return results
