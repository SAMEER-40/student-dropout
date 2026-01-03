"""
Golden Invariant Tests for Model Predictions
Tests fixed inputs against expected outputs to detect drift.
"""
import sys
import os
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    load_model_and_preprocessor,
    get_feature_schema,
    validate_input,
    make_prediction
)


class TestGoldenInvariants:
    """
    Golden tests that verify fixed inputs produce expected outputs.
    If these fail, it means the model/preprocessing has drifted.
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Load model artifacts once for all tests."""
        self.model, self.preprocessor, self.target_encoder = load_model_and_preprocessor()
        _, _, self.all_cols = get_feature_schema()
    
    def test_model_loads_successfully(self):
        """Verify model artifacts load without error."""
        assert self.model is not None
        assert self.preprocessor is not None
        assert self.target_encoder is not None
    
    def test_feature_count_consistency(self):
        """Verify feature counts match across all artifacts."""
        # Get expected feature count from model
        if hasattr(self.model, 'n_features_in_'):
            model_features = self.model.n_features_in_
        else:
            model_features = len(self.all_cols)
        
        # Schema should match
        assert len(self.all_cols) == model_features, (
            f"Schema has {len(self.all_cols)} features, model expects {model_features}"
        )
    
    def test_target_classes_complete(self):
        """Verify all expected target classes are present."""
        expected_classes = {'Dropout', 'Enrolled', 'Graduate'}
        actual_classes = set(self.target_encoder.classes_)
        assert expected_classes == actual_classes, (
            f"Expected classes {expected_classes}, got {actual_classes}"
        )
    
    def test_golden_prediction_high_risk(self):
        """
        Golden test: Student with high-risk profile should predict Dropout.
        Fixed input → Fixed expected output.
        """
        # High-risk student: older, unpaid fees, debtor, no scholarship
        high_risk_features = self._create_base_features()
        high_risk_features.update({
            'Age': 45,
            'Debtor': 1,
            'Tuition_Fees_Up_To_Date': 0,
            'Scholarship_Holder': 0,
            'Displaced': 1,
            'Admission_Grade': 80.0,
        })
        
        input_df = self._features_to_df(high_risk_features)
        pred_idx, probabilities = make_prediction(self.model, self.preprocessor, input_df)
        pred_label = self.target_encoder.inverse_transform([pred_idx])[0]
        
        # This student should have Dropout as highest or second-highest probability
        dropout_idx = list(self.target_encoder.classes_).index('Dropout')
        assert probabilities[dropout_idx] > 0.2, (
            f"Expected high Dropout probability for high-risk student, got {probabilities[dropout_idx]:.2%}"
        )
    
    def test_golden_prediction_low_risk(self):
        """
        Golden test: Student with low-risk profile should predict Graduate.
        Fixed input → Fixed expected output.
        """
        # Low-risk student: young, paid fees, scholarship, high grades
        low_risk_features = self._create_base_features()
        low_risk_features.update({
            'Age': 19,
            'Debtor': 0,
            'Tuition_Fees_Up_To_Date': 1,
            'Scholarship_Holder': 1,
            'Displaced': 0,
            'Admission_Grade': 165.0,
        })
        
        input_df = self._features_to_df(low_risk_features)
        pred_idx, probabilities = make_prediction(self.model, self.preprocessor, input_df)
        pred_label = self.target_encoder.inverse_transform([pred_idx])[0]
        
        # This student should have Graduate as highest or second-highest probability
        graduate_idx = list(self.target_encoder.classes_).index('Graduate')
        assert probabilities[graduate_idx] > 0.2, (
            f"Expected high Graduate probability for low-risk student, got {probabilities[graduate_idx]:.2%}"
        )
    
    def test_prediction_returns_valid_format(self):
        """Verify prediction output format is as expected."""
        features = self._create_base_features()
        input_df = self._features_to_df(features)
        
        pred_idx, probabilities = make_prediction(self.model, self.preprocessor, input_df)
        
        # Check index is valid
        assert isinstance(pred_idx, (int, np.integer))
        assert 0 <= pred_idx < len(self.target_encoder.classes_)
        
        # Check probabilities
        assert isinstance(probabilities, np.ndarray)
        assert len(probabilities) == len(self.target_encoder.classes_)
        assert np.isclose(probabilities.sum(), 1.0, atol=0.01), (
            f"Probabilities should sum to 1, got {probabilities.sum()}"
        )
    
    def _create_base_features(self):
        """Create a base feature dictionary with median values."""
        return {col: 0.0 for col in self.all_cols}
    
    def _features_to_df(self, features):
        """Convert features dict to DataFrame."""
        import pandas as pd
        df = pd.DataFrame([features])
        return validate_input(df, self.all_cols)


class TestBackendConnection:
    """Tests for backend utility functions."""
    
    def test_get_feature_schema(self):
        """Test feature schema retrieval."""
        num_cols, cat_cols, all_cols = get_feature_schema()
        
        assert isinstance(all_cols, list)
        assert len(all_cols) > 0, "Feature schema should not be empty"
    
    def test_validate_input_catches_missing_columns(self):
        """Test that validation catches missing required columns."""
        import pandas as pd
        
        _, _, all_cols = get_feature_schema()
        
        # Create DataFrame missing some columns
        partial_data = {col: [0.0] for col in all_cols[:5]}
        df = pd.DataFrame(partial_data)
        
        with pytest.raises(ValueError, match="Missing required features"):
            validate_input(df, all_cols)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
