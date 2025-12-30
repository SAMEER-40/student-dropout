"""
Pydantic schemas for API request/response validation.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature name to value",
        examples=[{
            "Age": 20,
            "Gender": 1,
            "Admission_Grade": 120.5,
            "Scholarship_Holder": 1,
            "Tuition_Fees_Up_To_Date": 1
        }]
    )
    explain: bool = Field(
        default=False,
        description="Include SHAP explanation (slower)"
    )
    top_k_features: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of top features to include in explanation"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "Age": 20,
                    "Gender": 1,
                    "Marital_Status": 1,
                    "Admission_Grade": 120.5,
                    "Scholarship_Holder": 1,
                    "Tuition_Fees_Up_To_Date": 1,
                    "Debtor": 0,
                    "Displaced": 0,
                    "Unemployment_Rate": 10.5,
                    "Inflation_Rate": 1.2,
                    "GDP": 0.5
                },
                "explain": True,
                "top_k_features": 5
            }
        }


class FeatureExplanation(BaseModel):
    """SHAP explanation for a single feature."""
    
    feature: str
    value: float
    shap_value: float
    impact: str = Field(description="'positive' or 'negative' impact on dropout")


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    
    prediction: int = Field(description="Predicted class index")
    prediction_label: str = Field(description="Predicted class label (Dropout/Enrolled/Graduate)")
    probabilities: Dict[str, float] = Field(
        description="Probability for each class"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (max probability)"
    )
    explanation: Optional[List[FeatureExplanation]] = Field(
        default=None,
        description="SHAP feature explanations (if requested)"
    )


class FeatureInfo(BaseModel):
    """Metadata for a single feature."""
    
    min: float
    max: float
    mean: float
    default: float
    dtype: str
    description: str


class FeatureSchemaResponse(BaseModel):
    """Schema information for frontend form rendering."""
    
    features: List[str] = Field(description="Ordered list of feature names")
    feature_info: Dict[str, FeatureInfo] = Field(
        description="Metadata for each feature"
    )
    target_classes: List[str] = Field(
        description="Possible prediction outcomes"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(examples=["healthy"])
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    detail: str
