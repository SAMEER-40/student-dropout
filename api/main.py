"""
FastAPI Backend for Student Dropout Prediction
Provides REST API endpoints for predictions and model introspection.
"""
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    PredictionRequest,
    PredictionResponse,
    FeatureSchemaResponse,
    HealthResponse,
    ErrorResponse
)
from .services.prediction import PredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global prediction service (loaded once at startup)
prediction_service: Optional[PredictionService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    Load model artifacts once at startup to avoid per-request loading.
    """
    global prediction_service
    
    logger.info("🚀 Starting up - Loading model artifacts...")
    try:
        prediction_service = PredictionService()
        prediction_service.load_artifacts()
        logger.info("✓ Model artifacts loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model artifacts: {e}")
        # Allow startup but predictions will fail gracefully
        prediction_service = None
    
    yield  # Server runs
    
    logger.info("👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Student Dropout Prediction API",
    description="ML-powered API for predicting student dropout risk",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # React dev
        "http://localhost:5173",     # Vite dev
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model loading status."""
    return HealthResponse(
        status="healthy",
        model_loaded=prediction_service is not None and prediction_service.is_ready(),
        version="2.0.0"
    )


@app.get("/schema", response_model=FeatureSchemaResponse, tags=["Schema"])
async def get_feature_schema():
    """
    Get feature schema for dynamic form rendering.
    Frontend should use this to build forms - never hardcode features.
    """
    if prediction_service is None or not prediction_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    return FeatureSchemaResponse(
        features=prediction_service.get_feature_names(),
        feature_info=prediction_service.get_feature_info(),
        target_classes=prediction_service.get_target_classes()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a dropout prediction for a student.
    
    Optionally include SHAP explanations with `explain=true`.
    Use `top_k_features` to limit explanation size (default: 10).
    """
    if prediction_service is None or not prediction_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        result = prediction_service.predict(
            features=request.features,
            explain=request.explain,
            top_k=request.top_k_features
        )
        
        return PredictionResponse(
            prediction=result["prediction"],
            prediction_label=result["prediction_label"],
            probabilities=result["probabilities"],
            confidence=result["confidence"],
            explanation=result.get("explanation")
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler to prevent leaking internal errors."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred. Please try again later."}
    )
