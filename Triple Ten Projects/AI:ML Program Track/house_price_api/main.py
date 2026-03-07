from fastapi import FastAPI, HTTPException
from schemas import (
    HousePredictionRequest,
    PredictionResponse,
    ModelInfoResponse,
    HealthCheckResponse
)
from api import (
    load_model_and_metadata,
    make_prediction,
    get_model_info,
    check_health
)

# Create FastAPI application
app = FastAPI(
    title="House Price Prediction API",
    description="Machine learning service for predicting US apartment prices based on 13 features",
    version="1.0.0"
)

# Load model at startup (loads ONCE)
@app.on_event("startup")
async def startup_event():
    """Load model and metadata when the service starts"""
    success = load_model_and_metadata()
    if not success:
        print("WARNING: Failed to load model at startup")

# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Check if the service is healthy and model is loaded.

    Returns:
        HealthCheckResponse with current service status
    """
    health = check_health()
    return HealthCheckResponse(**health)

# Model info endpoint
@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about the loaded model.

    Returns:
        ModelInfoResponse with model metadata

    Raises:
        HTTPException: If model metadata is not loaded
    """
    try:
        info = get_model_info()
        return ModelInfoResponse(**info)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HousePredictionRequest):
    """
    Predict house price based on property features.

    Args:
        request: HousePredictionRequest with all 13 property features

    Returns:
        PredictionResponse with predicted price

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    try:
        # Convert request to dictionary
        features_dict = request.dict()

        # Make prediction
        predicted_price = make_prediction(features_dict)

        # Get model version from metadata
        model_info = get_model_info()
        model_version = model_info.get("version", "unknown")

        # Create and return PredictionResponse
        return PredictionResponse(
            predicted_price=predicted_price,
            currency="USD",
            model_version=model_version
        )

    except ValueError as e:
        # Model not loaded / metadata not loaded
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Other errors
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Root endpoint for basic information
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "House Price Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/health - Check service health",
            "/model/info - Get model information",
            "/predict - Make price prediction",
            "/docs - Interactive API documentation"
        ]
    }
