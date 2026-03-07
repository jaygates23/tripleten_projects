import joblib
import numpy as np
import json
from typing import Dict, Any
import os

# Global variable to store the loaded model
model = None
metadata = None


def load_model_and_metadata():
    """
    Load the trained model and metadata from disk.
    """
    global model, metadata

    try:
        # Load the model
        model = joblib.load("model.pkl")

        # Load the metadata
        with open("model_metadata.json", "r") as f:
            metadata = json.load(f)

        print("Model and metadata loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading model or metadata: {e}")
        return False


def make_prediction(house_features: Dict[str, Any]) -> float:
    """
    Make a price prediction for a single house.
    """
    global model, metadata

    if model is None:
        raise ValueError("Model not loaded")

    if metadata is None:
        raise ValueError("Model metadata not loaded")

    # Extract features in the correct order
    feature_values = [
        house_features[feature_name]
        for feature_name in metadata["features"]
    ]

    # Convert to numpy array with shape (1, 13)
    X = np.array(feature_values).reshape(1, -1)

    # Make prediction
    prediction = model.predict(X)[0]

    # Round to 2 decimal places for currency
    return round(float(prediction), 2)


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.
    """
    global metadata

    if metadata is None:
        raise ValueError("Model metadata not loaded")

    return metadata


def check_health() -> Dict[str, Any]:
    """
    Check the health status of the service.
    """
    global model, metadata

    model_loaded = model is not None
    metadata_loaded = metadata is not None

    if model_loaded and metadata_loaded:
        status = "healthy"
        message = "Model and metadata are loaded and ready"
    else:
        status = "unhealthy"
        message = "Model or metadata not loaded"

    health_status = {
        "status": status,
        "model_loaded": model_loaded,
        "message": message
    }

    return health_status
