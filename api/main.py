"""
Phase 4: FastAPI Prediction Service
API to serve trading signal predictions using the best CNN model
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import pandas as pd
from tensorflow import keras
import os
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="NVDA Trading Signal API",
    description="CNN-based trading signal prediction for NVDA stock",
    version="1.0.0"
)

# Global model variable
model = None
feature_columns = None
MODEL_PATH = "models/saved_models/CustomCNN.h5"  # Best model from Phase 3

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    features: List[float] = Field(
        ...,
        description="List of 28 normalized feature values",
        min_length=28,
        max_length=28
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5] * 28  # Example with 28 features
            }
        }

class SequencePredictionRequest(BaseModel):
    """Request model for sequence-based prediction"""
    sequences: List[List[float]] = Field(
        ...,
        description="List of 10 timesteps, each with 28 features",
        min_length=10,
        max_length=10
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "sequences": [[0.5] * 28 for _ in range(10)]
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    signal: int = Field(..., description="Predicted signal: -1 (Short), 0 (Hold), 1 (Long)")
    signal_name: str = Field(..., description="Signal name: 'SHORT', 'HOLD', or 'LONG'")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: dict = Field(..., description="Class probabilities")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_name: str = Field(..., description="Model used for prediction")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    model_path: str
    timestamp: str

# ============================================================
# MODEL LOADING
# ============================================================

def load_model_on_startup():
    """Load the trained model on API startup"""
    global model, feature_columns
    
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        # Load model
        model = keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        
        # Load feature columns from training data
        train_df = pd.read_csv('data/NVDA_train.csv', index_col=0, parse_dates=True)
        feature_columns = [col for col in train_df.columns if col.endswith('_norm')]
        print(f"✅ Loaded {len(feature_columns)} feature columns")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def signal_to_name(signal: int) -> str:
    """Convert signal integer to name"""
    signal_map = {
        0: "SHORT",
        1: "HOLD",
        2: "LONG"
    }
    return signal_map.get(signal, "UNKNOWN")

def adjusted_signal(signal: int) -> int:
    """Convert model output [0,1,2] back to [-1,0,1]"""
    return signal - 1

def validate_features(features: List[float], expected_length: int = 28) -> None:
    """Validate feature input"""
    if len(features) != expected_length:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_length} features, got {len(features)}"
        )
    
    if any(not isinstance(f, (int, float)) for f in features):
        raise HTTPException(
            status_code=400,
            detail="All features must be numeric values"
        )

# ============================================================
# API ENDPOINTS
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    success = load_model_on_startup()
    if not success:
        print("⚠️ Warning: Model failed to load on startup")

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "NVDA Trading Signal API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=MODEL_PATH,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: SequencePredictionRequest):
    """
    Predict trading signal from a sequence of features
    
    **Input**: 10 timesteps, each with 28 normalized features
    
    **Output**: Trading signal (-1: Short, 0: Hold, 1: Long) with confidence
    
    **Example**:
    ```json
    {
      "sequences": [
        [0.5, 0.3, -0.2, ...],  // Timestep 1 (28 features)
        [0.4, 0.2, -0.1, ...],  // Timestep 2 (28 features)
        ...                      // 10 timesteps total
      ]
    }
    ```
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Validate input
        if len(request.sequences) != 10:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 10 timesteps, got {len(request.sequences)}"
            )
        
        for i, timestep in enumerate(request.sequences):
            validate_features(timestep)
        
        # Prepare input for model
        input_array = np.array(request.sequences).reshape(1, 10, 28)
        
        # Make prediction
        predictions = model.predict(input_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        # Get probabilities for all classes
        probabilities = {
            "SHORT": float(predictions[0][0]),
            "HOLD": float(predictions[0][1]),
            "LONG": float(predictions[0][2])
        }
        
        # Convert signal back to [-1, 0, 1]
        signal = adjusted_signal(predicted_class)
        signal_name = signal_to_name(predicted_class)
        
        return PredictionResponse(
            signal=signal,
            signal_name=signal_name,
            confidence=confidence,
            probabilities=probabilities,
            timestamp=datetime.now().isoformat(),
            model_name="CustomCNN"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict/single", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(request: PredictionRequest):
    """
    Predict trading signal from a single feature vector
    (Creates a sequence by repeating the same features)
    
    **Input**: 28 normalized features
    
    **Output**: Trading signal with confidence
    
    **Note**: This is a simplified endpoint. For better predictions, use /predict with proper sequences.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Validate input
        validate_features(request.features)
        
        # Create sequence by repeating features (not ideal, but works for demo)
        sequence = np.array([request.features] * 10).reshape(1, 10, 28)
        
        # Make prediction
        predictions = model.predict(sequence, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        # Get probabilities
        probabilities = {
            "SHORT": float(predictions[0][0]),
            "HOLD": float(predictions[0][1]),
            "LONG": float(predictions[0][2])
        }
        
        # Convert signal
        signal = adjusted_signal(predicted_class)
        signal_name = signal_to_name(predicted_class)
        
        return PredictionResponse(
            signal=signal,
            signal_name=signal_name,
            confidence=confidence,
            probabilities=probabilities,
            timestamp=datetime.now().isoformat(),
            model_name="CustomCNN (single-input mode)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        return {
            "model_name": "CustomCNN",
            "model_path": MODEL_PATH,
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "num_parameters": int(model.count_params()),
            "num_features": len(feature_columns) if feature_columns else 28,
            "sequence_length": 10,
            "output_classes": {
                "0": "SHORT (-1)",
                "1": "HOLD (0)",
                "2": "LONG (1)"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("NVDA TRADING SIGNAL API")
    print("="*60)
    print("\nStarting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )