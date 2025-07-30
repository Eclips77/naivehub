from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import os
from services.classifier import NaiveBayesPredictor
from typing import Dict, Optional, Any
from datetime import datetime

app = FastAPI(title="NaiveHub Prediction Server")

# Configuration
STORAGE_URL = os.getenv("STORAGE_URL", "http://localhost:8002")
MODELS_DIR = "models"
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "5"))  # Maximum number of models in cache

# Global storage for loaded predictors and cache management
loaded_models: Dict[str, NaiveBayesPredictor] = {}
model_cache_info: Dict[str, Dict[str, Any]] = {}  # Store cache metadata

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)


def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0  # Return 0 if psutil not available


def manage_cache():
    """Manage cache size by removing least recently used models."""
    while len(loaded_models) >= MAX_CACHE_SIZE:
        # Find the least recently used model
        oldest_model = min(
            model_cache_info.items(),
            key=lambda x: x[1]["last_used"]
        )[0]
        
        # Remove from cache
        del loaded_models[oldest_model]
        del model_cache_info[oldest_model]
        
        # Try to remove local file (optional)
        model_file = os.path.join(MODELS_DIR, f"{oldest_model}.json")
        if os.path.exists(model_file):
            try:
                os.remove(model_file)
            except:
                pass  # Ignore file removal errors


class PredictRequest(BaseModel):
    model_name: str
    record: dict


class LoadModelRequest(BaseModel):
    model_name: str


@app.post("/load_model")
async def load_model_from_storage(req: LoadModelRequest):
    """Load a model from storage server and cache it locally."""
    try:
        # Check cache first
        manage_cache()
        
        # Request model from storage server
        response = requests.get(f"{STORAGE_URL}/models/{req.model_name}")
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        model_response = response.json()
        model_data = model_response["model_data"]
        
        # Save model to local JSON file (optional backup)
        model_file_path = os.path.join(MODELS_DIR, f"{req.model_name}.json")
        with open(model_file_path, "w") as f:
            json.dump(model_data, f, indent=2)
        
        # Load model into memory for predictions
        loaded_models[req.model_name] = NaiveBayesPredictor(model_data)
        
        # Update cache info
        model_cache_info[req.model_name] = {
            "loaded_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "file_path": model_file_path,
            "classes": model_data.get("classes", []),
            "features": list(model_data.get("likelihoods", {}).keys()),
            "metadata": model_response.get("metadata", {})
        }
        
        return {
            "message": f"Model '{req.model_name}' loaded successfully",
            "classes": model_data.get("classes", []),
            "features": list(model_data.get("likelihoods", {}).keys()),
            "cache_size": len(loaded_models),
            "memory_usage_mb": get_memory_usage()
        }
        
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to storage server: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.post("/predict")
async def predict(req: PredictRequest):
    """Make a prediction using a cached model."""
    try:
        # Auto-load model if not in cache
        if req.model_name not in loaded_models:
            # Try to load model from storage
            try:
                load_request = LoadModelRequest(model_name=req.model_name)
                await load_model_from_storage(load_request)
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model '{req.model_name}' not found or cannot be loaded: {str(e)}"
                )

        # Update last used time
        if req.model_name in model_cache_info:
            model_cache_info[req.model_name]["last_used"] = datetime.now().isoformat()

        predictor = loaded_models[req.model_name]
        prediction, probabilities = predictor.predict_with_confidence(req.record)
        
        return {
            "prediction": prediction,
            "confidence": probabilities,
            "model_name": req.model_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/unload_model")
async def unload_model(model_name: str):
    """Remove a model from cache."""
    try:
        if model_name not in loaded_models:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not loaded")
        
        # Remove from cache
        del loaded_models[model_name]
        
        # Remove cache info
        if model_name in model_cache_info:
            cache_info = model_cache_info[model_name]
            del model_cache_info[model_name]
            
            # Optionally remove local file
            if "file_path" in cache_info and os.path.exists(cache_info["file_path"]):
                try:
                    os.remove(cache_info["file_path"])
                except:
                    pass  # Ignore file removal errors
        
        return {
            "message": f"Model '{model_name}' unloaded successfully",
            "cache_size": len(loaded_models),
            "memory_usage_mb": get_memory_usage()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error unloading model: {str(e)}")


@app.get("/models")
async def list_available_models():
    """Get list of models available on storage server."""
    try:
        response = requests.get(f"{STORAGE_URL}/models")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        storage_models = response.json()
        
        # Add cache status to each model
        for model in storage_models.get("available_models", []):
            model["cached"] = model["model_name"] in loaded_models
            if model["cached"]:
                cache_info = model_cache_info.get(model["model_name"], {})
                model["cache_info"] = {
                    "loaded_at": cache_info.get("loaded_at"),
                    "last_used": cache_info.get("last_used")
                }
        
        return storage_models
        
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to storage server: {str(e)}")


@app.get("/cache")
async def get_cache_status():
    """Get current cache status."""
    return {
        "cached_models": list(loaded_models.keys()),
        "cache_size": len(loaded_models),
        "max_cache_size": MAX_CACHE_SIZE,
        "memory_usage_mb": get_memory_usage(),
        "cache_details": model_cache_info
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check storage server connectivity
        storage_response = requests.get(f"{STORAGE_URL}/health", timeout=5)
        storage_healthy = storage_response.status_code == 200
    except:
        storage_healthy = False
    
    return {
        "status": "healthy",
        "cached_models": len(loaded_models),
        "cache_limit": MAX_CACHE_SIZE,
        "memory_usage_mb": get_memory_usage(),
        "storage_server_url": STORAGE_URL,
        "storage_server_healthy": storage_healthy
    }
