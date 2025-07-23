from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import os
from services.classifier import NaiveBayesPredictor
from services.evaluator import NaiveBayesEvaluator
from typing import Dict, Any
import pandas as pd

app = FastAPI(title="NaiveHub Classification Server")

# Configuration
TRAINER_URL = os.getenv("TRAINER_URL", "http://localhost:8001")
MODELS_DIR = "models"

# Global storage for loaded predictors
loaded_models: Dict[str, NaiveBayesPredictor] = {}

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)


class PredictRequest(BaseModel):
    model_name: str
    record: dict


class LoadModelRequest(BaseModel):
    model_name: str


class EvaluateRequest(BaseModel):
    model_name: str
    test_data: list  # List of records with features and target


@app.post("/load_model")
async def load_model_from_trainer(req: LoadModelRequest):
    """Load a model from training server and save it locally."""
    try:
        # Request model from training server
        response = requests.get(f"{TRAINER_URL}/model/{req.model_name}")
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        model_response = response.json()
        model_data = model_response["model_data"]
        
        # Save model to local JSON file
        model_file_path = os.path.join(MODELS_DIR, f"{req.model_name}.json")
        with open(model_file_path, "w") as f:
            json.dump(model_data, f, indent=2)
        
        # Load model into memory for predictions
        loaded_models[req.model_name] = NaiveBayesPredictor(model_data)
        
        return {"message": f"Model '{req.model_name}' loaded successfully"}
        
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to training server: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.post("/predict")
async def predict(req: PredictRequest):
    """Make a prediction using a loaded model."""
    try:
        if req.model_name not in loaded_models:
            raise HTTPException(status_code=400, detail=f"Model '{req.model_name}' not loaded")

        predictor = loaded_models[req.model_name]
        result = predictor.predict(req.record)
        
        return {"prediction": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/evaluate")
async def evaluate_model(req: EvaluateRequest):
    """Evaluate model accuracy on test data."""
    try:
        if req.model_name not in loaded_models:
            raise HTTPException(status_code=400, detail=f"Model '{req.model_name}' not loaded")

        predictor = loaded_models[req.model_name]
        evaluator = NaiveBayesEvaluator(predictor)
        
        # Convert test data to DataFrame
        df = pd.DataFrame(req.test_data)
        
        # Assume last column is the target
        target_column = df.columns[-1]
        X_test = df.drop(columns=[target_column])
        y_test = df[target_column]
        
        # Evaluate
        results = evaluator.evaluate(X_test, y_test)
        
        return {
            "accuracy": results["accuracy"],
            "accuracy_percentage": f"{results['accuracy'] * 100:.2f}%"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")


@app.get("/models")
async def list_available_models():
    """Get list of models available on training server."""
    try:
        response = requests.get(f"{TRAINER_URL}/models")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to training server: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "loaded_models": list(loaded_models.keys())
    }
