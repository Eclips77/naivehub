from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.trainer import NaiveBayesTrainer
from services.classifier import NaiveBayesPredictor
from services.evaluator import NaiveBayesEvaluator
from typing import Dict, Any, Optional
import os
import requests
import pandas as pd

app = FastAPI(title="NaiveHub Training Server")

# Configuration
STORAGE_URL = os.getenv("STORAGE_URL", "http://localhost:8002")

# Global storage for training sessions and evaluation data
training_sessions: Dict[str, Dict[str, Any]] = {}  # Store evaluation data for trained models


class TrainRequest(BaseModel):
    dataset_id: str  # Dataset ID from storage server
    target_column: str
    model_name: str = "default_model"
    metadata: Optional[Dict[str, Any]] = None


@app.post("/train")
async def train_model(req: TrainRequest):
    """Train a Naive Bayes model using data from storage server."""
    try:
        # Get prepared data from storage server
        response = requests.get(f"{STORAGE_URL}/data/{req.dataset_id}")
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        dataset_response = response.json()
        
        # Validate that it's prepared data
        if dataset_response.get("type") != "prepared":
            raise HTTPException(
                status_code=400,
                detail=f"Dataset '{req.dataset_id}' is not prepared for training. Use /data/prepare endpoint first."
            )
        
        # Extract training data
        train_data = dataset_response["train_data"]
        test_data = dataset_response["test_data"]
        
        # Convert to DataFrames
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        
        # Validate target column
        if req.target_column not in train_df.columns:
            available_columns = list(train_df.columns)
            raise ValueError(f"Target column '{req.target_column}' not found. Available columns: {available_columns}")

        # Train the model
        trainer = NaiveBayesTrainer()
        model = trainer.fit(train_df, req.target_column)
        
        # Immediate evaluation using test data
        # Prepare test data for evaluation
        X_test = test_df.drop(columns=[req.target_column])
        y_test = test_df[req.target_column]
        
        # Create predictor and evaluator
        predictor = NaiveBayesPredictor(model)
        evaluator = NaiveBayesEvaluator(predictor)
        
        # Calculate accuracy
        eval_results = evaluator.evaluate(X_test, y_test)
        accuracy = eval_results["accuracy"]
        
        # Save model to storage server with accuracy
        save_model_data = {
            "model_name": req.model_name,
            "model_data": model,
            "metadata": {
                "dataset_id": req.dataset_id,
                "target_column": req.target_column,
                "accuracy": accuracy,
                "trained_at": pd.Timestamp.now().isoformat(),
                "training_samples": len(train_df),
                "test_samples": len(test_df),
                **(req.metadata or {})
            }
        }
        
        save_response = requests.post(f"{STORAGE_URL}/models/save", json=save_model_data)
        if save_response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save model to storage: {save_response.text}"
            )
        
        # Store evaluation data for this model
        training_sessions[req.model_name] = {
            "dataset_id": req.dataset_id,
            "test_data": test_df,
            "target_column": req.target_column,
            "trained_at": pd.Timestamp.now().isoformat()
        }
        
        return {
            "message": f"Model '{req.model_name}' trained and saved successfully",
            "accuracy": accuracy,
            "features": list(model.get("likelihoods", {}).keys()),
            "classes": model["classes"]
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to storage server: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model/{model_name}")
async def get_model(model_name: str):
    """Get a trained model from storage server."""
    try:
        # Request model from storage server
        response = requests.get(f"{STORAGE_URL}/models/{model_name}")
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        return response.json()
        
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to storage server: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model: {str(e)}")


@app.get("/models")
async def list_models():
    """Get list of available models from storage server."""
    try:
        response = requests.get(f"{STORAGE_URL}/models")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to storage server: {str(e)}")


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
        "training_sessions": len(training_sessions),
        "storage_server_url": STORAGE_URL,
        "storage_server_healthy": storage_healthy
    }