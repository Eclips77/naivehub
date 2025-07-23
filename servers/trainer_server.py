from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from managers.data_manager import DataManager
from services.trainer import NaiveBayesTrainer
from services.classifier import NaiveBayesPredictor
from services.evaluator import NaiveBayesEvaluator
from typing import Dict, Any
import os

app = FastAPI(title="NaiveHub Training Server")

# Global storage for trained models and test data
trained_models: Dict[str, Dict[str, Any]] = {}
model_test_data: Dict[str, Any] = {}  # Store test data for each model


class TrainRequest(BaseModel):
    file_path: str
    target_column: str
    model_name: str = "default_model"


class EvaluateRequest(BaseModel):
    model_name: str


@app.post("/train")
async def train_model(req: TrainRequest):
    """Train a Naive Bayes model and store it in memory."""
    try:
        # Check if file exists and handle different path scenarios
        file_path = req.file_path
        
        # Try different possible paths
        possible_paths = [
            file_path,  # Original path as provided
            os.path.join("Data", file_path),  # In Data directory
            os.path.join("/app", file_path),  # In Docker app directory
            os.path.join("/app/Data", file_path),  # In Docker Data directory
        ]
        
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                break
        
        if not found_path:
            raise FileNotFoundError(f"File not found: {file_path}. Checked paths: {possible_paths}")
        
        # Load and prepare data
        data_manager = DataManager(found_path)
        train_df, test_df = data_manager.prepare_data()

        # Validate target column
        if req.target_column not in train_df.columns:
            available_columns = list(train_df.columns)
            raise ValueError(f"Target column '{req.target_column}' not found. Available columns: {available_columns}")

        # Train the model
        trainer = NaiveBayesTrainer()
        model = trainer.fit(train_df, req.target_column)
        
        # Store model in memory
        trained_models[req.model_name] = model
        
        # Store test data for evaluation
        model_test_data[req.model_name] = {
            "test_df": test_df,
            "target_column": req.target_column
        }
        
        return {
            "message": f"Model '{req.model_name}' trained successfully",
            "classes": model["classes"],
            "training_samples": len(train_df),
            "test_samples": len(test_df),
            "file_path_used": found_path,
            "features": model.get("feature_columns", [])
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model/{model_name}")
async def get_model(model_name: str):
    """Get a trained model for the classification server."""
    try:
        if model_name not in trained_models:
            available_models = list(trained_models.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not found. Available: {available_models}"
            )
            
        model = trained_models[model_name]
        
        return {
            "model_name": model_name,
            "model_data": model,
            "classes": model["classes"],
            "features": list(model["likelihoods"].keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model: {str(e)}")


@app.post("/evaluate")
async def evaluate_model(req: EvaluateRequest):
    """Evaluate a trained model using stored test data."""
    try:
        # Check if model exists
        if req.model_name not in trained_models:
            available_models = list(trained_models.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Model '{req.model_name}' not found. Available: {available_models}"
            )
        
        # Check if test data exists for this model
        if req.model_name not in model_test_data:
            raise HTTPException(
                status_code=404,
                detail=f"No test data found for model '{req.model_name}'. Retrain the model to generate test data."
            )
        
        # Get model and test data
        model = trained_models[req.model_name]
        test_info = model_test_data[req.model_name]
        test_df = test_info["test_df"]
        target_column = test_info["target_column"]
        
        # Check if test data is not empty
        if len(test_df) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No test data available for model '{req.model_name}'. Dataset too small to split."
            )
        
        # Create predictor and evaluator
        predictor = NaiveBayesPredictor(model)
        evaluator = NaiveBayesEvaluator(predictor)
        
        # Prepare test data
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        
        # Evaluate model
        results = evaluator.evaluate(X_test, y_test)
        
        return {
            "model_name": req.model_name,
            "accuracy": results["accuracy"],
            "accuracy_percentage": f"{results['accuracy'] * 100:.2f}%",
            "test_samples": len(test_df),
            "classes": model["classes"],
            "features": model.get("feature_columns", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")


@app.get("/models")
async def list_models():
    """Get list of available model names."""
    return {
        "available_models": list(trained_models.keys()),
        "count": len(trained_models)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_in_memory": len(trained_models),
        "models_with_test_data": len(model_test_data)
    }