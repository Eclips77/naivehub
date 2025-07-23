from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from managers.data_manager import DataManager
from services.trainer import NaiveBayesTrainer
from typing import Dict, Any

app = FastAPI(title="NaiveHub Training Server")

# Global storage for trained models
trained_models: Dict[str, Dict[str, Any]] = {}


class TrainRequest(BaseModel):
    file_path: str
    target_column: str
    model_name: str = "default_model"


@app.post("/train")
async def train_model(req: TrainRequest):
    """Train a Naive Bayes model and store it in memory."""
    try:
        # Load and prepare data
        data_manager = DataManager(req.file_path)
        train_df, test_df = data_manager.prepare_data()

        # Validate target column
        if req.target_column not in train_df.columns:
            raise ValueError(f"Target column '{req.target_column}' not found in data.")

        # Train the model
        trainer = NaiveBayesTrainer()
        model = trainer.fit(train_df, req.target_column)
        
        # Store model in memory
        trained_models[req.model_name] = model
        
        return {
            "message": f"Model '{req.model_name}' trained successfully",
            "classes": model["classes"],
            "training_samples": len(train_df)
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
        "models_in_memory": len(trained_models)
    }