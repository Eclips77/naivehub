from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import json
import os
import requests
from typing import Dict, Any, List, Optional
from managers.data_manager import DataManager
from utils.data_loader import DataLoader
import uuid
from datetime import datetime

app = FastAPI(title="NaiveHub Storage Server")

# Configuration
DATA_DIR = "Data"
MODELS_DIR = "models"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Global storage for datasets and models metadata
datasets_cache: Dict[str, Dict[str, Any]] = {}
models_metadata: Dict[str, Dict[str, Any]] = {}


class DataLoadRequest(BaseModel):
    file_name: str  # Name of CSV file in Data directory
    dataset_id: Optional[str] = None  # Optional custom ID


class DataUrlRequest(BaseModel):
    url: str
    dataset_id: Optional[str] = None  # Optional custom ID


class DataPrepareRequest(BaseModel):
    dataset_id: str
    target_column: str
    train_size: float = 0.7
    random_state: int = 42


class ModelSaveRequest(BaseModel):
    model_name: str
    model_data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@app.post("/data/load")
async def load_data_from_file(req: DataLoadRequest):
    """Load CSV data from Data directory."""
    try:
        # Generate dataset ID if not provided
        dataset_id = req.dataset_id or f"dataset_{uuid.uuid4().hex[:8]}"
        
        # Check if file exists
        file_path = os.path.join(DATA_DIR, req.file_name)
        if not os.path.exists(file_path):
            # List available files only if directory exists
            if os.path.exists(DATA_DIR):
                available_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
            else:
                available_files = []
            raise HTTPException(
                status_code=404, 
                detail=f"File '{req.file_name}' not found in Data directory. Available files: {available_files}"
            )
        
        # Load data
        df = DataLoader.load_data(file_path)
        
        # Store in cache with metadata
        datasets_cache[dataset_id] = {
            "raw_data": df,
            "source": f"file:{req.file_name}",
            "loaded_at": datetime.now().isoformat(),
            "shape": df.shape,
            "columns": list(df.columns)
        }
        
        return {
            "dataset_id": dataset_id,
            "message": f"Dataset loaded successfully from {req.file_name}",
            "shape": df.shape,
            "columns": list(df.columns),
            "sample_data": df.head(3).to_dict(orient="records")
        }
        
    except FileNotFoundError:
        if os.path.exists(DATA_DIR):
            available_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        else:
            available_files = []
        raise HTTPException(
            status_code=404, 
            detail=f"File '{req.file_name}' not found in Data directory. Available files: {available_files}"
        )
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format in file '{req.file_name}': {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading data: {str(e)}")


@app.post("/data/load_from_url")
async def load_data_from_url(req: DataUrlRequest):
    """Download and load CSV data from URL."""
    temp_path = None
    try:
        # Generate dataset ID if not provided
        dataset_id = req.dataset_id or f"url_dataset_{uuid.uuid4().hex[:8]}"
        
        # Download file from URL
        response = requests.get(req.url, timeout=30)
        response.raise_for_status()
        
        # Check content type if available
        content_type = response.headers.get('content-type', '').lower()
        if content_type and 'text/csv' not in content_type and 'application/csv' not in content_type:
            # Only warn, don't fail, as many CSV files are served without proper content type
            pass
        
        # Save to temporary file
        temp_filename = f"temp_{dataset_id}.csv"
        temp_path = os.path.join(DATA_DIR, temp_filename)
        
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        # Load data
        df = DataLoader.load_data(temp_path)
        
        # Store in cache with metadata
        datasets_cache[dataset_id] = {
            "raw_data": df,
            "source": f"url:{req.url}",
            "loaded_at": datetime.now().isoformat(),
            "shape": df.shape,
            "columns": list(df.columns),
            "temp_file": temp_path
        }
        
        return {
            "dataset_id": dataset_id,
            "message": f"Dataset loaded successfully from URL",
            "shape": df.shape,
            "columns": list(df.columns),
            "sample_data": df.head(3).to_dict(orient="records")
        }
        
    except requests.Timeout:
        raise HTTPException(status_code=408, detail=f"Timeout while downloading from URL: {req.url}")
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail=f"Connection error while accessing URL: {req.url}")
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading from URL: {str(e)}")
    except pd.errors.ParserError as e:
        # Clean up temp file if parsing fails
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=f"Invalid CSV format in downloaded file: {str(e)}")
    except Exception as e:
        # Clean up temp file on any other error
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")


@app.post("/data/prepare")
async def prepare_data_for_training(req: DataPrepareRequest):
    """Clean and split data for training."""
    try:
        # Check if dataset exists
        if req.dataset_id not in datasets_cache:
            available_datasets = list(datasets_cache.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{req.dataset_id}' not found. Available: {available_datasets}"
            )
        
        # Get raw data
        dataset_info = datasets_cache[req.dataset_id]
        raw_df = dataset_info["raw_data"]
        
        # Validate target column
        if req.target_column not in raw_df.columns:
            available_columns = list(raw_df.columns)
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{req.target_column}' not found. Available columns: {available_columns}"
            )
        
        # Create DataManager for cleaning and splitting
        # We'll create a temporary file for DataManager to work with
        temp_file_path = f"{DATA_DIR}/temp_prepare_{req.dataset_id}.csv"
        raw_df.to_csv(temp_file_path, index=False)
        
        data_manager = DataManager(temp_file_path)
        train_df, test_df = data_manager.prepare_data()
        
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        # Store prepared data
        prepared_key = f"{req.dataset_id}_prepared"
        datasets_cache[prepared_key] = {
            "train_data": train_df,
            "test_data": test_df,
            "target_column": req.target_column,
            "source": dataset_info["source"],
            "prepared_at": datetime.now().isoformat(),
            "train_shape": train_df.shape,
            "test_shape": test_df.shape,
            "parent_dataset": req.dataset_id
        }
        
        return {
            "dataset_id": prepared_key,
            "message": "Data prepared successfully",
            "train_shape": train_df.shape,
            "test_shape": test_df.shape,
            "target_column": req.target_column,
            "train_sample": train_df.head(2).to_dict(orient="records"),
            "test_sample": test_df.head(2).to_dict(orient="records")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing data: {str(e)}")


@app.get("/data/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset (raw or prepared)."""
    try:
        if dataset_id not in datasets_cache:
            available_datasets = list(datasets_cache.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_id}' not found. Available: {available_datasets}"
            )
        
        dataset_info = datasets_cache[dataset_id]
        
        # Return different structure based on whether it's prepared or raw data
        if "train_data" in dataset_info:
            # Prepared data
            return {
                "dataset_id": dataset_id,
                "type": "prepared",
                "train_data": dataset_info["train_data"].to_dict(orient="records"),
                "test_data": dataset_info["test_data"].to_dict(orient="records"),
                "target_column": dataset_info["target_column"],
                "train_shape": dataset_info["train_shape"],
                "test_shape": dataset_info["test_shape"],
                "metadata": {
                    "source": dataset_info["source"],
                    "prepared_at": dataset_info["prepared_at"],
                    "parent_dataset": dataset_info.get("parent_dataset")
                }
            }
        else:
            # Raw data
            return {
                "dataset_id": dataset_id,
                "type": "raw",
                "data": dataset_info["raw_data"].to_dict(orient="records"),
                "shape": dataset_info["shape"],
                "columns": dataset_info["columns"],
                "metadata": {
                    "source": dataset_info["source"],
                    "loaded_at": dataset_info["loaded_at"]
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dataset: {str(e)}")


@app.post("/models/save")
async def save_model(req: ModelSaveRequest):
    """Save a trained model."""
    try:
        # Save model to JSON file
        model_file_path = os.path.join(MODELS_DIR, f"{req.model_name}.json")
        
        with open(model_file_path, 'w') as f:
            json.dump(req.model_data, f, indent=2)
        
        # Store metadata
        models_metadata[req.model_name] = {
            "file_path": model_file_path,
            "saved_at": datetime.now().isoformat(),
            "classes": req.model_data.get("classes", []),
            "features": list(req.model_data.get("likelihoods", {}).keys()),
            "metadata": req.metadata or {}
        }
        
        return {
            "message": f"Model '{req.model_name}' saved successfully",
            "file_path": model_file_path,
            "classes": req.model_data.get("classes", []),
            "features": list(req.model_data.get("likelihoods", {}).keys())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving model: {str(e)}")


@app.get("/models/{model_name}")
async def get_model(model_name: str):
    """Get a trained model."""
    try:
        # Check if model exists in metadata
        if model_name not in models_metadata:
            available_models = list(models_metadata.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {available_models}"
            )
        
        # Load model from file
        model_info = models_metadata[model_name]
        model_file_path = model_info["file_path"]
        
        if not os.path.exists(model_file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {model_file_path}"
            )
        
        with open(model_file_path, 'r') as f:
            model_data = json.load(f)
        
        return {
            "model_name": model_name,
            "model_data": model_data,
            "classes": model_data.get("classes", []),
            "features": list(model_data.get("likelihoods", {}).keys()),
            "metadata": {
                "saved_at": model_info["saved_at"],
                "file_path": model_file_path,
                "accuracy": model_info.get("metadata", {}).get("accuracy", None),
                **model_info.get("metadata", {})
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.get("/models")
async def list_models():
    """List all available models."""
    try:
        models_list = []
        for model_name, metadata in models_metadata.items():
            model_info = {
                "model_name": model_name,
                "classes": metadata.get("classes", []),
                "features": metadata.get("features", []),
                "saved_at": metadata.get("saved_at"),
                "metadata": {
                    "accuracy": metadata.get("metadata", {}).get("accuracy", None),
                    **metadata.get("metadata", {})
                }
            }
            models_list.append(model_info)
        
        return {
            "available_models": models_list,
            "count": len(models_list)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@app.get("/data")
async def list_datasets():
    """List all available datasets."""
    try:
        datasets_list = []
        for dataset_id, info in datasets_cache.items():
            if "train_data" in info:
                # Prepared dataset
                datasets_list.append({
                    "dataset_id": dataset_id,
                    "type": "prepared",
                    "train_shape": info["train_shape"],
                    "test_shape": info["test_shape"],
                    "target_column": info["target_column"],
                    "source": info["source"],
                    "prepared_at": info["prepared_at"]
                })
            else:
                # Raw dataset
                datasets_list.append({
                    "dataset_id": dataset_id,
                    "type": "raw",
                    "shape": info["shape"],
                    "columns": info["columns"],
                    "source": info["source"],
                    "loaded_at": info["loaded_at"]
                })
        
        return {
            "available_datasets": datasets_list,
            "count": len(datasets_list)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")


@app.get("/files")
async def list_data_files():
    """List CSV files in Data directory."""
    try:
        if not os.path.exists(DATA_DIR):
            return {"available_files": [], "count": 0}
        
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and not f.startswith('temp_')]
        
        files_info = []
        for file_name in csv_files:
            file_path = os.path.join(DATA_DIR, file_name)
            file_stats = os.stat(file_path)
            files_info.append({
                "file_name": file_name,
                "size_bytes": file_stats.st_size,
                "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            })
        
        return {
            "available_files": files_info,
            "count": len(files_info)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@app.delete("/data/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset from cache."""
    try:
        if dataset_id not in datasets_cache:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
        
        # Clean up temporary file if exists
        dataset_info = datasets_cache[dataset_id]
        if "temp_file" in dataset_info and os.path.exists(dataset_info["temp_file"]):
            os.remove(dataset_info["temp_file"])
        
        # Remove from cache
        del datasets_cache[dataset_id]
        
        return {"message": f"Dataset '{dataset_id}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {str(e)}")


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a model."""
    try:
        if model_name not in models_metadata:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Delete model file
        model_info = models_metadata[model_name]
        if os.path.exists(model_info["file_path"]):
            os.remove(model_info["file_path"])
        
        # Remove from metadata
        del models_metadata[model_name]
        
        return {"message": f"Model '{model_name}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "datasets_in_cache": len(datasets_cache),
        "models_in_storage": len(models_metadata),
        "data_directory": DATA_DIR,
        "models_directory": MODELS_DIR
    }
