from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from managers.data_manager import DataManager
from managers.trainer_manager import NaiveBayesTrainingManager
import os
import pandas as pd

app = FastAPI()


class TrainRequest(BaseModel):
    file_path: str
    target_column: str
    model_name: str = "model.json"



@app.post("/train")
async def train_model(req: TrainRequest):
    """
    Train and save a Naive Bayes model from given CSV file and label column.

    Body:
        file_path (str): Path to the CSV file.
        target_column (str): Name of the target label column.
        model_name (str): Optional filename for saving the model.

    Returns:
        dict: Training status.
    """
    try:
        # Load and clean data
        data_manager = DataManager(req.file_path)
        train_df, _ = data_manager.prepare_data()

        # Check that label exists
        if req.target_column not in train_df.columns:
            raise ValueError(f"Target column '{req.target_column}' not found in data.")

        # Train and save
        trainer = NaiveBayesTrainingManager(train_df, req.target_column, output_path=req.model_name)
        trainer.train_and_save()

        return {"message": f"Model trained and saved as '{req.model_name}'"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models")
async def return_model():
    pass