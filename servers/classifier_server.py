from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from services.classifier import NaiveBayesPredictor
from utils.model_loader import ModelLoader
from typing import Dict
import os

app = FastAPI()

TRAINER_URL = os.getenv("TRAINER_URL", "http://localhost:8001")
loaded_models: Dict[str, NaiveBayesPredictor] = {}

class PredictRequest(BaseModel):
    model_name: str
    record: dict



@app.get("/models")
async def list_models():
    try:
        resp = requests.get(f"{TRAINER_URL}/models")
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/{model_name}")
async def load_model(model_name: str):
    try:
        resp = requests.get(f"{TRAINER_URL}/model/{model_name}")
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        model_data = resp.json()
        loaded_models[model_name] = NaiveBayesPredictor(model_data)
        return {"message": f"Model '{model_name}' loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/all")
async def load_all_models():
    try:
        resp = requests.get(f"{TRAINER_URL}/model/all")
        models = resp.json()

        for name, model in models.items():
            loaded_models[name] = NaiveBayesPredictor(model)

        return {"message": f"{len(models)} models loaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        if req.model_name not in loaded_models:
            raise HTTPException(status_code=400, detail="Model not loaded.")

        predictor = loaded_models[req.model_name]
        result = predictor.predict(req.record)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
