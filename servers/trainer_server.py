from managers.data_manager import DataManager
from managers.trainer_manager import NaiveBayesTrainingManager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()
data_service = DataManager("./Data")

@app.on_event("startup")
async def startup_event():
    """Load data, set the target column and train the model on startup."""
    file_name = os.getenv("Data", "play_tennis.csv")
    try:
        # Load and clean the dataset
        data_service.load_data()
        # Use the last column as the target by default
        target = data_app.train_df.columns[-1]
        data_app.set_target_column(target)
        # Train the model so the API is ready to serve predictions
        data_app.train_model()
    except Exception as e:
        # Log any issue during startup but allow the server to keep running
        print(f"Failed to initialise model: {e}")