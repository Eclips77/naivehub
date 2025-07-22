import pandas as pd
import json
from typing import Dict, Any
from services.trainer import NaiveBayesTrainer

class NaiveBayesTrainingManager:
    """Handles training and saving of a Naive Bayes model."""

    def __init__(self, df: pd.DataFrame, label_column: str, output_path: str = "model.json"):
        """
        Args:
            df (pd.DataFrame): Training dataset.
            label_column (str): Name of the label column.
            output_path (str): Path to save the trained model JSON.
        """
        self.df = df
        self.label_column = label_column
        self.output_path = output_path

    def train_and_save(self) -> Dict[str, Any]:
        """Train the model and save it to a JSON file.

        Returns:
            Dict[str, Any]: Trained model.

        Raises:
            ValueError: If dataset or label column is invalid.
            IOError: If saving the model fails.
        """
        if self.df.empty:
            raise ValueError("Dataset is empty.")

        if self.label_column not in self.df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in dataset.")

        trainer = NaiveBayesTrainer()
        model = trainer.fit(self.df, self.label_column)

        try:
            with open(self.output_path, "w") as f:
                json.dump(model, f)
        except Exception as e:
            raise IOError(f"Failed to save model to {self.output_path}: {e}")

        return model
