import pandas as pd
import json
from typing import Dict, Any
from services.trainer import NaiveBayesTrainer

class NaiveBayesTrainingManager:
    """High-level manager for Naive Bayes model training and persistence.
    
    This class provides a complete workflow for training Naive Bayes models
    and saving them to JSON files for later use. It handles model training
    orchestration and file I/O operations with comprehensive error handling.
    """

    def __init__(self, df: pd.DataFrame, label_column: str, output_path: str = "model.json"):
        """Initialize the training manager with dataset and configuration.
        
        Args:
            df (pd.DataFrame): Training dataset containing features and target variable.
            label_column (str): Name of the column containing class labels.
            output_path (str, optional): File path where the trained model will be saved.
                Defaults to "model.json".
                
        Attributes:
            df (pd.DataFrame): Stored training dataset.
            label_column (str): Target column name for classification.
            output_path (str): Destination path for model serialization.
        """
        self.df = df
        self.label_column = label_column
        self.output_path = output_path

    def train_and_save(self) -> Dict[str, Any]:
        """Train a Naive Bayes model and save it to a JSON file.
        
        Performs model training using the configured dataset and label column,
        then serializes the trained model to the specified output path.
        
        Returns:
            Dict[str, Any]: The trained model dictionary containing classes,
                priors, and likelihoods that was saved to file.

        Raises:
            ValueError: If the dataset is empty or the label column is not found.
            IOError: If saving the model to file fails due to permission or disk issues.
            
        Example:
            >>> manager = NaiveBayesTrainingManager(train_df, "target", "my_model.json")
            >>> model = manager.train_and_save()
            >>> print("Model trained and saved successfully")
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
