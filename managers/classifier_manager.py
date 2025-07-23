import pandas as pd
from typing import Dict, Any, List
from services.classifier import NaiveBayesPredictor
from services.evaluator import NaiveBayesEvaluator
from utils.model_loader import ModelLoader


class NaiveBayesClassificationManager:
    """High-level manager for Naive Bayes classification operations.
    
    This class provides a complete workflow for loading trained models,
    making predictions, and evaluating model performance. It serves as
    a facade that simplifies classification tasks.
    """

    def __init__(self, model_path: str = None, model_dict: Dict[str, Any] = None):
        """Initialize the classification manager.
        
        Args:
            model_path (str, optional): Path to a saved model JSON file.
            model_dict (Dict[str, Any], optional): Pre-loaded model dictionary.
            
        Note:
            Either model_path or model_dict must be provided, but not both.
            
        Raises:
            ValueError: If neither or both parameters are provided.
        """
        if (model_path is None) == (model_dict is None):
            raise ValueError("Provide either model_path or model_dict, but not both")
            
        if model_path:
            self.model = ModelLoader.load_model(model_path)
        else:
            self.model = model_dict
            
        self.predictor = NaiveBayesPredictor(self.model)
        self.evaluator = NaiveBayesEvaluator(self.predictor)

    def predict_single(self, sample: Dict[str, Any]) -> Any:
        """Make a prediction for a single sample.
        
        Args:
            sample (Dict[str, Any]): Feature dictionary for a single sample.
            
        Returns:
            Any: Predicted class label.
            
        Example:
            >>> manager = NaiveBayesClassificationManager(model_path="model.json")
            >>> prediction = manager.predict_single({"feature1": "value1", "feature2": "value2"})
        """
        return self.predictor.predict(sample)

    def predict_batch(self, df: pd.DataFrame) -> List[Any]:
        """Make predictions for multiple samples.
        
        Args:
            df (pd.DataFrame): DataFrame containing samples to predict.
            
        Returns:
            List[Any]: List of predicted class labels.
            
        Example:
            >>> predictions = manager.predict_batch(test_features_df)
        """
        return self.predictor.predict_batch(df)

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance on test data.
        
        Args:
            X_test (pd.DataFrame): Test features dataset.
            y_test (pd.Series): True class labels for test set.
            
        Returns:
            Dict[str, Any]: Evaluation results including accuracy and detailed report.
            
        Example:
            >>> results = manager.evaluate_model(X_test, y_test)
            >>> print(f"Model accuracy: {results['accuracy']:.3f}")
        """
        return self.evaluator.evaluate(X_test, y_test)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information including classes and structure.
        """
        return {
            "classes": self.model.get("classes", []),
            "num_classes": len(self.model.get("classes", [])),
            "features": list(self.model.get("likelihoods", {}).keys()),
            "num_features": len(self.model.get("likelihoods", {}))
        }
