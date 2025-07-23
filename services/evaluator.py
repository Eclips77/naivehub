from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from typing import Dict, Any

class NaiveBayesEvaluator:
    """Evaluator class for assessing Naive Bayes model performance.
    
    This class provides comprehensive evaluation metrics for classification models,
    including accuracy, precision, recall, and F1-score through scikit-learn's
    classification report functionality.
    """

    def __init__(self, predictor):
        """Initialize the evaluator with a trained predictor.
        
        Args:
            predictor: A trained NaiveBayesPredictor instance that can make predictions.
        """
        self.predictor = predictor

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
        """Evaluate the model performance on test data.
        
        Generates predictions for the test set and calculates various performance
        metrics including accuracy and detailed classification report.
        
        Args:
            X (pd.DataFrame): Test features dataset without the target column.
            y_true (pd.Series): True class labels for the test set.
            
        Returns:
            Dict[str, Any]: Dictionary containing evaluation results:
                - 'accuracy': Overall classification accuracy (float)
                - 'report': Detailed classification report with precision, recall, 
                           F1-score for each class (dict)
                - 'predictions': List of predicted class labels (list)
                
        Example:
            >>> evaluator = NaiveBayesEvaluator(predictor)
            >>> results = evaluator.evaluate(X_test, y_test)
            >>> print(f"Accuracy: {results['accuracy']:.3f}")
        """
        predictions = self.predictor.predict_batch(X)
        accuracy = accuracy_score(y_true, predictions)
        report_dict = classification_report(y_true, predictions, output_dict=True)

        return {
            "accuracy": accuracy,
            "report": report_dict,
            "predictions": predictions
        }
