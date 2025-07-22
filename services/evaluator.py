from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from typing import Dict, Any

class NaiveBayesEvaluator:
    """Evaluates a Naive Bayes model and returns structured results."""

    def __init__(self, predictor):
        self.predictor = predictor

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
        predictions = self.predictor.predict_batch(X)
        accuracy = accuracy_score(y_true, predictions)
        report_dict = classification_report(y_true, predictions, output_dict=True)

        return {
            "accuracy": accuracy,
            "report": report_dict,
            "predictions": predictions
        }
