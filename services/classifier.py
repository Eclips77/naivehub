from typing import Dict, Any
import pandas as pd
import math

class NaiveBayesPredictor:
    """Predicts using a trained Naive Bayes model."""

    def __init__(self, model: Dict[str, Any]):
        self.model = model
        self.classes = model["classes"]
        self.priors = model["priors"]
        self.likelihoods = model["likelihoods"]

    def predict(self, sample: Dict[str, Any]) -> Any:
        log_probs = {cls: math.log(self.priors[cls]) for cls in self.classes}

        for feature, value in sample.items():
            if feature in self.likelihoods and value in self.likelihoods[feature]:
                for cls in self.classes:
                    prob = self.likelihoods[feature][value].get(cls, 1e-6)
                    log_probs[cls] += math.log(prob)
            else:
                # Feature or value not seen in training
                continue

        return max(log_probs, key=log_probs.get)

    def predict_batch(self, df: pd.DataFrame) -> list:
        return [self.predict(row.to_dict()) for _, row in df.iterrows()]
