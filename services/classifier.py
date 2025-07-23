from typing import Dict, Any
import pandas as pd
import math

class NaiveBayesPredictor:
    """Prediction class for trained Naive Bayes models.
    
    This class uses a pre-trained Naive Bayes model to make predictions on new data.
    It implements the Naive Bayes prediction algorithm using log probabilities
    to avoid numerical underflow issues.
    """

    def __init__(self, model: Dict[str, Any]):
        """Initialize the predictor with a trained model.
        
        Args:
            model (Dict[str, Any]): A trained Naive Bayes model containing:
                - 'classes': List of class labels
                - 'priors': Prior probabilities for each class
                - 'likelihoods': Feature conditional probabilities
                
        Raises:
            KeyError: If the model is missing required keys.
        """
        self.model = model
        self.classes = model["classes"]
        self.priors = model["priors"]
        self.likelihoods = model["likelihoods"]

    def predict(self, sample: Dict[str, Any]) -> Any:
        """Predict the class label for a single sample.
        
        Uses the Naive Bayes algorithm with log probabilities to calculate
        the most likely class for the given sample.
        
        Args:
            sample (Dict[str, Any]): A dictionary containing feature names as keys
                and their corresponding values.
                
        Returns:
            Any: The predicted class label with the highest probability.
            
        Note:
            Features not seen during training are ignored. Unknown feature values
            are assigned a small probability (1e-6) to avoid zero probabilities.
            
        Example:
            >>> predictor = NaiveBayesPredictor(trained_model)
            >>> prediction = predictor.predict({"feature1": "value1", "feature2": "value2"})
        """
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
        """Predict class labels for multiple samples in a DataFrame.
        
        Applies the predict method to each row in the DataFrame and returns
        a list of predictions.
        
        Args:
            df (pd.DataFrame): DataFrame containing samples to predict.
                Each row represents one sample with features as columns.
                
        Returns:
            list: List of predicted class labels, one for each row in the DataFrame.
            
        Example:
            >>> predictor = NaiveBayesPredictor(trained_model)
            >>> predictions = predictor.predict_batch(test_df)
        """
        return [self.predict(row.to_dict()) for _, row in df.iterrows()]
