import json
from typing import Dict, Any

class ModelLoader:
    """Utility class for loading trained Naive Bayes models from JSON files.
    
    This class provides functionality to load pre-trained machine learning models
    that have been serialized to JSON format, with proper validation and error handling.
    """

    @staticmethod
    def load_model(path: str) -> Dict[str, Any]:
        """Load a trained Naive Bayes model from a JSON file.

        This method loads a model that contains the necessary components for
        Naive Bayes classification: classes, priors, and likelihoods.

        Args:
            path (str): Absolute or relative path to the model JSON file.

        Returns:
            Dict[str, Any]: A dictionary containing the loaded model with keys:
                - 'classes': List of class labels
                - 'priors': Prior probabilities for each class
                - 'likelihoods': Feature likelihoods for each class

        Raises:
            FileNotFoundError: If the specified model file does not exist.
            ValueError: If the file content is not valid JSON or missing required keys.
            RuntimeError: For any other unexpected errors during loading.
            
        Example:
            >>> model = ModelLoader.load_model("models/naive_bayes_model.json")
            >>> predictor = NaiveBayesPredictor(model)
        """
        try:
            with open(path, "r") as f:
                model = json.load(f)

            if not all(key in model for key in ("classes", "priors", "likelihoods")):
                raise ValueError("Invalid model structure.")

            return model

        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at path: {path}")

        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {path}")

        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
