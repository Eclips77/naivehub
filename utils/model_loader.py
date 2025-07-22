import json
from typing import Dict, Any

class ModelLoader:
    """Loads a trained Naive Bayes model from a JSON file."""

    @staticmethod
    def load_model(path: str) -> Dict[str, Any]:
        """
        Load a model from a JSON file.

        Args:
            path (str): Path to the model JSON file.

        Returns:
            Dict[str, Any]: The loaded model dictionary.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file content is invalid.
            RuntimeError: For any other errors.
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
