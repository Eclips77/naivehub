import pandas as pd
from typing import Dict, Any

class NaiveBayesTrainer:
    """Trainer class for Naive Bayes classification algorithm.
    
    This class implements a Naive Bayes classifier that works with categorical features.
    It uses Laplace smoothing to handle unseen feature values and calculates prior
    probabilities and feature likelihoods for classification.
    """

    def __init__(self):
        """Initialize the NaiveBayesTrainer.
        
        The trainer starts with an empty model dictionary that will be populated
        during the training process.
        """
        self.model = {}

    def fit(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Train the Naive Bayes model on the provided dataset.

        This method calculates prior probabilities for each class and likelihood
        probabilities for each feature value given each class. Laplace smoothing
        is applied to handle zero probabilities.

        Args:
            df (pd.DataFrame): Training dataset containing features and target variable.
            target_column (str): Name of the column containing the target labels.

        Returns:
            Dict[str, Any]: Trained model containing:
                - 'classes': List of unique class labels
                - 'priors': Prior probability for each class
                - 'likelihoods': Conditional probabilities P(feature=value|class)

        Raises:
            KeyError: If the target_column is not found in the DataFrame.
            ValueError: If the DataFrame is empty or contains invalid data.
            
        Example:
            >>> trainer = NaiveBayesTrainer()
            >>> model = trainer.fit(train_df, 'target_column')
        """
        labels = df[target_column]
        features = df.drop(columns=[target_column])
        classes = labels.unique().tolist()
        priors = labels.value_counts(normalize=True).to_dict()

        likelihoods = {}

        for feature in features.columns:
            likelihoods[feature] = {}
            values = features[feature].unique()

            for value in values:
                likelihoods[feature][value] = {}
                for cls in classes:
                    # Laplace smoothing
                    count = len(df[(df[feature] == value) & (labels == cls)]) + 1
                    total = (labels == cls).sum() + len(values)
                    likelihoods[feature][value][cls] = count / total

        self.model = {
            "classes": classes,
            "priors": priors,
            "likelihoods": likelihoods
        }

        return self.model
