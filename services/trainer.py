import pandas as pd
from typing import Dict, Any

class NaiveBayesTrainer:
    """Trains a Naive Bayes model on categorical features."""

    def __init__(self):
        self.model = {}

    def fit(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
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
