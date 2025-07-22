from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple

class DataSplitter:
    """Class for splitting data into train and test sets."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df (pd.DataFrame): The cleaned dataset.
        """
        self.df = df

    def split_data(self, train_size: float = 0.7, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataset into train and test sets.

        Args:
            train_size (float): Proportion of the data to include in the training set.
            random_state (int): Seed used by the random number generator.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test sets.
        """
        return train_test_split(self.df, train_size=train_size, random_state=random_state)
