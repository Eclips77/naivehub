from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple

class DataSplitter:
    """Class for splitting data into train and test sets.
    
    This class provides functionality to split datasets into training and testing
    portions using various strategies and configurations.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize the DataSplitter with a DataFrame.
        
        Args:
            df (pd.DataFrame): The cleaned dataset to be split.
        """
        self.df = df

    def split_data(self, train_size: float = 0.7, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataset into train and test sets.

        Args:
            train_size (float, optional): Proportion of the data to include in the 
                training set. Must be between 0.0 and 1.0. Defaults to 0.7.
            random_state (int, optional): Seed used by the random number generator 
                for reproducible results. Defaults to 42.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training set
                as the first element and the test set as the second element.
                
        Raises:
            ValueError: If train_size is not between 0.0 and 1.0.
        """
        if not 0.0 < train_size < 1.0:
            raise ValueError("train_size must be between 0.0 and 1.0")
            
        return train_test_split(self.df, train_size=train_size, random_state=random_state)
