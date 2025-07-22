import pandas as pd
from typing import Tuple
from utils.data_cleaner import DataCleaner
from utils.data_spliter import DataSplitter
from utils.data_loader import DataLoader

class DataManager:
    """Handles full data preparation pipeline: load, clean, split."""

    def __init__(self, file_path: str):
        """
        Args:
            file_path (str): Path to the CSV file.
        """
        self.file_path = file_path
        self.df = None

    def load_data(self) -> None:
        """Load CSV data into a DataFrame with error handling."""
        try:
            self.df = DataLoader.load_data(self.file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")
        except pd.errors.ParserError:
            raise ValueError(f"Failed to parse CSV: {self.file_path}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while loading data: {e}")

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes full pipeline: load, clean, and split the data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.
        """
        self.load_data()

        cleaner = DataCleaner(self.df)
        cleaned_df = cleaner.clean_data()

        splitter = DataSplitter(cleaned_df)
        train_df, test_df = splitter.split_data()

        return train_df, test_df
