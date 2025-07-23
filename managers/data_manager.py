import pandas as pd
from typing import Tuple
from utils.data_cleaner import DataCleaner
from utils.data_splitter import DataSplitter
from utils.data_loader import DataLoader

class DataManager:
    """Comprehensive data preparation pipeline manager.
    
    This class orchestrates the complete data preparation workflow including
    loading data from files, cleaning operations, and splitting into train/test sets.
    It provides a high-level interface for data preprocessing tasks.
    """

    def __init__(self, file_path: str):
        """Initialize the DataManager with a data file path.
        
        Args:
            file_path (str): Path to the CSV file containing the dataset.
            
        Attributes:
            file_path (str): Stored path to the data file.
            df (pd.DataFrame or None): Loaded DataFrame, initially None.
        """
        self.file_path = file_path
        self.df = None

    def load_data(self) -> None:
        """Load CSV data into a DataFrame with comprehensive error handling.
        
        Attempts to load the CSV file specified in the constructor and stores
        the result in self.df. Provides specific error messages for different
        failure scenarios.
        
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file cannot be parsed as a valid CSV.
            RuntimeError: For any other unexpected errors during loading.
            
        Example:
            >>> manager = DataManager("data/training_data.csv")
            >>> manager.load_data()
        """
        try:
            self.df = DataLoader.load_data(self.file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")
        except pd.errors.ParserError:
            raise ValueError(f"Failed to parse CSV: {self.file_path}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while loading data: {e}")

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the complete data preparation pipeline.
        
        Performs the full workflow: loads data, applies cleaning operations,
        and splits the cleaned data into training and testing sets.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - First element: Training DataFrame
                - Second element: Testing DataFrame
                
        Raises:
            Any exceptions from load_data(), DataCleaner, or DataSplitter operations.
            
        Example:
            >>> manager = DataManager("data/dataset.csv")
            >>> train_df, test_df = manager.prepare_data()
            >>> print(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")
        """
        self.load_data()

        if self.df is None:
            raise RuntimeError("Data loading failed - DataFrame is None")

        cleaner = DataCleaner(self.df)
        cleaned_df = cleaner.clean_data()

        splitter = DataSplitter(cleaned_df)
        train_df, test_df = splitter.split_data()

        return train_df, test_df
