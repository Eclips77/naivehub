import pandas as pd

class DataCleaner:
    """Class for cleaning data."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df (pd.DataFrame): The raw dataset.
        """
        self.df = df

    def clean_data(self) -> pd.DataFrame:
        """Remove rows with missing values.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        self.df.dropna(inplace=True)
        self.df.drop_duplicates()
        return self.df
