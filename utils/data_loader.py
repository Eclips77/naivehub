import pandas as pd

class DataLoader:
    """Helpers for loading various data formats."""

    @staticmethod
    def load_data(file_path):
        """Load a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pandas.DataFrame: Loaded DataFrame.

        Usage:
            df = DataLoader.load_data("data.csv")
        """
        return pd.read_csv(file_path)

    @staticmethod
    def load_json(file_path):
        """Load a JSON file into a DataFrame."""
        return pd.read_json(file_path)

    @staticmethod
    def load_excel(file_path):
        """Load an Excel file into a DataFrame."""
        return pd.read_excel(file_path)

    @staticmethod
    def load_from_df(df: pd.DataFrame):
        """Return the provided DataFrame as is."""
        return df

