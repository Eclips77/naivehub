import pandas as pd

class DataLoader:
    """Utility class for loading data from various file formats.
    
    This class provides static methods to load data from different sources
    such as CSV, JSON, Excel files, and existing DataFrames.
    """

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Load data from a CSV file.

        Args:
            file_path (str): Path to the CSV file to be loaded.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            pd.errors.ParserError: If the file cannot be parsed as CSV.
            
        Example:
            >>> df = DataLoader.load_data("data/sample.csv")
        """
        return pd.read_csv(file_path)

    @staticmethod
    def load_json(file_path: str) -> pd.DataFrame:
        """Load data from a JSON file into a DataFrame.
        
        Args:
            file_path (str): Path to the JSON file to be loaded.
            
        Returns:
            pd.DataFrame: DataFrame containing the loaded JSON data.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the JSON file cannot be parsed or converted to DataFrame.
        """
        return pd.read_json(file_path)

    @staticmethod
    def load_excel(file_path: str) -> pd.DataFrame:
        """Load data from an Excel file into a DataFrame.
        
        Args:
            file_path (str): Path to the Excel file to be loaded.
            
        Returns:
            pd.DataFrame: DataFrame containing the loaded Excel data.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the Excel file cannot be parsed or read.
        """
        return pd.read_excel(file_path)

    @staticmethod
    def load_from_df(df: pd.DataFrame) -> pd.DataFrame:
        """Return the provided DataFrame as is.
        
        This method is useful for maintaining consistency in data loading
        pipelines when the data is already in DataFrame format.
        
        Args:
            df (pd.DataFrame): The DataFrame to be returned.
            
        Returns:
            pd.DataFrame: The same DataFrame that was passed as input.
        """
        return df

