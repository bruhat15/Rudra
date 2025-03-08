import pandas as pd

def drop_rows_with_excess_nulls(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Drops rows where the fraction of missing values is greater than the threshold.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Maximum allowed fraction of missing values per row.
        
    Returns:
        pd.DataFrame: DataFrame with rows dropped.
    """
    return df[df.isnull().mean(axis=1) <= threshold]

def drop_columns_with_excess_nulls(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Drops columns where the fraction of missing values is greater than the threshold.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Maximum allowed fraction of missing values per column.
        
    Returns:
        pd.DataFrame: DataFrame with columns dropped.
    """
    return df.loc[:, df.isnull().mean() <= threshold]

def impute_missing_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in numeric columns with the column mean.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with numeric missing values imputed.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    return df

def impute_missing_categorical(df: pd.DataFrame, placeholder: str = 'Missing') -> pd.DataFrame:
    """
    Imputes missing values in categorical columns with the mode.
    If mode is not available, uses a provided placeholder.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        placeholder (str): Value to fill when mode is not available.
        
    Returns:
        pd.DataFrame: DataFrame with categorical missing values imputed.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        mode = df[col].mode()
        if not mode.empty:
            df[col].fillna(mode[0], inplace=True)
        else:
            df[col].fillna(placeholder, inplace=True)
    return df
