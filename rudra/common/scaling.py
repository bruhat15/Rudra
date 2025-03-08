import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def min_max_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Min-Max Scaling (0 to 1) to numerical columns."""
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes numerical columns (Z-score standardization)."""
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df