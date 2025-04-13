
import pandas as pd # rudra/scaling.py
from sklearn.preprocessing import StandardScaler

def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales numeric features using Standard Scaling (Z-score normalization).
    
    Args:
        df (pd.DataFrame): Input dataframe with numeric columns.
    
    Returns:
        pd.DataFrame: Dataframe with numeric features scaled.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])  # Scale numeric features
    
    return df
