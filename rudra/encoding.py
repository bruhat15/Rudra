# rudra/encoding.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categoricals_method(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical variables using Label Encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe with categorical columns.
    
    Returns:
        pd.DataFrame: Dataframe with categorical columns encoded as numeric.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    label_encoder = LabelEncoder()
    
    # Apply Label Encoding to each categorical column
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col].astype(str))  # Convert to string and apply encoding
    
    return df
