import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler,LabelEncoder

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
from sklearn.preprocessing import LabelEncoder

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Encodes categorical columns using One-Hot Encoding for multi-class variables  
    and Label Encoding for binary categories."""
    # Find categorical columns in the DataFrame
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    one_hot_cols = []
    label_encode_cols = []
    # Categorize columns based on the number of unique values
    for col in cat_cols:
        if df[col].nunique() > 2:
            one_hot_cols.append(col)
        else:
            label_encode_cols.append(col)
    if one_hot_cols:
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)
    if label_encode_cols:
        for col in label_encode_cols:
            df[col] = LabelEncoder().fit_transform(df[col])
    
    return df
