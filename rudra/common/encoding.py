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