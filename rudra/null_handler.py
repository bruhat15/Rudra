import pandas as pd

def handle_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> (pd.DataFrame, list):
    """
    Drops rows with more than threshold% nulls and imputes the rest.

    Args:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Fraction of allowed nulls in a row (default = 0.5).

    Returns:
        (pd.DataFrame, list): Tuple of cleaned DataFrame and summary list.
    """
    summary = []

    # Drop rows with > threshold% nulls
    row_thresh = int(df.shape[1] * threshold)
    df_cleaned = df.dropna(thresh=row_thresh)
    dropped_count = df.shape[0] - df_cleaned.shape[0]
    if dropped_count > 0:
        summary.append(f"Dropped {dropped_count} rows with more than {int(threshold * 100)}% missing values.")

    # Impute remaining nulls
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().any():
            if df_cleaned[col].dtype == 'object':
                mode_val = df_cleaned[col].mode().iloc[0]
                df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                summary.append(f"Filled nulls in categorical column '{col}' with mode: {mode_val}")
            else:
                mean_val = df_cleaned[col].mean()
                df_cleaned[col] = df_cleaned[col].fillna(mean_val)
                summary.append(f"Filled nulls in numeric column '{col}' with mean: {round(mean_val, 2)}")

    return df_cleaned, summary
