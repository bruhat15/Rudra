import pandas as pd

# --- Import common preprocessing utilities ---
from rudra.common.null_values import (
    drop_rows_with_excess_nulls,
    drop_columns_with_excess_nulls,
    impute_missing_numeric,
    impute_missing_categorical
)
from rudra.common.encoding import encode_features
from rudra.common.scaling import normalize  # Standard scaling

# --- Optional preprocessing steps ---
from rudra.outliers import remove_outliers_method  # Custom function to remove outliers
from rudra.encoding import encode_categoricals_method  # Custom function for encoding
from rudra.scaling import scale_numeric_features  # Custom function for scaling

# --------------------------------------------------
# ✅ Main Preprocessing Pipeline with Flags
# --------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    remove_outliers: bool = False,
    scale_data: bool = False,
    encode_categoricals: bool = False
):
    """
    Main preprocessing pipeline with optional flags for Streamlit app.

    Args:
        df (pd.DataFrame): Input dataset.
        remove_outliers (bool): Apply outlier removal if True.
        scale_data (bool): Apply scaling if True.
        encode_categoricals (bool): Apply encoding if True.

    Returns:
        pd.DataFrame: Processed dataset.
        List[str]: Summary of applied transformations.
    """
    summary = []

    # Handle missing values
    df = drop_rows_with_excess_nulls(df, threshold=0.5)
    df = drop_columns_with_excess_nulls(df, threshold=0.4)
    df = impute_missing_numeric(df)
    df = impute_missing_categorical(df)
    summary.append("Handled missing values")

    # Optional: Outlier removal
    if remove_outliers:
        df = remove_outliers_method(df)
        summary.append("Outliers removed")

    # Optional: Encoding
    if encode_categoricals:
        df = encode_categoricals_method(df)
        summary.append("Categorical variables encoded")

    # Optional: Scaling
    if scale_data:
        df = scale_numeric_features(df)
        summary.append("Numeric features scaled")

    return df, summary

# --------------------------------------------------
# ✅ Specialized Distance-Based Preprocessing (e.g., KNN)
# --------------------------------------------------

def preprocess_distance_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing strategy suitable for distance-based models (e.g., KNN, SVM).

    Steps:
    - Handle missing values
    - Encode categoricals using label encoding
    - Normalize numeric features

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    row_threshold = 0.5
    col_threshold = 0.4

    df = drop_rows_with_excess_nulls(df, row_threshold)
    df = drop_columns_with_excess_nulls(df, col_threshold)
    df = impute_missing_numeric(df)
    df = impute_missing_categorical(df)

    if len(df.select_dtypes(include=['object', 'category']).columns) > 0:
        df = encode_features(df)  # Label encoding for distance-based models

    if len(df.select_dtypes(include='number').columns) > 0:
        df = normalize(df)  # Standard scaling (Z-score)

    return df

# --------------------------------------------------
# ✅ Optional Testing
# --------------------------------------------------

if __name__ == "__main__":
    # Sample dataset for quick testing
    data = {
        "Age": [25, None, 35, 45, 50],
        "Salary": [50000, 60000, None, 80000, 90000],
        "Department": ["HR", "Finance", None, "HR", "IT"],
        "Experience": [2, 5, None, 8, 10]
    }
    df_sample = pd.DataFrame(data)

    print("🔹 Original DataFrame:")
    print(df_sample)

    # Run standard pipeline
    df_processed, summary = run_pipeline(df_sample, remove_outliers=True, scale_data=True, encode_categoricals=True)

    print("\n✅ Processed DataFrame:")
    print(df_processed)

    print("\n📋 Transformations Applied:")
    for step in summary:
        print("-", step)
