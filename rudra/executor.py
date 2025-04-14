import pandas as pd
from rudra.common.null_values import (
    drop_rows_with_excess_nulls,
    drop_columns_with_excess_nulls,
    impute_missing_numeric,
    impute_missing_categorical
)
from rudra.common.encoding import encode_features
from rudra.common.scaling import normalize
from rudra.common.outlier_detection import detect_and_handle_outliers

def preprocess_distance_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the DataFrame for distance-based models (e.g., KNN, SVM)
    using a strategic, conditional pipeline.
    
    The pipeline includes:
    1. Row and Column Null Filtering:
       - Drop rows with more than row_threshold missing values.
       - Drop columns with more than col_threshold missing values.
       
    2. Missing Value Handling:
       - Impute numeric columns with mean.
       - Impute categorical columns with mode.
       
    3. Categorical Encoding:
       - Use label encoding for categorical features (via encode_features).
       
    4. Feature Scaling:
       - Normalize the numeric features using standard scaling.
       
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    # Handle empty DataFrame
    if df.empty:
        return df
        
    # --- Step 1: Filter out rows/columns with excessive nulls ---
    row_threshold = 0.5  # Drop row if > 50% values are missing
    col_threshold = 0.4  # Drop column if > 40% values are missing
    
    df = drop_rows_with_excess_nulls(df, row_threshold)
    df = drop_columns_with_excess_nulls(df, col_threshold)
    
    # --- Step 2: Impute remaining missing values ---
    df = impute_missing_numeric(df)
    df = impute_missing_categorical(df)
    
    # --- Step 3: Identify data types ---
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # --- Step 4: Categorical Encoding ---
    if len(categorical_cols) > 0:
        # For distance-based models, label encoding is preferred.
        df = encode_features(df)
    
    # --- Step 5: Feature Scaling ---
    if len(numerical_cols) > 0:
        # Normalize numeric features so that distances are not biased.
        df = normalize(df)
    
    return df

def preprocess_data(file_path: str,
                    null_threshold_row: float = 0.5,
                    null_threshold_col: float = 0.5,
                    outlier_method: str = 'iqr',
                    outlier_handling: str = 'median',
                    scale_method: str = 'standard') -> pd.DataFrame:
    """
    Preprocesses the CSV dataset for linear regression, including:
    - Handling missing values
    - Detecting and handling outliers
    - Encoding categorical variables
    - Scaling numerical features
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file to preprocess
    null_threshold_row : float, optional
        Threshold for dropping rows with too many null values, by default 0.5
    null_threshold_col : float, optional
        Threshold for dropping columns with too many null values, by default 0.5
    outlier_method : str, optional
        Method to detect outliers ('iqr', 'zscore', 'isolation_forest'), by default 'iqr'
    outlier_handling : str, optional
        How to handle outliers ('remove', 'mean', 'median'), by default 'median'
    scale_method : str, optional
        Method to scale numerical features ('minmax', 'standard'), by default 'standard'
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df = drop_rows_with_excess_nulls(df, null_threshold_row)
    df = drop_columns_with_excess_nulls(df, null_threshold_col)
    df = impute_missing_numeric(df)
    df = impute_missing_categorical(df)
    
    # Detect and handle outliers
    df = detect_and_handle_outliers(df, method=outlier_method, handling=outlier_handling)
    
    # Encode categorical variables
    df = encode_features(df)
    
    # Scale numerical features
    if scale_method == 'minmax':
        from rudra.common.scaling import min_max_scale
        df = min_max_scale(df)
    else:  # standard scaling
        df = normalize(df)
    
    return df

if __name__ == "__main__":
    # --- Test the Distance-Based Preprocessing Pipeline ---
    data = {
        "Age": [25, None, 35, 45, 50],
        "Salary": [50000, 60000, None, 80000, 90000],
        "Department": ["HR", "Finance", None, "HR", "IT"],
        "Experience": [2, 5, None, 8, 10]
    }
    df_test = pd.DataFrame(data)
    
    print("🔹 Original DataFrame:")
    print(df_test)
    
    processed_df = preprocess_distance_data(df_test)
    
    print("\n✅ Processed DataFrame for Distance-Based Models:")
    print(processed_df)

