import pandas as pd
from rudra.common.null_values import (
    drop_rows_with_excess_nulls,
    drop_columns_with_excess_nulls,
    impute_missing_numeric,
    impute_missing_categorical
)
from rudra.common.encoding import encode_features
from rudra.common.scaling import normalize  # Using normalization from scaling.py


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



import pandas as pd
from encoding import encode_features  # Function from encoding.py
from null_values import (drop_rows_with_excess_nulls, drop_columns_with_excess_nulls,
                         impute_missing_numeric, impute_missing_categorical)  # Functions from null_values.py
from outlier_detection import detect_and_handle_outliers  # Function from outlier_detection.py
from scaling import min_max_scale, normalize  # Functions from scaling.py

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
    - Scaling the features

    Parameters:
    - file_path (str): Path to the CSV file.
    - null_threshold_row (float): Threshold for dropping rows with excess nulls (0-1).
    - null_threshold_col (float): Threshold for dropping columns with excess nulls (0-1).
    - outlier_method (str): Method for outlier detection ('iqr', 'zscore', or 'isolation_forest').
    - outlier_handling (str): How to handle outliers ('remove', 'mean', or 'median').
    - scale_method (str): Method for scaling ('standard' for Z-score normalization or 'minmax' for Min-Max scaling).

    Returns:
    - pd.DataFrame: Preprocessed DataFrame ready for modeling.
    """
    # Step 1: Load dataset
    df = pd.read_csv(file_path)

    # Step 2: Handle missing values
    df = drop_rows_with_excess_nulls(df, threshold=null_threshold_row)  # Drop rows with too many nulls
    df = drop_columns_with_excess_nulls(df, threshold=null_threshold_col)  # Drop columns with too many nulls
    df = impute_missing_numeric(df)  # Impute numeric columns with mean
    df = impute_missing_categorical(df)  # Impute categorical columns with mode

    # Step 3: Detect and handle outliers
    # You may call detect_and_handle_outliers for outlier detection
    # Since it's saving a file, we'll skip calling it here. You can also modify the function to return the processed dataframe.
    detect_and_handle_outliers(method=outlier_method, handling=outlier_handling, save_path='outlier_free_dataset.csv')

    # Step 4: Encode categorical features
    df = encode_features(df)  # Encode categorical columns using One-Hot and Label Encoding

    # Step 5: Scale features (based on chosen scaling method)
    if scale_method == 'minmax':
        df = min_max_scale(df)  # Apply Min-Max scaling
    elif scale_method == 'standard':
        df = normalize(df)  # Apply Z-score normalization
    else:
        raise ValueError("Invalid scale method. Choose 'standard' or 'minmax'.")

    return df

