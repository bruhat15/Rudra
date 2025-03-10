"""
tree_based_preprocessing.py

This module implements preprocessing optimized for tree-based models
(e.g., Decision Trees, Gradient Boosting, XGBoost). It covers:
  1. Missing value handling
  2. Categorical variable encoding with special treatment for high-cardinality features
  3. (Skipping scaling/outlier handling as tree-based models are robust)
  4. Feature selection via correlation filtering
  5. Data balancing (placeholder for further integration)

Each function is modular so that common functions (if they can be shared) can also be placed in the common folder.
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
# Optionally, if you want to integrate any functions from your common modules,
# you might import them like:
# from rudra.common.null_values import impute_missing_numeric, impute_missing_categorical
# from rudra.common.encoding import encode_features

def rare_binning(series: pd.Series, threshold: float = 0.05) -> pd.Series:
    """
    Groups categories that occur less frequently than the threshold into 'Other'.
    
    Parameters:
        series (pd.Series): Categorical column.
        threshold (float): Relative frequency threshold.
    
    Returns:
        pd.Series: Series with rare categories binned as 'Other'.
    """
    freq = series.value_counts(normalize=True)
    rare_categories = freq[freq < threshold].index
    return series.apply(lambda x: "Other" if x in rare_categories else x)

def handle_missing_values_tree(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    numeric_placeholder: float = -999,
    categorical_placeholder: str = "Missing"
) -> pd.DataFrame:
    """
    Handles missing values with strategies optimized for tree-based models.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        numeric_strategy (str): "median" or "placeholder" for numeric imputation.
        categorical_strategy (str): "mode" or "placeholder" for categorical imputation.
        numeric_placeholder (float): Placeholder value for numeric columns.
        categorical_placeholder (str): Placeholder for categorical columns.
    
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(include=["object", "category"]).columns
    
    # Numeric columns
    for col in numeric_cols:
        if numeric_strategy == "median":
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        elif numeric_strategy == "placeholder":
            df_processed[col].fillna(numeric_placeholder, inplace=True)
        else:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Categorical columns
    for col in categorical_cols:
        if categorical_strategy == "mode":
            mode = df_processed[col].mode()
            if not mode.empty:
                df_processed[col].fillna(mode[0], inplace=True)
            else:
                df_processed[col].fillna(categorical_placeholder, inplace=True)
        elif categorical_strategy == "placeholder":
            df_processed[col].fillna(categorical_placeholder, inplace=True)
        else:
            mode = df_processed[col].mode()
            if not mode.empty:
                df_processed[col].fillna(mode[0], inplace=True)
            else:
                df_processed[col].fillna(categorical_placeholder, inplace=True)
    
    return df_processed

def handle_categorical_variables_tree(
    df: pd.DataFrame,
    target: str = None,
    high_card_threshold: int = 10
) -> pd.DataFrame:
    """
    Handles encoding of categorical variables for tree-based models.
    For high-cardinality features, applies rare binning and, if a target is provided,
    a simple target encoding; otherwise, applies label encoding.
    For low-cardinality features, uses one-hot encoding for >2 unique values and label encoding otherwise.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target (str): Optional target variable for target encoding.
        high_card_threshold (int): Threshold to determine high cardinality.
    
    Returns:
        pd.DataFrame: DataFrame with encoded categorical features.
    """
    df_processed = df.copy()
    categorical_cols = df_processed.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        unique_vals = df_processed[col].nunique()
        if unique_vals > high_card_threshold:
            # For high-cardinality features: apply rare binning
            df_processed[col] = rare_binning(df_processed[col], threshold=0.05)
            if target is not None and target in df_processed.columns:
                # Simple target (mean) encoding; note: use with caution to avoid leakage.
                target_means = df_processed.groupby(col)[target].mean()
                df_processed[col] = df_processed[col].map(target_means)
            else:
                # If target not provided, use label encoding after rare binning.
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
        else:
            # Low-cardinality: one-hot encode if more than 2 unique values
            if unique_vals > 2:
                dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                df_processed = pd.concat([df_processed.drop(columns=[col]), dummies], axis=1)
            else:
                # Binary categorical: use label encoding
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                
    return df_processed

def select_features_tree(df: pd.DataFrame, corr_threshold: float = 0.9) -> pd.DataFrame:
    """
    Removes highly correlated features using a correlation threshold.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        corr_threshold (float): Threshold above which one feature in a pair is dropped.
    
    Returns:
        pd.DataFrame: DataFrame with less correlated features.
    """
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    corr_matrix = df_processed[numeric_cols].corr().abs()
    
    # Use upper triangle matrix to identify redundant features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    
    df_processed.drop(columns=to_drop, inplace=True)
    return df_processed

def balance_data_tree(df: pd.DataFrame, target: str, method: str = "class_weight") -> pd.DataFrame:
    """
    Placeholder for handling class imbalance. For tree-based models, setting class weights
    is typically the recommended approach. If oversampling/undersampling is needed (e.g., for boosting models),
    you could integrate SMOTE or similar techniques.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target (str): Target variable name.
        method (str): Method for balancing (default: "class_weight").
    
    Returns:
        pd.DataFrame: DataFrame (balanced if oversampling/undersampling applied).
    """
    # In this example, we assume that handling imbalance will be done in the model (e.g., using class_weight)
    # If needed, integrate oversampling/undersampling here.
    return df

def tree_based_preprocess(
    df: pd.DataFrame,
    target: str = None,
    numeric_missing_strategy: str = "median",
    categorical_missing_strategy: str = "mode",
    high_card_threshold: int = 10,
    corr_threshold: float = 0.9,
    balance_method: str = "class_weight"
) -> pd.DataFrame:
    """
    Executes the complete tree-based preprocessing pipeline:
      1. Missing value imputation.
      2. Categorical variable encoding with special handling for high-cardinality features.
      3. (Skips scaling and outlier removal as tree-based models are robust.)
      4. Feature selection via correlation filtering.
      5. Optional data balancing.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target (str): Optional target column name (used in target encoding and balancing).
        numeric_missing_strategy (str): Strategy for numeric imputation ("median" or "placeholder").
        categorical_missing_strategy (str): Strategy for categorical imputation ("mode" or "placeholder").
        high_card_threshold (int): Threshold for high cardinality in categorical features.
        corr_threshold (float): Correlation threshold to drop highly correlated features.
        balance_method (str): Method to handle imbalance ("class_weight" by default).
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df_processed = df.copy()
    
    try:
        # 1. Handle missing values
        df_processed = handle_missing_values_tree(
            df_processed,
            numeric_strategy=numeric_missing_strategy,
            categorical_strategy=categorical_missing_strategy
        )
    except Exception as e:
        print("Error in missing value handling:", e)
        raise

    try:
        # 2. Handle categorical variables (encoding and high-cardinality treatment)
        df_processed = handle_categorical_variables_tree(
            df_processed,
            target=target,
            high_card_threshold=high_card_threshold
        )
    except Exception as e:
        print("Error in categorical variable handling:", e)
        raise

    # 3. Skip scaling and outlier removal for tree-based models (they are generally robust)
    
    try:
        # 4. Feature selection: remove highly correlated features
        df_processed = select_features_tree(df_processed, corr_threshold=corr_threshold)
    except Exception as e:
        print("Error in feature selection:", e)
        raise

    try:
        # 5. Data balancing (if target is provided)
        if target is not None and target in df_processed.columns:
            df_processed = balance_data_tree(df_processed, target=target, method=balance_method)
    except Exception as e:
        print("Error in data balancing:", e)
        raise

    return df_processed

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame to demonstrate the pipeline
    data = {
        'num_feature': [1, 2, None, 4, 5, 1000],
        'cat_feature': ['A', 'B', 'A', None, 'C', 'A'],
        'high_card_feature': ['id1', 'id2', 'id1', 'id3', 'id4', 'id1'],
        'target': [0, 1, 0, 1, 0, 1]
    }
    df_sample = pd.DataFrame(data)
    processed_df = tree_based_preprocess(df_sample, target='target')
    print("Processed DataFrame:\n", processed_df.head())
