import pandas as pd
import numpy as np
import pytest
from rudra.common.null_values import (
    drop_rows_with_excess_nulls,
    drop_columns_with_excess_nulls,
    impute_missing_numeric,
    impute_missing_categorical
)

def test_empty_dataframe():
    """Test that empty DataFrame is handled correctly."""
    df = pd.DataFrame()
    result_rows = drop_rows_with_excess_nulls(df, 0.5)
    result_cols = drop_columns_with_excess_nulls(df, 0.5)
    result_numeric = impute_missing_numeric(df)
    result_categorical = impute_missing_categorical(df)
    assert result_rows.empty
    assert result_cols.empty
    assert result_numeric.empty
    assert result_categorical.empty

def test_drop_rows_with_excess_nulls():
    """Test dropping rows with excess null values."""
    data = {
        'col1': [1, 2, np.nan, 4, 5],
        'col2': [10, np.nan, 30, 40, 50],
        'col3': [100, 200, 300, np.nan, 500]
    }
    df = pd.DataFrame(data)
    
    # Drop rows with more than 50% null values
    result = drop_rows_with_excess_nulls(df, 0.5)
    
    # Check that rows with more than 50% null values were dropped
    assert len(result) < len(df)
    
    # Check that the remaining rows have at most 50% null values
    for _, row in result.iterrows():
        assert row.isnull().mean() <= 0.5

def test_drop_columns_with_excess_nulls():
    """Test dropping columns with excess null values."""
    data = {
        'col1': [1, 2, 3, 4, 5],
        'col2': [10, np.nan, 30, 40, 50],
        'col3': [100, np.nan, np.nan, np.nan, np.nan]  # 80% nulls
    }
    df = pd.DataFrame(data)
    
    # Drop columns with more than 50% null values
    result = drop_columns_with_excess_nulls(df, 0.5)
    
    # Check that columns with more than 50% null values were dropped
    assert 'col3' not in result.columns
    assert 'col1' in result.columns
    assert 'col2' in result.columns

def test_impute_missing_numeric():
    """Test imputing missing values in numeric columns."""
    data = {
        'numeric1': [1, 2, np.nan, 4, 5],
        'numeric2': [10, np.nan, 30, 40, 50],
        'categorical': ['A', 'B', 'C', 'D', 'E']
    }
    df = pd.DataFrame(data)
    
    # Impute missing values in numeric columns
    result = impute_missing_numeric(df)
    
    # Check that numeric missing values were imputed
    assert not result['numeric1'].isnull().any()
    assert not result['numeric2'].isnull().any()
    
    # Check that categorical column was not changed
    assert result['categorical'].equals(df['categorical'])

def test_impute_missing_categorical():
    """Test imputing missing values in categorical columns."""
    data = {
        'numeric': [1, 2, 3, 4, 5],
        'categorical1': ['A', 'B', np.nan, 'D', 'E'],
        'categorical2': ['X', 'Y', 'Z', np.nan, 'X']
    }
    df = pd.DataFrame(data)
    
    # Impute missing values in categorical columns
    result = impute_missing_categorical(df)
    
    # Check that categorical missing values were imputed
    assert not result['categorical1'].isnull().any()
    assert not result['categorical2'].isnull().any()
    
    # Check that numeric column was not changed
    assert result['numeric'].equals(df['numeric'])

def test_impute_missing_categorical_with_placeholder():
    """Test imputing missing values in categorical columns with a custom placeholder."""
    data = {
        'numeric': [1, 2, 3, 4, 5],
        'categorical': [np.nan, np.nan, np.nan, np.nan, np.nan]  # All nulls
    }
    df = pd.DataFrame(data)
    
    # Impute missing values in categorical columns with a custom placeholder
    result = impute_missing_categorical(df, placeholder='Custom')
    
    # Check that categorical missing values were imputed with the custom placeholder
    assert not result['categorical'].isnull().any()
    assert (result['categorical'] == 'Custom').all()
    
    # Check that numeric column was not changed
    assert result['numeric'].equals(df['numeric'])

def test_complete_null_handling_pipeline():
    """Test the complete null handling pipeline."""
    data = {
        'col1': [1, 2, np.nan, 4, 5],
        'col2': [10, np.nan, np.nan, 40, 50],  # 60% nulls
        'col3': [100, 200, 300, np.nan, 500],
        'cat1': ['A', 'B', np.nan, 'D', 'E'],
        'cat2': ['X', 'Y', 'Z', np.nan, 'X']
    }
    df = pd.DataFrame(data)
    
    # Step 1: Drop columns with excess nulls
    df = drop_columns_with_excess_nulls(df, 0.5)
    assert 'col2' not in df.columns
    
    # Step 2: Drop rows with excess nulls
    df = drop_rows_with_excess_nulls(df, 0.5)
    
    # Step 3: Impute missing values
    df = impute_missing_numeric(df)
    df = impute_missing_categorical(df)
    
    # Check that all missing values were handled
    assert not df.isnull().any().any() 