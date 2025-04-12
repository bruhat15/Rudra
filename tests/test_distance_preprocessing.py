import pandas as pd
import numpy as np
import pytest
from rudra.executor import preprocess_distance_data

def test_empty_dataframe():
    """Test that empty DataFrame is handled correctly."""
    df = pd.DataFrame()
    result = preprocess_distance_data(df)
    assert result.empty

def test_complete_preprocessing_pipeline():
    """Test the complete preprocessing pipeline with mixed data types."""
    # Create test data with mixed types and missing values
    data = {
        'numeric1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'numeric2': [10.0, np.nan, 30.0, 40.0, 50.0],
        'category1': ['A', 'B', None, 'A', 'C'],
        'category2': ['X', 'Y', 'Z', None, 'X']
    }
    df = pd.DataFrame(data)
    
    # Process the data
    result = preprocess_distance_data(df)
    
    # Verify results
    assert not result.isnull().any().any()  # No null values
    assert result.shape[0] == df.shape[0]  # Same number of rows
    assert result.shape[1] == df.shape[1]  # Same number of columns
    assert result['numeric1'].dtype in ['float64', 'float32']
    assert result['numeric2'].dtype in ['float64', 'float32']
    
def test_excessive_nulls():
    """Test handling of rows and columns with excessive null values."""
    data = {
        'good_col': [1.0, 2.0, 3.0, 4.0],
        'bad_col': [np.nan, np.nan, np.nan, 4.0],  # 75% nulls
        'numeric': [1.0, np.nan, 3.0, 4.0]
    }
    df = pd.DataFrame(data)
    
    result = preprocess_distance_data(df)
    
    # bad_col should be dropped due to excessive nulls
    assert 'bad_col' not in result.columns
    assert 'good_col' in result.columns
    
def test_categorical_encoding():
    """Test categorical encoding in the pipeline."""
    data = {
        'category': ['A', 'B', 'A', 'C', 'B'],
        'numeric': [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    df = pd.DataFrame(data)
    
    result = preprocess_distance_data(df)
    
    # Check if categorical column was encoded to numeric
    assert result['category'].dtype in ['int64', 'float64']
    # Verify unique values are preserved
    assert len(result['category'].unique()) == len(df['category'].unique())
    
def test_numeric_scaling():
    """Test numeric feature scaling."""
    data = {
        'numeric': [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    df = pd.DataFrame(data)
    
    result = preprocess_distance_data(df)
    
    # Check if values are normalized (mean ≈ 0, std ≈ 1)
    assert abs(result['numeric'].mean()) < 1e-10  # Very close to 0
    assert abs(result['numeric'].std() - 1.0) < 1e-10  # Very close to 1 