import pandas as pd
import numpy as np
import pytest
from rudra.common.scaling import min_max_scale, normalize

def test_empty_dataframe():
    """Test that empty DataFrame is handled correctly."""
    df = pd.DataFrame()
    result_min_max = min_max_scale(df)
    result_normalize = normalize(df)
    assert result_min_max.empty
    assert result_normalize.empty

def test_no_numeric_columns():
    """Test that DataFrame with no numeric columns is returned unchanged."""
    data = {
        'categorical1': ['A', 'B', 'C', 'D', 'E'],
        'categorical2': ['X', 'Y', 'Z', 'X', 'Y']
    }
    df = pd.DataFrame(data)
    result_min_max = min_max_scale(df)
    result_normalize = normalize(df)
    assert result_min_max.equals(df)
    assert result_normalize.equals(df)

def test_min_max_scaling():
    """Test that min-max scaling is applied correctly."""
    data = {
        'numeric': [1, 2, 3, 4, 5],
        'categorical': ['A', 'B', 'C', 'D', 'E']
    }
    df = pd.DataFrame(data)
    result = min_max_scale(df)
    
    # Check that the numeric column was scaled to [0, 1]
    assert result['numeric'].min() == 0
    assert result['numeric'].max() == 1
    
    # Check that the categorical column was not changed
    assert result['categorical'].equals(df['categorical'])

def test_normalization():
    """Test that normalization (Z-score standardization) is applied correctly."""
    data = {
        'numeric': [1, 2, 3, 4, 5],
        'categorical': ['A', 'B', 'C', 'D', 'E']
    }
    df = pd.DataFrame(data)
    result = normalize(df)
    
    # Check that the numeric column was normalized (mean ≈ 0, std ≈ 1)
    assert abs(result['numeric'].mean()) < 1e-10  # Very close to 0
    assert abs(result['numeric'].std() - 1.0) < 1e-10  # Very close to 1
    
    # Check that the categorical column was not changed
    assert result['categorical'].equals(df['categorical'])

def test_multiple_numeric_columns():
    """Test that multiple numeric columns are scaled correctly."""
    data = {
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [10, 20, 30, 40, 50],
        'categorical': ['A', 'B', 'C', 'D', 'E']
    }
    df = pd.DataFrame(data)
    result_min_max = min_max_scale(df)
    result_normalize = normalize(df)
    
    # Check min-max scaling
    assert result_min_max['numeric1'].min() == 0
    assert result_min_max['numeric1'].max() == 1
    assert result_min_max['numeric2'].min() == 0
    assert result_min_max['numeric2'].max() == 1
    
    # Check normalization
    assert abs(result_normalize['numeric1'].mean()) < 1e-10
    assert abs(result_normalize['numeric1'].std() - 1.0) < 1e-10
    assert abs(result_normalize['numeric2'].mean()) < 1e-10
    assert abs(result_normalize['numeric2'].std() - 1.0) < 1e-10
    
    # Check that the categorical column was not changed
    assert result_min_max['categorical'].equals(df['categorical'])
    assert result_normalize['categorical'].equals(df['categorical'])

def test_missing_values():
    """Test that missing values are handled correctly."""
    data = {
        'numeric': [1, 2, np.nan, 4, 5],
        'categorical': ['A', 'B', 'C', 'D', 'E']
    }
    df = pd.DataFrame(data)
    result_min_max = min_max_scale(df)
    result_normalize = normalize(df)
    
    # Check that the numeric column was scaled correctly
    assert result_min_max['numeric'].min() == 0
    assert result_min_max['numeric'].max() == 1
    
    # Check that the numeric column was normalized correctly
    assert abs(result_normalize['numeric'].mean()) < 1e-10
    assert abs(result_normalize['numeric'].std() - 1.0) < 1e-10
    
    # Check that the categorical column was not changed
    assert result_min_max['categorical'].equals(df['categorical'])
    assert result_normalize['categorical'].equals(df['categorical']) 