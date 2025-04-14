import pandas as pd
import numpy as np
import pytest
from rudra.common.encoding import encode_features

def test_empty_dataframe():
    """Test that empty DataFrame is handled correctly."""
    df = pd.DataFrame()
    result = encode_features(df)
    assert result.empty

def test_no_categorical_columns():
    """Test that DataFrame with no categorical columns is returned unchanged."""
    data = {
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [10, 20, 30, 40, 50]
    }
    df = pd.DataFrame(data)
    result = encode_features(df)
    assert result.equals(df)

def test_binary_categorical_columns():
    """Test that binary categorical columns are label encoded."""
    data = {
        'binary': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'numeric': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    result = encode_features(df)
    
    # Check that the binary column was label encoded
    assert result['binary'].dtype in ['int64', 'int32']
    assert set(result['binary'].unique()) == {0, 1}
    
    # Check that the numeric column was not changed
    assert result['numeric'].equals(df['numeric'])

def test_multi_class_categorical_columns():
    """Test that multi-class categorical columns are one-hot encoded."""
    data = {
        'multi_class': ['A', 'B', 'C', 'A', 'B'],
        'numeric': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    result = encode_features(df)
    
    # Check that the multi-class column was one-hot encoded
    assert 'multi_class_A' in result.columns
    assert 'multi_class_B' in result.columns
    assert 'multi_class_C' not in result.columns  # drop_first=True
    
    # Check that the numeric column was not changed
    assert result['numeric'].equals(df['numeric'])

def test_mixed_categorical_columns():
    """Test that mixed categorical columns are handled correctly."""
    data = {
        'binary': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'multi_class': ['A', 'B', 'C', 'A', 'B'],
        'numeric': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    result = encode_features(df)
    
    # Check that the binary column was label encoded
    assert result['binary'].dtype in ['int64', 'int32']
    assert set(result['binary'].unique()) == {0, 1}
    
    # Check that the multi-class column was one-hot encoded
    assert 'multi_class_A' in result.columns
    assert 'multi_class_B' in result.columns
    assert 'multi_class_C' not in result.columns  # drop_first=True
    
    # Check that the numeric column was not changed
    assert result['numeric'].equals(df['numeric'])

def test_missing_values():
    """Test that missing values are handled correctly."""
    data = {
        'binary': ['Yes', 'No', None, 'No', 'Yes'],
        'multi_class': ['A', 'B', 'C', None, 'B'],
        'numeric': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    result = encode_features(df)
    
    # Check that the binary column was label encoded
    assert result['binary'].dtype in ['int64', 'int32']
    
    # Check that the multi-class column was one-hot encoded
    assert 'multi_class_A' in result.columns
    assert 'multi_class_B' in result.columns
    assert 'multi_class_C' not in result.columns  # drop_first=True
    
    # Check that the numeric column was not changed
    assert result['numeric'].equals(df['numeric']) 