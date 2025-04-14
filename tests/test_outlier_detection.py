import pandas as pd
import numpy as np
import pytest
import os
import tempfile
from unittest.mock import patch
from rudra.common.outlier_detection import detect_and_handle_outliers

def create_test_data():
    """Create a test DataFrame with known outliers."""
    data = {
        'normal': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'outliers': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],  # 100 is an outlier
        'multiple_outliers': [1, 2, 3, 4, 5, 100, 7, 8, 9, 200]  # 100 and 200 are outliers
    }
    return pd.DataFrame(data)

def test_iqr_method():
    """Test the IQR method for outlier detection."""
    # Create a temporary file with test data
    df = create_test_data()
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        df.to_csv(temp_file.name, index=False)
        
        # Mock the input function to return the temp file path
        with patch('builtins.input', return_value=temp_file.name):
            # Create another temp file for the output
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as output_file:
                # Run the function
                detect_and_handle_outliers(method='iqr', handling='median', save_path=output_file.name)
                
                # Read the output file
                result_df = pd.read_csv(output_file.name)
                
                # Check that the outlier was handled
                assert result_df['outliers'].max() < 100
                
                # Clean up
                os.unlink(temp_file.name)
                os.unlink(output_file.name)

def test_zscore_method():
    """Test the Z-score method for outlier detection."""
    # Create a temporary file with test data
    df = create_test_data()
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        df.to_csv(temp_file.name, index=False)
        
        # Mock the input function to return the temp file path
        with patch('builtins.input', return_value=temp_file.name):
            # Create another temp file for the output
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as output_file:
                # Run the function
                detect_and_handle_outliers(method='zscore', handling='mean', save_path=output_file.name)
                
                # Read the output file
                result_df = pd.read_csv(output_file.name)
                
                # Check that the outlier was handled
                assert result_df['outliers'].max() < 100
                
                # Clean up
                os.unlink(temp_file.name)
                os.unlink(output_file.name)

def test_isolation_forest_method():
    """Test the Isolation Forest method for outlier detection."""
    # Create a temporary file with test data
    df = create_test_data()
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        df.to_csv(temp_file.name, index=False)
        
        # Mock the input function to return the temp file path
        with patch('builtins.input', return_value=temp_file.name):
            # Create another temp file for the output
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as output_file:
                # Run the function
                detect_and_handle_outliers(method='isolation_forest', handling='remove', save_path=output_file.name)
                
                # Read the output file
                result_df = pd.read_csv(output_file.name)
                
                # Check that the outlier was removed (row count should be less)
                assert len(result_df) < len(df)
                
                # Clean up
                os.unlink(temp_file.name)
                os.unlink(output_file.name)

def test_invalid_method():
    """Test that an invalid method raises a ValueError."""
    # Create a temporary file with test data
    df = create_test_data()
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        df.to_csv(temp_file.name, index=False)
        
        # Mock the input function to return the temp file path
        with patch('builtins.input', return_value=temp_file.name):
            # Create another temp file for the output
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as output_file:
                # Run the function with an invalid method
                with pytest.raises(ValueError):
                    detect_and_handle_outliers(method='invalid_method', handling='median', save_path=output_file.name)
                
                # Clean up
                os.unlink(temp_file.name)
                os.unlink(output_file.name)

def test_invalid_handling():
    """Test that an invalid handling method raises a ValueError."""
    # Create a temporary file with test data
    df = create_test_data()
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        df.to_csv(temp_file.name, index=False)
        
        # Mock the input function to return the temp file path
        with patch('builtins.input', return_value=temp_file.name):
            # Create another temp file for the output
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as output_file:
                # Run the function with an invalid handling method
                with pytest.raises(ValueError):
                    detect_and_handle_outliers(method='iqr', handling='invalid_handling', save_path=output_file.name)
                
                # Clean up
                os.unlink(temp_file.name)
                os.unlink(output_file.name)

def test_non_numeric_columns():
    """Test that the function handles non-numeric columns correctly."""
    # Create a DataFrame with both numeric and non-numeric columns
    data = {
        'numeric': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'categorical': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        df.to_csv(temp_file.name, index=False)
        
        # Mock the input function to return the temp file path
        with patch('builtins.input', return_value=temp_file.name):
            # Create another temp file for the output
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as output_file:
                # Run the function
                detect_and_handle_outliers(method='iqr', handling='median', save_path=output_file.name)
                
                # Read the output file
                result_df = pd.read_csv(output_file.name)
                
                # Check that the categorical column was not modified
                assert 'categorical' in result_df.columns
                
                # Clean up
                os.unlink(temp_file.name)
                os.unlink(output_file.name) 