import pytest
import pandas as pd
import sys
import os

# Add the repo directory to sys.path so pytest can find the module
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import preprocess_tree_based as tbp

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'num_feature': [1, 2, None, 4, 5, 1000],
        'cat_feature': ['A', 'B', 'A', None, 'C', 'A'],
        'high_card_feature': ['id1', 'id2', 'id1', 'id3', 'id4', 'id1'],
        'target': [0, 1, 0, 1, 0, 1]
    })

def test_handle_missing_values(sample_data):
    df = tbp.handle_missing_values_tree(sample_data)
    assert df.isnull().sum().sum() == 0  # Ensure no missing values remain

def test_handle_categorical_variables(sample_data):
    df = tbp.handle_categorical_variables_tree(sample_data, target='target')
    assert "cat_feature" not in df.columns or df["cat_feature"].dtype != "object"

def test_feature_selection(sample_data):
    df = tbp.select_features_tree(sample_data)
    assert 'num_feature' in df.columns  # Ensure important features remain

if __name__ == "__main__":
    pytest.main()
