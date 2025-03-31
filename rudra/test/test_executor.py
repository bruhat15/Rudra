import pytest
import pandas as pd
import sys
import os

# Ensure the 'rudra' directory is in sys.path to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rudra import executor  # Import the module being tested

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "Age": [25, None, 35, 45, 50],
        "Salary": [50000, 60000, None, 80000, 90000],
        "Department": ["HR", "Finance", None, "HR", "IT"],
        "Experience": [2, 5, None, 8, 10]
    })

def test_preprocess_distance_data(sample_data):
    processed_df = executor.preprocess_distance_data(sample_data)
    assert processed_df.isnull().sum().sum() == 0  # Ensure no missing values remain

def test_preprocess_data(tmp_path):
    # Create a temporary CSV file
    csv_file = tmp_path / "test_data.csv"
    sample_df = pd.DataFrame({
        "Age": [25, None, 35, 45, 50],
        "Salary": [50000, 60000, None, 80000, 90000],
        "Department": ["HR", "Finance", None, "HR", "IT"],
        "Experience": [2, 5, None, 8, 10]
    })
    sample_df.to_csv(csv_file, index=False)
    
    processed_df = executor.preprocess_data(str(csv_file))
    assert processed_df.isnull().sum().sum() == 0  # Ensure missing values are handled

if __name__ == "__main__":
    pytest.main()
