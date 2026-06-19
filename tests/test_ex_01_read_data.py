import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ex_01_read_data import load_data, remove_unlabeled_data, convert_to_np, get_welding_data

@pytest.fixture
def sample_data():
    # Create a sample DataFrame that mimics the welding data structure
    data = pd.DataFrame({
        'labels': [0, 1, -1, 2],
        'exp_ids': [1, 2, 3, 4],
        'V1': [1.0, 2.0, 3.0, 4.0],
        'V2': [1.1, 2.1, 3.1, 4.1],
        'I1': [0.1, 0.2, 0.3, 0.4],
        'I2': [0.11, 0.21, 0.31, 0.41]
    })
    return data

def test_remove_unlabeled_data(sample_data):
    # Test removing unlabeled data (where label == -1)
    result = remove_unlabeled_data(sample_data)
    assert len(result) == 3
    assert -1 not in result['labels'].values
    
def test_convert_to_np(sample_data):
    # Test converting DataFrame to numpy arrays
    labels, exp_ids, data = convert_to_np(sample_data)
    
    # Check shapes and types
    assert isinstance(labels, np.ndarray)
    assert isinstance(exp_ids, np.ndarray)
    assert isinstance(data, np.ndarray)
    
    # Check data organization (current data followed by voltage data)
    assert data.shape[1] == 2, f"Expected 2 columns, got {data.shape[1]}"
    assert data.shape[2] == 2, f"Expected 2 dimensions (current and voltage), got {data.shape[2]}"
    
def test_load_data(tmp_path):
    # Create a temporary CSV file for testing
    test_df = pd.DataFrame({
        'labels': [0, 1],
        'exp_ids': [1, 2],
        'V1': [1.0, 2.0],
        'I1': [0.1, 0.2]
    })
    test_file = tmp_path / "test_data.csv"
    test_df.to_csv(test_file, index=False)
    
    # Test loading the data
    result = load_data(test_file)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

def test_load_data_file_not_found():
    # Test handling of non-existent file
    with pytest.raises(FileNotFoundError):
        load_data(Path("nonexistent_file.csv"))

def test_get_welding_data(tmp_path):
    # Create temporary test data
    test_df = pd.DataFrame({
        'labels': [0, 1],
        'exp_ids': [1, 2],
        'V1': [1.0, 2.0],
        'I1': [0.1, 0.2]
    })
    test_file = tmp_path / "data.csv"
    test_df.to_csv(test_file, index=False)
    
    # Test the full data loading pipeline
    data, labels, exp_ids = get_welding_data(test_file)
    
    assert isinstance(data, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(exp_ids, np.ndarray)
    
    # Clean up the generated numpy files
    (tmp_path / "data.npy").unlink(missing_ok=True)
    (tmp_path / "labels.npy").unlink(missing_ok=True)
    (tmp_path / "exp_ids.npy").unlink(missing_ok=True)

def test_data_shape_structure(tmp_path):
    # Create temporary test data with multiple columns to simulate time series
    # Create 200 voltage and 200 current columns to match expected shape
    data_dict = {'labels': [0, 1], 'exp_ids': [1, 2]}
    
    # Add 200 voltage columns
    for i in range(1, 201):
        data_dict[f'V{i}'] = i
    
    # Add 200 current columns
    for i in range(1, 201):
        data_dict[f'I{i}'] = i
    
    test_df = pd.DataFrame(data_dict)
    test_file = tmp_path / "shape_test_data.csv"
    test_df.to_csv(test_file, index=False)
    
    # Test the shape of the data
    data, labels, exp_ids = get_welding_data(test_file)
    
    # Check shapes
    assert data.shape[0] == 2, f"Expected 2 samples, got {data.shape[0]}"
    assert data.shape[1] == 200, f"Expected 200 time steps, got {data.shape[1]}"
    assert data.shape[2] == 2, f"Expected 2 dimensions (current and voltage), got {data.shape[2]}"
    assert labels.shape[0] == exp_ids.shape[0] == data.shape[0], f"Expected labels and exp_ids to have the same number of samples, got {labels.shape[0]} and {exp_ids.shape[0]} and {data.shape[0]}"


    # Clean up the generated numpy files
    (tmp_path / "data.npy").unlink(missing_ok=True)
    (tmp_path / "labels.npy").unlink(missing_ok=True)
    (tmp_path / "exp_ids.npy").unlink(missing_ok=True)

def test_empty_dataframe(tmp_path):
    # Create empty dataframe with only headers
    empty_df = pd.DataFrame(columns=['labels', 'exp_ids', 'V1', 'I1'])
    test_file = tmp_path / "empty_data.csv"
    empty_df.to_csv(test_file, index=False)
    
    # Test loading empty data
    with pytest.raises(ValueError):  
        get_welding_data(test_file)

def test_missing_columns(tmp_path):
    # Create dataframe missing required columns
    incomplete_df = pd.DataFrame({
        'exp_ids': [1, 2],
        'V1': [1.0, 2.0],
        'I1': [0.1, 0.2]
    })  # Missing 'labels' column
    test_file = tmp_path / "incomplete_data.csv"
    incomplete_df.to_csv(test_file, index=False)
    
    # Test loading data with missing columns
    with pytest.raises(KeyError):  # Adjust based on your implementation's expected behavior
        get_welding_data(test_file)

def test_invalid_data_types(tmp_path):
    # Create dataframe with non-numeric values
    invalid_df = pd.DataFrame({
        'labels': [0, 1],
        'exp_ids': [1, 2],
        'V1': [1.0, 'error'],  # Non-numeric value
        'I1': [0.1, 0.2]
    })
    test_file = tmp_path / "invalid_data.csv"
    invalid_df.to_csv(test_file, index=False)
    
    # Test handling of invalid data types
    with pytest.raises(ValueError):  # Adjust based on your implementation's expected behavior
        get_welding_data(test_file)

def test_nan_values(tmp_path):
    # Create dataframe with NaN values
    nan_df = pd.DataFrame({
        'labels': [0, 1],
        'exp_ids': [1, 2],
        'V1': [1.0, np.nan],
        'I1': [0.1, 0.2]
    })
    test_file = tmp_path / "nan_data.csv"
    nan_df.to_csv(test_file, index=False)
    
    # Test handling of NaN values
    # Modify this based on your expected behavior (skip, fill, or error)
    try:
        data, labels, exp_ids = get_welding_data(test_file)
        # Assert expected behavior if NaNs are handled
        assert not np.isnan(data).any(), "NaN values should be handled"
    except Exception as e:
        # If your implementation is expected to raise an error, adjust accordingly
        assert isinstance(e, (ValueError, pd.errors.EmptyDataError))

def test_all_unlabeled_data(sample_data):
    # Test behavior when all data is unlabeled
    all_unlabeled = sample_data.copy()
    all_unlabeled['labels'] = -1
    
    result = remove_unlabeled_data(all_unlabeled)
    assert len(result) == 0, "Expected empty DataFrame when all data is unlabeled"

def test_data_sampling(tmp_path):
    # Create temporary test data with multiple rows
    test_df = pd.DataFrame({
        'labels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'exp_ids': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        'V1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'I1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    test_file = tmp_path / "sample_data.csv"
    test_df.to_csv(test_file, index=False)
    
    # Test with n_samples parameter
    n_samples = 5
    data, labels, exp_ids = get_welding_data(test_file, n_samples=n_samples)
    
    # Check that we got exactly n_samples
    assert data.shape[0] == n_samples, f"Expected {n_samples} samples, got {data.shape[0]}"
    assert labels.shape[0] == n_samples, f"Expected {n_samples} labels, got {labels.shape[0]}"
    assert exp_ids.shape[0] == n_samples, f"Expected {n_samples} exp_ids, got {exp_ids.shape[0]}"
    
    # Verify all sampled values are from the original dataset
    for label in labels:
        assert label in test_df['labels'].values, f"Label {label} not found in original data"
    
    for exp_id in exp_ids:
        assert exp_id in test_df['exp_ids'].values, f"Experiment ID {exp_id} not found in original data"
    
    # Test with n_samples equal to the full dataset
    full_data, full_labels, full_exp_ids = get_welding_data(test_file, n_samples=len(test_df))
    assert full_data.shape[0] == len(test_df), "Should return all samples when n_samples equals dataset size"
    
    # Test with n_samples=None (should return all data)
    all_data, all_labels, all_exp_ids = get_welding_data(test_file, n_samples=None)
    assert all_data.shape[0] == len(test_df), "Should return all samples when n_samples is None"
    
    # Test with n_samples larger than dataset (should handle gracefully or raise error)
    try:
        large_data, large_labels, large_exp_ids = get_welding_data(test_file, n_samples=len(test_df)+10)
        # If it doesn't raise an error, it should at least return the full dataset
        assert large_data.shape[0] <= len(test_df), "Should not return more samples than available"
    except ValueError:
        # Alternative: function could raise ValueError when n_samples > available data
        pass
    
    # Clean up the generated numpy files
    (tmp_path / "data.npy").unlink(missing_ok=True)
    (tmp_path / "labels.npy").unlink(missing_ok=True)
    (tmp_path / "exp_ids.npy").unlink(missing_ok=True)

def test_sequence_window_structure_and_content(tmp_path):
    """Test to verify the exact structure and content of the sequence windows."""
    # Create the same test data as in the main script
    data_dict = {'labels': list(range(4)), 'exp_ids': list(range(1, 5))}
    
    # Add voltage and current columns
    for i in range(1, 5):  # 4 timesteps
        data_dict[f'V{i}'] = [i * j for j in range(4)]
        data_dict[f'I{i}'] = [i * j * 0.1 for j in range(4)]
    
    # Create a temporary file for testing
    test_file = Path(tmp_path) / "sequence_test_data.csv"
    pd.DataFrame(data_dict).to_csv(test_file, index=False)
    
    # Test with sequence length 2
    seq_length = 2
    data, labels, exp_ids = get_welding_data(
        test_file, 
        return_sequences=True, 
        sequence_length=seq_length
    )
    
    # Check shapes for sequence length 2
    assert data.shape == (3, 2, 4, 2), f"Expected shape (3, 2, 4, 2) for sequence length 2, got {data.shape}"
    assert labels.shape == (3, 2), f"Expected labels shape (3, 2), got {labels.shape}"
    assert exp_ids.shape == (3, 2), f"Expected exp_ids shape (3, 2), got {exp_ids.shape}"
    
    # Test with sequence length 4
    seq_length = 4
    data_long, labels_long, exp_ids_long = get_welding_data(
        test_file, 
        return_sequences=True, 
        sequence_length=seq_length
    )
    
    # Check shapes for sequence length 4
    assert data_long.shape == (1, 4, 4, 2), f"Expected shape (1, 4, 4, 2) for sequence length 4, got {data_long.shape}"
    assert labels_long.shape == (1, 4), f"Expected labels shape (1, 4), got {labels_long.shape}"
    assert exp_ids_long.shape == (1, 4), f"Expected exp_ids shape (1, 4), got {exp_ids_long.shape}"
    
    # Verify sample content for sequence length 2
    # First window should contain data from samples 0 and 1
    first_window = data[0]
    # Check the first few values from each feature for the first window
    assert first_window[0, 0, 0] == 0.0
    assert first_window[0, 0, 1] == 0.0
    assert first_window[1, 0, 0] == 0.1
    assert first_window[1, 0, 1] == 1.0

    
    # Clean up the generated numpy files
    (tmp_path / "data.npy").unlink(missing_ok=True)
    (tmp_path / "labels.npy").unlink(missing_ok=True)
    (tmp_path / "exp_ids.npy").unlink(missing_ok=True)

