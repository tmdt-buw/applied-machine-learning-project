import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from ex_03_feature_extraction import find_dominant_frequencies, extract_features

# Create synthetic test data
@pytest.fixture
def synthetic_data():
    # Create small synthetic dataset: 10 samples, 50 timepoints, 2 channels (V and I)
    num_samples = 10
    seq_len = 50
    np.random.seed(42)  # For reproducibility
    
    # Random voltage and current data
    data = np.random.rand(num_samples, seq_len, 2) 
    # Random binary labels (good/bad quality)
    labels = np.random.randint(0, 2, size=num_samples)
    
    return data, labels

def test_find_dominant_frequencies_shape():
    """Test that find_dominant_frequencies returns expected shape"""
    # Generate test signals: 5 samples, 100 time points
    x = np.random.rand(5, 100)
    fs =.10
    
    # Get dominant frequencies
    dom_freqs = find_dominant_frequencies(x, fs)
    
    # Check shape is correct (one frequency per sample)
    assert dom_freqs.shape == (5,)
    # Check frequencies are within expected range based on fs
    assert np.all(dom_freqs <= fs/2)

def test_extract_features_shape(synthetic_data):
    """Test that the feature extraction produces correctly shaped output with expected columns"""
    data, labels = synthetic_data
    
    # Extract features
    features_df = extract_features(data, labels)
    
    # Test shape
    expected_cols = 21  # 20 features + 1 label column
    assert features_df.shape == (len(data), expected_cols)
    

def test_extract_features_no_nans(synthetic_data):
    """Test that the feature extraction doesn't produce NaN values"""
    data, labels = synthetic_data
    
    # Extract features
    features_df = extract_features(data, labels)
    
    # Check no NaNs
    assert not features_df.isna().any().any()

