import numpy as np
import pandas as pd  
from pathlib import Path
import logging


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file. Remove rows with unlabeled data.

    Args:
        data_path (Path): Path to the CSV data file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with unlabeled data removed.

    Raises:
        FileNotFoundError: If the specified data file does not exist.
    """
    raise NotImplementedError("Implement Exercise 1.1")

def remove_unlabeled_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with unlabeled data (where labels == -1).

    Args:
        data (pd.DataFrame): Input DataFrame containing a 'labels' column.

    Returns:
        pd.DataFrame: DataFrame with unlabeled data removed.
    """
    raise NotImplementedError("Implement Exercise 1.2")


def convert_to_np(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DataFrame to numpy arrays, separating labels, experiment IDs, and features.

    Args:
        data (pd.DataFrame): Input DataFrame containing 'labels', 'exp_ids', and feature columns.

    Returns:
        tuple: A tuple containing:
            - labels (np.ndarray): Array of labels
            - exp_ids (np.ndarray): Array of experiment IDs
            - data (np.ndarray): Combined array of current and voltage features
    """
    raise NotImplementedError("Implement Exercise 1.3")


def create_sliding_windows_first_dim(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Create sliding windows over the first dimension of a 3D array.
    
    Args:
        data (np.ndarray): Input array of shape (n_samples, timesteps, features)
        sequence_length (int): Length of each window
    
    Returns:
        np.ndarray: Windowed data of shape (n_windows, sequence_length, timesteps, features)
    """
    raise NotImplementedError("Implement Exercise 1.4")

def get_welding_data(path: Path, n_samples: int | None = None, return_sequences: bool = False, sequence_length: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load welding data from CSV or cached numpy files.

    If numpy cache files don't exist, loads from CSV and creates cache files.
    If cache files exist, loads directly from them.

    Args:
        path (Path): Path to the CSV data file.
        n_samples (int | None): Number of samples to sample from the data. If None, all data is returned.
        return_sequences (bool): If True, return sequences of length sequence_length.
        sequence_length (int): Length of sequences to return.
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of welding data features
            - np.ndarray: Array of labels
            - np.ndarray: Array of experiment IDs
    """
    np_file_path_data = path.parent / "data.npy"
    np_file_path_labels = path.parent / "labels.npy"
    np_file_path_exp_ids = path.parent / "exp_ids.npy"

    if not np_file_path_data.exists() or not np_file_path_labels.exists() or not np_file_path_exp_ids.exists():
        data = load_data(path)
        labels, exp_ids, np_data = convert_to_np(data)
        logging.info(f"Saving data to {np_file_path_data}")

        np.save(np_file_path_data, np_data)
        np.save(np_file_path_labels, labels)
        np.save(np_file_path_exp_ids, exp_ids)

    np_data = np.load(np_file_path_data)
    labels = np.load(np_file_path_labels)
    exp_ids = np.load(np_file_path_exp_ids)

    if return_sequences:
        np_data = create_sliding_windows_first_dim(np_data, sequence_length)
        labels = np.repeat(labels.reshape(-1, 1), sequence_length, axis=1)[:-sequence_length + 1]
        exp_ids = np.repeat(exp_ids.reshape(-1, 1), sequence_length, axis=1)[:-sequence_length + 1]

    if n_samples is not None:
        sample_idx = np.random.choice(np_data.shape[0], n_samples, replace=False)
        logging.info(f"Sampling {n_samples} samples from {np_data.shape[0]}")
        np_data = np_data[sample_idx]
        labels = labels[sample_idx]
        exp_ids = exp_ids[sample_idx]

    return np_data, labels, exp_ids
