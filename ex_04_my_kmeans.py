from typing import Literal
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from dtaidistance import dtw

DISTANCE_METRICS = Literal["euclidean", "manhattan", "dtw"]
INIT_METHOD = Literal["random", "kmeans++"]

class MyKMeans:
    """
    Custom K-means clustering implementation with support for multiple distance metrics.
    
    Args:
        k (int): Number of clusters.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        distance_metric (str, optional): Distance metric to use. Options are "euclidean", 
                                         "manhattan", or "dtw". Defaults to "euclidean".
        init_method (str, optional): Initialization method to use. Options are "kmeans++" or "random". Defaults to "kmeans++".
    """
    def __init__(self, k: int, max_iter: int = 100, distance_metric: DISTANCE_METRICS = "euclidean", init_method: INIT_METHOD = "kmeans++"):
        self.k: int = k
        self.max_iter: int = max_iter
        self.centroids: np.ndarray | None = None
        self.distance_metric: DISTANCE_METRICS = distance_metric
        self.inertia_: float | None = None
        self.init_method: INIT_METHOD = init_method

    def fit(self, x: np.ndarray | pd.DataFrame):
        """
        Fit the K-means model to the data.
        
        Args:
            x (np.ndarray | pd.DataFrame): Training data of shape (n_samples, n_features).
        
        Returns:
            MyKMeans: Fitted estimator instance.
        """
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError("Input data must be a numpy array or a pandas DataFrame")
        
        # Code here
        
        return self
    

    def fit_predict(self, x: np.ndarray):
        """
        Fit the K-means model to the data and return the predicted labels.
        """
        if isinstance(x, pd.DataFrame):
            x = x.values
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError("Input data must be a numpy array or a pandas DataFrame")
        self.fit(x)
        return self.predict(x)

    def predict(self, x: np.ndarray):
        """
        Predict the closest cluster for each sample in x.
        
        Args:
            x (np.ndarray): New data to predict, of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        # Compute distances between samples and centroids
        distances = self._compute_distance(x, self.centroids)
        
        # Return the index of the closest centroid for each sample
        return np.argmin(distances, axis=1)

    def _initialize_centroids(self, x: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using the kmeans++ method.
        
        Args:
            x (np.ndarray): Training data.
            
        Returns:
            np.ndarray: Initial centroids.
        """
        pass

    def _compute_distance(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute the distance between samples and centroids.
        
        Args:
            x (np.ndarray): Data points of shape (n_samples, n_features) or (n_samples, time_steps, n_features).
            centroids (np.ndarray): Centroids of shape (k, n_features) or (k, time_steps, n_features).
            
        Returns:
            np.ndarray: Distances between each sample and each centroid, shape (n_samples, k).
        """
        pass


    def _dtw(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Simplified DTW distance computation using dtaidistance.
        
        Args:
            x (np.ndarray): Data points of shape (n_samples, time_steps, n_features) or (n_samples, n_features)
            centroids (np.ndarray): Centroids of shape (k, time_steps, n_features) or (k, n_features)
            
        Returns:
            np.ndarray: DTW distances between each sample and each centroid, shape (n_samples, k).
        """
        pass
