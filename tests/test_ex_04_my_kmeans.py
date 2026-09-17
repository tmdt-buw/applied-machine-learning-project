import pytest
import numpy as np
import pandas as pd
from ex_04_my_kmeans import MyKMeans
from dtaidistance import dtw

@pytest.fixture
def data_2d():
    """Fixture for 2D data (samples x features)"""
    np.random.seed(42)
    return np.random.rand(100, 10)

@pytest.fixture
def data_3d():
    """Fixture for 3D data (samples x time_steps x features)"""
    np.random.seed(42)
    return np.random.rand(100, 15, 2)

@pytest.fixture
def data_3d_short_seq():
    """Fixture for 3D data with shorter sequence length"""
    np.random.seed(42)
    return np.random.rand(100, 8, 2)

@pytest.fixture
def data_3d_long_seq():
    """Fixture for 3D data with longer sequence length"""
    np.random.seed(42)
    return np.random.rand(100, 25, 2)

@pytest.fixture
def df_data_2d(data_2d):
    """Fixture for 2D data as pandas DataFrame"""
    return pd.DataFrame(data_2d)

class TestMyKMeansInitialization:
    def test_init_default_params(self):
        """Test initialization with default parameters"""
        kmeans = MyKMeans(k=3)
        assert kmeans.k == 3
        assert kmeans.max_iter == 100
        assert kmeans.distance_metric == "euclidean"
        assert kmeans.centroids is None
        assert kmeans.inertia_ is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters"""
        kmeans = MyKMeans(k=5, max_iter=200, distance_metric="manhattan")
        assert kmeans.k == 5
        assert kmeans.max_iter == 200
        assert kmeans.distance_metric == "manhattan"
        assert kmeans.centroids is None
        assert kmeans.inertia_ is None

    def test_init_invalid_metric(self):
        """Test initialization with invalid distance metric"""
        with pytest.raises(ValueError):
            # This will raise during _compute_distance, not during initialization
            kmeans = MyKMeans(k=3, distance_metric="invalid_metric")
            kmeans.fit(np.random.rand(10, 5))


class TestMyKMeansInput:
    def test_fit_numpy_input_2d(self, data_2d):
        """Test fitting with 2D numpy array"""
        kmeans = MyKMeans(k=3)
        kmeans.fit(data_2d)
        assert kmeans.centroids.shape == (3, 10)
        assert isinstance(kmeans.inertia_, float)

    def test_fit_numpy_input_3d(self, data_3d):
        """Test fitting with 3D numpy array"""
        kmeans = MyKMeans(k=3)
        kmeans.fit(data_3d)
        assert kmeans.centroids.shape == (3, 15, 2)
        assert isinstance(kmeans.inertia_, float)

    def test_fit_pandas_input(self, df_data_2d):
        """Test fitting with pandas DataFrame"""
        kmeans = MyKMeans(k=3)
        kmeans.fit(df_data_2d)
        assert kmeans.centroids.shape == (3, 10)
        assert isinstance(kmeans.inertia_, float)

    def test_fit_invalid_input(self):
        """Test fitting with invalid input"""
        kmeans = MyKMeans(k=3)
        with pytest.raises(ValueError, match="Input data must be a numpy array or a pandas DataFrame"):
            kmeans.fit("invalid_input")

    def test_fit_invalid_dimensions(self):
        """Test fitting with invalid dimensions"""
        kmeans = MyKMeans(k=3)
        with pytest.raises(ValueError, match="Input data must be a 2D or 3D array"):
            kmeans.fit(np.random.rand(10))  # 1D array


class TestMyKMeansDistanceMetrics:
    def test_euclidean_distance_2d(self, data_2d):
        """Test euclidean distance metric with 2D data"""
        kmeans = MyKMeans(k=3, distance_metric="euclidean")
        kmeans.fit(data_2d)
        assert kmeans.centroids.shape == (3, 10)
        
        # Verify predictions work
        labels = kmeans.predict(data_2d)
        assert labels.shape == (100,)
        assert np.all(labels >= 0) and np.all(labels < 3)

    def test_euclidean_distance_3d(self, data_3d):
        """Test euclidean distance metric with 3D data"""
        kmeans = MyKMeans(k=3, distance_metric="euclidean")
        kmeans.fit(data_3d)
        assert kmeans.centroids.shape == (3, 15, 2)
        
        # Verify predictions work
        labels = kmeans.predict(data_3d)
        assert labels.shape == (100,)
        assert np.all(labels >= 0) and np.all(labels < 3)

    def test_manhattan_distance_2d(self, data_2d):
        """Test manhattan distance metric with 2D data"""
        kmeans = MyKMeans(k=3, distance_metric="manhattan")
        kmeans.fit(data_2d)
        assert kmeans.centroids.shape == (3, 10)
        
        # Verify predictions work
        labels = kmeans.predict(data_2d)
        assert labels.shape == (100,)
        assert np.all(labels >= 0) and np.all(labels < 3)

    def test_manhattan_distance_3d(self, data_3d):
        """Test manhattan distance metric with 3D data"""
        kmeans = MyKMeans(k=3, distance_metric="manhattan")
        kmeans.fit(data_3d)
        assert kmeans.centroids.shape == (3, 15, 2)
        
        # Verify predictions work
        labels = kmeans.predict(data_3d)
        assert labels.shape == (100,)
        assert np.all(labels >= 0) and np.all(labels < 3)

    def test_dtw_distance_2d(self, data_2d):
        """Test DTW distance metric with 2D data"""
        kmeans = MyKMeans(k=2, max_iter=10, distance_metric="dtw")  # Use fewer clusters and iterations to speed up test
        kmeans.fit(data_2d)
        assert kmeans.centroids.shape == (2, 10)
        
        # Verify predictions work
        labels = kmeans.predict(data_2d)
        assert labels.shape == (100,)
        assert np.all(labels >= 0) and np.all(labels < 2)

    def test_dtw_distance_3d(self, data_3d):
        """Test DTW distance metric with 3D data"""
        kmeans = MyKMeans(k=2, max_iter=10, distance_metric="dtw")  # Use fewer clusters and iterations to speed up test
        kmeans.fit(data_3d)
        assert kmeans.centroids.shape == (2, 15, 2)
        
        # Verify predictions work
        labels = kmeans.predict(data_3d)
        assert labels.shape == (100,)
        assert np.all(labels >= 0) and np.all(labels < 2)


class TestMyKMeansMethods:
    def test_predict(self, data_2d):
        """Test predict method"""
        kmeans = MyKMeans(k=3)
        kmeans.fit(data_2d)
        
        # Test prediction on the same data
        labels = kmeans.predict(data_2d)
        assert labels.shape == (100,)
        assert np.all(labels >= 0) and np.all(labels < 3)
        
        # Test prediction on new data
        new_data = np.random.rand(20, 10)
        new_labels = kmeans.predict(new_data)
        assert new_labels.shape == (20,)
        assert np.all(new_labels >= 0) and np.all(new_labels < 3)

    def test_fit_predict(self, data_2d):
        """Test fit_predict method"""
        kmeans = MyKMeans(k=3)
        labels = kmeans.fit_predict(data_2d)
        assert labels.shape == (100,)
        assert np.all(labels >= 0) and np.all(labels < 3)
        assert kmeans.centroids is not None
        assert kmeans.inertia_ is not None

    def test_centroid_initialization(self, data_2d):
        """Test centroid initialization"""
        np.random.seed(42)  # Set seed for reproducibility
        kmeans = MyKMeans(k=3)
        centroids = kmeans._initialize_centroids(data_2d)
        
        # Check that centroids are taken from data points
        assert centroids.shape == (3, 10)
        
        # Check that each centroid exists in the data
        for centroid in centroids:
            assert np.any(np.all(data_2d == centroid, axis=1))


class TestMyKMeansConvergence:
    def test_clustering_quality(self):
        """Test that the KMeans algorithm produces reasonable clusters for well-separated data"""
        # Create a simple dataset with clear clusters
        np.random.seed(42)
        cluster1 = np.random.normal(0, 0.1, (30, 2))    # Cluster 1 near (0,0)
        cluster2 = np.random.normal(5, 0.1, (30, 2))    # Cluster 2 near (5,5)
        cluster3 = np.random.normal(10, 0.1, (30, 2))   # Cluster 3 near (10,10)
        
        # Combine the clusters
        data = np.vstack([cluster1, cluster2, cluster3])
        
        # Create a deterministic set of initial centroids for testing
        # We'll manually pick points from each cluster to ensure good initialization
        initial_centroids = np.array([
            cluster1[0],    # A point from cluster 1
            cluster2[0],    # A point from cluster 2
            cluster3[0]     # A point from cluster 3
        ])
        
        # Create KMeans instance
        kmeans = MyKMeans(k=3, max_iter=100)
        
        # Override the random initialization with our deterministic one
        # We'll use monkey patching for the test
        original_init_fn = kmeans._initialize_centroids
        kmeans._initialize_centroids = lambda x: initial_centroids
        
        # Fit the model
        kmeans.fit(data)
        
        # Restore the original function
        kmeans._initialize_centroids = original_init_fn
        
        # Get cluster assignments
        labels = kmeans.predict(data)
        
        # Test 1: Check that we have 3 clusters
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 3
        
        # Test 2: Verify that points from the same original cluster tend to be 
        # assigned to the same KMeans cluster
        
        # Get the dominant cluster for each of our original clusters
        cluster1_labels = labels[:30]  # First 30 points are from cluster 1
        cluster2_labels = labels[30:60]  # Next 30 are from cluster 2
        cluster3_labels = labels[60:]  # Last 30 are from cluster 3
        
        # For each original cluster, get most common assigned label
        most_common_label1 = np.bincount(cluster1_labels).argmax()
        most_common_label2 = np.bincount(cluster2_labels).argmax()
        most_common_label3 = np.bincount(cluster3_labels).argmax()
        
        # These most common labels should be different from each other
        assert most_common_label1 != most_common_label2
        assert most_common_label1 != most_common_label3
        assert most_common_label2 != most_common_label3
        
        # Test 3: The majority of points in each original cluster should be 
        # assigned to the same kmeans cluster
        assert np.sum(cluster1_labels == most_common_label1) >= 20  # At least 20/30 points
        assert np.sum(cluster2_labels == most_common_label2) >= 20
        assert np.sum(cluster3_labels == most_common_label3) >= 20
        
        # Test 4: The final centroids should be reasonably close to the true cluster centers
        # Get the centroids sorted by their first coordinate
        sorted_centroids = kmeans.centroids[np.argsort(kmeans.centroids[:, 0])]
        
        # Check reasonable proximity to true centers
        assert np.linalg.norm(sorted_centroids[0] - [0, 0]) < 1.0
        assert np.linalg.norm(sorted_centroids[1] - [5, 5]) < 1.0
        assert np.linalg.norm(sorted_centroids[2] - [10, 10]) < 1.0


class TestMyKMeansVariableSequenceLengths:
    """Test KMeans with variable sequence lengths"""
    
    def test_short_sequence(self, data_3d_short_seq):
        """Test KMeans with shorter sequence length"""
        kmeans = MyKMeans(k=3)
        kmeans.fit(data_3d_short_seq)
        assert kmeans.centroids.shape == (3, 8, 2)  # Check centroid shape matches sequence length
        
        # Verify predictions work
        labels = kmeans.predict(data_3d_short_seq)
        assert labels.shape == (100,)
        assert np.all(labels >= 0) and np.all(labels < 3)
    
    def test_long_sequence(self, data_3d_long_seq):
        """Test KMeans with longer sequence length"""
        kmeans = MyKMeans(k=3)
        kmeans.fit(data_3d_long_seq)
        assert kmeans.centroids.shape == (3, 25, 2)  # Check centroid shape matches sequence length
        
        # Verify predictions work
        labels = kmeans.predict(data_3d_long_seq)
        assert labels.shape == (100,)
        assert np.all(labels >= 0) and np.all(labels < 3)
    
    def test_dtw_different_seq_lengths(self, data_3d_short_seq, data_3d_long_seq):
        """Test DTW distance with different sequence lengths"""
        # Train on short sequences
        kmeans = MyKMeans(k=2, max_iter=10, distance_metric="dtw")
        kmeans.fit(data_3d_short_seq)
        assert kmeans.centroids.shape == (2, 8, 2)
        
        # Test on both sequence lengths to ensure DTW works with different lengths
        labels_short = kmeans.predict(data_3d_short_seq)
        assert labels_short.shape == (100,)
        
        # This should work because DTW can handle sequences of different lengths
        labels_long = kmeans.predict(data_3d_long_seq)
        assert labels_long.shape == (100,) 