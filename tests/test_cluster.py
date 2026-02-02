import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ISLP.cluster import compute_linkage

def test_linkage():
    # Create a simple dataset
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
    
    # Fit an AgglomerativeClustering model
    hclust = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=0, compute_distances=True)
    hclust.fit(X)
    
    # Compute the linkage matrix
    linkage_matrix = compute_linkage(hclust)
    
    # Check the shape of the output
    n_samples = X.shape[0]
    assert linkage_matrix.shape == (n_samples - 1, 4)
    
    # Check the dtype of the output
    assert linkage_matrix.dtype == float
    
    # Check the counts column
    assert np.all(linkage_matrix[:, 3] > 0)
    final_count = linkage_matrix[-1, 3]
    assert final_count == n_samples
