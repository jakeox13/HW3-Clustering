import pytest
import numpy as np
from cluster import (
        KMeans,
        make_clusters)

def test_kmeans_correct():
    """
    Unit test for correct implimentatino of k means cluster
    """
    # Generate a cluster and fit model
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    
    # Check that there is a preditction generated for each value in cluster
    assert pred.shape[0]==clusters.shape[0]

    # Check that there are k numbers of cluster
    assert len(np.unique(pred))==4

    # Check that it runs on higer dimesional data
    clusters, labels = make_clusters(k=4, scale=1,m=30)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)

    # Check that centroid given are the correct dimensions at higher dimensions
    assert km.get_centroids().shape[1]==30



def test_kmeans_errors():
    """
    Unit test for incorrect implimentation of k means cluster
    """
    
    
    # k less than 1
    with pytest.raises(ValueError, match= r"Invaild k, K must be an integer greater or equal to 1"):
        km = KMeans(k=-4)
    
    # tol less than 0
    with pytest.raises(ValueError, match= r"tol must be a postive value"):
        km = KMeans(k=4, tol=-1)
    
    # max iter negative
    with pytest.raises(ValueError, match= r"max_iter must be an integer greater or equal to 1"):
        km = KMeans(k=4,max_iter=-1)
   
   # k less than data points present
    with pytest.raises(ValueError, match= r"Data must include at least k datapoints"):
        km = KMeans(k=200)
        # Generate a cluster
        clusters, labels = make_clusters(k=5, scale=1,n=100)
        km.fit(clusters)
