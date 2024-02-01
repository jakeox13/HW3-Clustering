import numpy as np
import statistics
from sklearn.metrics import silhouette_samples
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters)


def test_silhouette():
    """
    Unit test for correct implimentatino of Silhouette scoring 
    """
    # Generate a cluster and fit model
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    scores_sk =silhouette_samples(clusters, pred)
    
    # Check that the average differnce between my scoring and sk learn scoring is less than 0.1
    assert statistics.mean(abs(x - y) for x, y in zip(scores, scores_sk))<=0.1

    