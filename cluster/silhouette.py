import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # Establish results as acessible
        self._y=y
        # Generate a distance matrix
        dist_mat=self._get_dist_mat(X)
        
        scores=[]
        # For each point
        for i in range(dist_mat.shape[0]):
            
            # Calculate a (avergae distance to points within own cluster)
            a=self._get_cluster_avg(i, dist_mat)
        # Calculate b (average distance to points in other clusters and take min)
            b=self._get_nearest_avg(i,dist_mat)
        # Give score = a-b/max (a,b)
            scores.append((b-a)/max(a,b))
        

        return scores
    
    def _get_dist_mat(self, clust: np.array):
        """""
        Generates an array with pairwise distances between each point to be used to look up cluster distances
        Args:
            clust (np.array): _description_

        Returns:
            np.array: an array with the pairwise distcance calculated between each point 
        """        
        return cdist(clust, clust, 'euclidean')
                          
    def _get_cluster_avg(self,i, dist_mat:np.array):
        """
        Finds the average distance to all other points in the same cluster
        Args:
            i (int): Which row of matrix to look at
            dist_mat (np.array): The array with pairwise distance values

        Returns:
            float: The average distance from a point given by i to all point with in the same cluster
        """        
        
        #Filter pairwise distances for those within same cluster
        sum_of_distances=np.sum(dist_mat[self._y==self._y[i]][:,i])

        number_in_cluster=dist_mat[self._y==self._y[i]].shape[0]
    
        #return the average of distance to all points 
        return sum_of_distances/number_in_cluster
    
    def _get_nearest_avg(self,i, dist_mat:np.array):
        """
        Finds the average distance to points within the nearest cluster
        Args:
            i (int): Which row of matrix to look at
            dist_mat (np.array): The array with pairwise distance values

        Returns:
            float: The average distance from a point given by i to all point with in the nearest cluster
        """        
        # Find all unique valuesof cluster names
        other_clusters=np.unique(self._y[self._y != self._y[i]])
        
        # Loop through and calculate scores for each and save in result matrix
        results=[]                         
        for cluster in other_clusters:
            results.append(np.sum(dist_mat[self._y==cluster][:,i])/dist_mat[self._y==cluster].shape[0])
        # Return min value (closest cluster)
        return min(results)