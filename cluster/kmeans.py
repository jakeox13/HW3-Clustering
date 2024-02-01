import numpy as np
from scipy.spatial.distance import cdist
import warnings


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
                """
        # Check for valid Inputs
        if (type(k)!= int) or k <1 :
            raise ValueError("Invaild k, K must be an integer greater or equal to 1")
        if tol < 0:
            raise ValueError("tol must be a postive value")
        if (type(max_iter)!= int) or max_iter <1 :
            raise ValueError("max_iter must be an integer greater or equal to 1")
        
        # assign Inital values
        self.k=k
        self.tol =tol
        self.max_iter= max_iter

       





    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.


        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # Check that there are at least as many data points as clusters
        if mat.shape[0] < self.k:
            raise ValueError("Data must include at least k datapoints")

        # Intlize random seed
        rng=np.random.default_rng(None)
        # Intilize centers to random values

        #  Choose k random starting points as initial clusters
        number_of_rows = mat.shape[0] 
        random_indices = rng.choice(number_of_rows,  
                                  size=self.k,  
                                  replace=False)
        self.centers= mat[random_indices, :]
        

        # Set up ending criteria
        iters=0
        derr=1
        err=1
        
        while iters < self.max_iter and derr > self.tol:
            # Calculate distance between all points and centers
            dists= cdist(self.centers, mat, 'euclidean')
            
            # Find which cluster each belongs to 
            min_dists = np.argmin(dists, axis=0)
            
            
            # One hot encode the cluster
            one_hot_matrix = np.eye(self.k)[min_dists]

            # Calculate total error
            # Get all the distances based on what the closest cluster is
            each_dist=[]
            for i, value in enumerate(min_dists):
                each_dist.append(dists[value][i])
            
            # square all distances and sum to get root mean squared erro
            each_dist=[x ** 2 for x in each_dist]
            rms=sum(each_dist)
            

            # Calculate new center as averge of all points in cluster

            # Calculate the sum of values for each group
            sum_values = np.dot(one_hot_matrix.T, mat)
            

            # Count the number of occurrences for each group
            group_counts = np.sum(one_hot_matrix, axis=0)

            # Avoid division by zero
            group_counts[group_counts == 0] = 1
            # If empty cluater == Failure
            #raise ValueError("Cluster lost, convergence failed")

            # Calculate the average values for each group
            # Note if no poiints are assigned to a cluster cluster is reset to 0,0
            self.centers = sum_values / group_counts[:, np.newaxis]
        
            # Derr is the percentage change is 
            derr= abs(err-rms)/err
            err=rms
            # Increment the interartions
            iters += 1
        if iters == self.max_iter and derr >self.tol:
            warnings.warn("Failed to converge after {} iterations".format(self.max_iter))
        if iters < self.max_iter and derr < self.tol:
            print("Converged after {} interations".format(iters))
        # Assign error
        self.error=err


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        # Calculate distance between all points and centers
        dists= cdist(self.centers, mat, 'euclidean')
        
        # Find which cluster each belongs to 
        min_dists = np.argmin(dists, axis=0)
        return min_dists

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centers