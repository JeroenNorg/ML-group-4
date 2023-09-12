'''
Implement a k-Nearest neighbors classifier from scratch in Python using only
basic matrix operations with numpy and scipy.
Train and evaluate the classifier on the breast cancer dataset, using all features.
Show the performance of the classifier for different values of k (plot the results in a graph).
Note that for optimal results, you should normalize the features
(e.g. to the [0,1] range or to have a zero mean and unit standard deviation).
'''

import numpy as np
from scipy.spatial import distance

class KnnCalculation:

    def __init__(self, k: int,
                 X_train: np.ndarray, X_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray = None,
                 scaling: bool = True) -> None:
        """
        Samples in rows; Features in columns
        :param k: # neighbors; preferably odd to prevent ties
        :param X_train: Training data
        :param X_test: Test data
        :param y_train: Binary train target [0,1]
        :param y_test:  Binary test target [0,1]
        :param scaling: If features need to be scaled to [0,1]
        """
        self.k = k
        self.distance_order = None

        # scaling; for Xtest use params of Xtrain
        if scaling:
            self.X_train, minmaxvalues = self.min_max(X_train)
            self.X_test, _ = self.min_max(X_test, minmaxvalues)
        else:
            self.X_train = X_train
            self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # generate a ranked distance matrix
        self.distance_calc()
        # get indices of nearest neigbors
        self.get_knn()
        # predict a target after initialization

    @staticmethod
    def min_max(X: np.ndarray, minmaxvalues: list=None) -> [np.ndarray, list]:
        """
        min-max scaling to [0,1]; column wise (per feature)
        For testset it must use the minmaxvalues calculated for trainset
        :param X: numpy array to be scaled
        :param minmaxvalues: list[min, max]
        :return: scaled version of X
        """
        if minmaxvalues:
            xmin = minmaxvalues[0]
            xmax = minmaxvalues[1]
        else:
            xmin = np.min(X, axis=0)
            xmax = np.max(X, axis=0)
            minmaxvalues = [xmin, xmax]
        X_scaled = (X-xmin) / (xmax - xmin)
        return X_scaled, minmaxvalues

    def distance_calc(self) -> None:
        """
        Calculate the euclidean distance between the samples of train set and test set
        Then the distances are ranked. The first col represents the index of the closest distance, etc...
        And one row per test sample.
        :return:
        """
        # rows = samples X_train; columns = samples X_test
        distance_data = distance.cdist(self.X_train, self.X_test, 'euclidean')
        # ranked distance; sample in rows; closest=lower colnr! --> store for re-use for any k
        self.distance_order = np.argpartition(distance_data.T, self.k)

    def get_knn(self) -> None:
        """
        Retrieve the k target values per sample
        :return:
        """
        n = len(self.X_test)
        # get indices of the k nearest neighbors (shape: # test samples x k)
        k_idx = self.distance_order[:, :self.k]
        # select target values
        self.nearest_neighbors = self.y_train[k_idx].reshape(n, self.k)

    def predict_target(self, mtype: str='classification') -> np.ndarray:
        """
        classification: calculate proportion of class == 1. If > 0.5 then class 1 is predicted
        :param mtype: to add option 'regression'
        :return: predictions
        """
        if mtype == 'classification':
            prediction = np.sum(self.nearest_neighbors, axis=1)/self.k > 0.5
            prediction = prediction.astype(int)[:, np.newaxis]
        else:
            raise Exception("Please enter a correct type for the prediction")

        return prediction

    def performance(self, prediction: np.ndarray, target: np.ndarray = None, mtype: str = 'classification'):
        """
        If you didn't input y_test while initializing, you can input a target here.
        classification: Calculate proportion of correctly classified samples
        :param prediction: output of self.predict_target
        :param target: y_train (not needed when already inputted while creating the class)
        :param mtype: to add option 'regression'
        :return:
        """
        if self.y_test is not None:
            target = self.y_test

        if mtype == 'classification':
            accuracy = np.mean(prediction == target)
        else:
            raise Exception("Please enter a correct type for the performance")

        return accuracy
