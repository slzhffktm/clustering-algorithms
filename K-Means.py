import math
import random

import numpy as np
from scipy.stats import mode
from sklearn import datasets
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.metrics import accuracy_score

class KMeans(object):
    
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = list(np.array(self.init_empty_clusters()))
        self.centroids = []
        self.labels = []
        
    
    def init_empty_clusters(self):
        return [[] for i in range(self.n_clusters)]
        
        
    def init_centroids(self, X):
        self.centroids = random.sample(list(X), self.n_clusters)
    
    
    def euclidean_distance(self, X, centroid):
        return [sum((x - centroid) ** 2) ** (1/2) for x in X]

    
    def find_data_clusters(self, X):
        distances = np.vstack([self.euclidean_distance(X, centroid) for centroid in self.centroids])
        
        return np.argmin(distances, axis=0)
    
    
    def generate_new_clusters(self, X, data_clusters):
        new_clusters = self.init_empty_clusters()
        for data_cluster, x in zip(data_clusters, X.tolist()):
            new_clusters[data_cluster].append(x)
        
        return [np.array(new_cluster) for new_cluster in new_clusters]
        
    
    def converged(self, new_clusters):
        # stop if new clusters is the same as previous clusters
        for old_cluster, new_cluster in zip(self.clusters, new_clusters):
            if not np.array_equal(old_cluster, new_cluster):
                return False
        
        return True
        
    
    def update_centroids(self):
        self.centroids = [sum(cluster) / len(cluster) for cluster in self.clusters]
        
        
    def fit(self, X):
        self.init_centroids(X)
        
        for _ in range(self.max_iter):
            data_clusters = self.find_data_clusters(X)
            new_clusters = self.generate_new_clusters(X, data_clusters)
            
            if self.converged(new_clusters):
                print('Already converged. Stopping.')
                break
            
            self.clusters = new_clusters
            self.update_centroids()
            
        self.labels = self.find_data_clusters(X)


def replace_labels(pred_labels):
    dict_replace = {
        mode(pred_labels[:50]).mode[0]: 0,
        mode(pred_labels[50:100]).mode[0]: 1,
        mode(pred_labels[100:]).mode[0]: 2
    }
    pred_labels = np.array([dict_replace[label] for label in pred_labels])
    
    return pred_labels


def main():
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target

    clf = KMeans(3)
    clf.fit(X)
    clf.labels

    clf_sklearn = SKLearnKMeans(3, init='random', n_init=1)
    clf_sklearn.fit(X)
    clf_sklearn.labels_

    clf_labels = replace_labels(clf.labels)
    clf_sklearn_labels = replace_labels(clf_sklearn.labels_)

    print('total same cluster with sklearn:', np.sum(clf_labels == clf_sklearn_labels), 'of 150')

    print('created model accuracy: {0:.2f}%'.format(accuracy_score(clf_labels, y)*100))
    print('sklearn model accuracy: {0:.2f}%'.format(accuracy_score(clf_sklearn_labels, y)*100))

if __name__ == "__main__":
    main()
