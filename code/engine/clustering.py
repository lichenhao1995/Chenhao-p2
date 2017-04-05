"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Clustering - Skeleton
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from utils import clustering_utils
import model.clustering_classes as ccs
import random

def build_face_image_points(X, y):
    """
    Input:
        X : (n,d) feature matrix, in which each row represents an image
        y: (n,1) array, vector, containing labels corresponding to X
    Returns:
        List of Points
    """
    (n, d) = X.shape
    images = {}
    points = []
    for i in range(0, n):
        if y[i] not in images.keys():
            images[y[i]] = []
        images[y[i]].append(X[i, :])
    for face in images.keys():
        count = 0
        for im in images[face]:
            points.append(ccs.Point(str(face) + '_' + str(count), face, im))
            count = count + 1

    return points


def random_init(points, k):
    """
    Input:
        points: a list of point objects
        k: Number of initial centroids/medoids
    Returns:
        List of k unique points randomly selected from points
    """
    return random.sample(points, k)


def k_means_pp_init(points, k):
    """
    Input:
        points: a list of point objects
        k: Number of initial centroids/medoids
    Returns:
        List of k unique points randomly selected from points
    """
    #random select one point as the first centroid
    centroids = random.sample(points, 1)  
    for i in range(k - 1):
        p = [min([pt.distance(centroid) for centroid in centroids]) ** 2 for pt in points] 
        probability = p / np.sum(p)
        centroids.append(np.random.choice(points, 1, p = probability)[0])
    return centroids


def k_means(points, k, init='random'):
    """
    Input:
        points: a list of Point objects
        k: the number of clusters we want to end up with
        init: The method of initialization, takes two valus 'cheat'
              and 'random'. If init='cheat', then use cheat_init to get
              initial clusters. If init='random', then use random_init
              to initialize clusters. Default value 'random'.

    Clusters points into k clusters using k_means clustering.

    Returns:
        Instance of ClusterSet corresponding to k clusters
    """
    pointList = [[] for _ in range(k)]
    if init == 'random':
        centroids = random_init(points, k)
    else:
        centroids = k_means_pp_init(points, k)
    resultSet = ccs.ClusterSet()
    temp = []
    for point in points:
        disList = []
        for i in range(k):
            disList.append(centroids[i].distance(point))
        temp.append(disList.index(min(disList)))
    for i in range(len(temp)):
        pointList[temp[i]].append(points[i])
    for entry in pointList:
        resultSet.add(ccs.Cluster(entry))
    return resultSet


def plot_performance(k_means_Scores, kpp_Scores, k_vals):
    """
    Input:
        KMeans_Scores: A list of len(k_vals) average purity scores from running the
                       KMeans algorithm with Random Init
        KPP_Scores: A list of len(k_vals) average purity scores from running the
                    KMeans algorithm with KMeans++ Init
        K_Vals: A list of integer k values used to calculate the above scores

    Uses matplotlib to generate a graph of performance vs. k
    """
    plt.plot(k_vals, k_means_Scores, '--', label = 'KPP') 
    plt.plot(k_vals, kpp_Scores, '--', label = 'KMeans') 
    plt.xlabel('K') 
    plt.ylabel('Performance')
    plt.legend(loc = 'upper left')
    plt.show()


def main():
    X, y = clustering_utils.get_data()
    points = build_face_image_points(X, y)

    numOfK = 10
    kPPScoresList = [0] * numOfK
    kMeansScoresList = [0] * numOfK
    kPPMaxList = [0] * numOfK
    kMeansMaxList = [0] * numOfK
    kPPMinList = [0] * numOfK
    kMeansMinList = [0] * numOfK
    kList = [i for i in range(1, numOfK + 1)]
    for it in range(numOfK):
        print("Loop ", it, ":" )
        for i, k in enumerate(kList):
            print()
            print("k is ", k, ": ")
            kMeansClusters = k_means(points, k)
            score = kMeansClusters.get_score()
            kMeansScoresList[i] += score
            #For part(d) min and max score
            if it == 0:
                kMeansMinList[i] = score
                kMeansMaxList[i] = score
            kMeansMinList[i] = min(score, kMeansMinList[i]) 
            kMeansMaxList[i] = max(score, kMeansMaxList[i])
            kPPClusters = k_means(points, k, 'cheat')
            score = kPPClusters.get_score()
            kPPScoresList[i] += score
            #For part(d) min and max score
            if it == 0:
                kPPMinList[i] = score
                kPPMaxList[i] = score
            kPPMinList[i] = min(score, kPPMinList[i]) 
            kPPMaxList[i] = max(score, kPPMaxList[i])
    for n in range(numOfK):
        kMeansScoresList[n] /= numOfK
        kPPScoresList[n] /= numOfK
    plot_performance(kMeansScoresList, kPPScoresList, kList)

    best_kMeans_k = np.argmax(kMeansScoresList)
    print('***For KMEANS*** best k:', best_kMeans_k + 1, 'avg:', kMeansScoresList[best_kMeans_k], 
        'min:', kMeansMinList[best_kMeans_k], 'max:', kMeansMaxList[best_kMeans_k])  
    best_kpp_k = np.argmax(kPPScoresList)
    print('***For KPP*** best k:', best_kpp_k + 1, 'avg:', kPPScoresList[best_kpp_k], 
        'min:', kPPMinList[best_kpp_k], 'max:', kPPMaxList[best_kpp_k])

if __name__ == '__main__':
    main()
