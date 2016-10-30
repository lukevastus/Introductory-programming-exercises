import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

#Implementation of K-Means algorithm
train_data = np.random.randint(0, 1000, (1000, 3))


def find_distance(point1, point2):
    return math.sqrt(np.sum((point1 - point2) ** 2))


def assign_center(points, centers):
    for i in range(points.shape[0]):
        center_distance = np.zeros(centers.shape[0])
        for j in range(centers.shape[0]):
            center_distance[j] = find_distance(points[i, :-1], centers[j, :])
        points[i, -1] = np.argmin(center_distance)


def find_centroid(points):
    centroid = np.mean(points, axis=0)[:-1]
    return centroid


def kmeans_2d(data, k, iterations=100, plot=False, saveplot=False):
    centers = np.random.randint(0, np.amax(data), (k, data.shape[1] - 1))
    i = 0
    old_centers = np.zeros(centers.shape)
    new_centers = np.ones(centers.shape)

    while not np.array_equal(old_centers, new_centers):
        if i == iterations:
            break
        assign_center(data, centers)

        if saveplot:
            plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=45, cmap="rainbow", edgecolors="none", alpha=0.75)
            plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=150)
            plt.savefig("Iteration_%d.png" %i)
            plt.clf()

        old_centers = centers.copy()
        for j in range(k):
            new_centers[j, :] = find_centroid(data[data[:, -1] == j])

        centers = new_centers.copy()
        i += 1

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection="3d")
    #ax.scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], zdir="z", c=data[:, 3], cmap="rainbow", edgecolors="none")

    if plot:
        plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=45, cmap="rainbow", edgecolors="none", alpha=0.75)
        plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=150)
        #plt.axis("off")
        plt.show()

    return i


print(kmeans_2d(train_data, 5, saveplot=True))
