import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from scipy import misc
import scipy.misc
fig = plt.figure()
def quantizeRGB(origImage, k):
    image_array = np.asarray(plt.imread(origImage))
    width, height, channels = image_array.shape
    two_dim = image_array.reshape(width*height, channels)
    kmeans_cluster = cluster.KMeans(n_clusters=k)
    kmeans_cluster.fit(two_dim)
    outputImage=np.empty([width*height,channels], dtype=int)
    for num in range(k):
        for i, x in enumerate(kmeans_cluster.labels_):
            if x == num:
                outputImage[i] = kmeans_cluster.cluster_centers_[x]
    outputImage=outputImage.reshape((width,height,channels))
    return (outputImage, kmeans_cluster.cluster_centers_)

