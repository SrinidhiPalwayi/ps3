import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from scipy import misc
import scipy.misc
from skimage import color
import collections

def getHueHists(im, k):
    print(" hue hists")
    image_hsv = color.rgb2hsv(plt.imread(im))
    width, height, channels = image_hsv.shape
    print(width, height)
    only_h = image_hsv[:,:,0].reshape(width*height,1)
    histEqual = np.histogram(only_h, bins=k)
    print(k)
    kmeans_cluster = cluster.KMeans(n_clusters=k)
    kmeans_cluster.fit_predict(only_h)
    meanColors = kmeans_cluster.cluster_centers_.flatten()
    clustered_list= list()
    meanColors = list(meanColors)
    meanColors.sort()
    better_means = list()
    for x in range(len(meanColors)-1):
        better_means.append( (meanColors[x] + meanColors[x+1])/2.0)

    better_means = [0]  + better_means+[1]
    histClustered = np.histogram(only_h, bins=better_means)
    return (histEqual,histClustered)
#getHueHists('fish.jpg', 10)
