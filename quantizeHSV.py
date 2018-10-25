import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from scipy import misc
import scipy.misc
from skimage import color


def quantizeHSV(origImage, k):
    image_hsv = color.rgb2hsv(plt.imread(origImage))
    width, height, channels = image_hsv.shape
    two_dim = image_hsv.reshape(width*height, channels)
    only_h = image_hsv[:,:,0]
    only_h = only_h.reshape((width*height, 1))
    kmeans_cluster = cluster.KMeans(n_clusters=k)
    kmeans_cluster.fit(only_h)
    meanColors = kmeans_cluster.cluster_centers_
    outputImage=np.empty([width*height,channels], dtype=float)
    for num in range(k):
        for i, x in enumerate(kmeans_cluster.labels_):
            if x == num:
                val = np.append(np.array(meanColors[x]), two_dim[i][1:])
                outputImage[i] = val
    outputImage=outputImage.reshape((width,height,channels))
    finalImage = color.hsv2rgb(outputImage)
    finalImage= finalImage*255

    return (finalImage.astype(int), meanColors)
#quantizeHSV('me.jpg', 7)
