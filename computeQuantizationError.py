import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from scipy import misc
import scipy.misc
from skimage import color

def computeQuantizationError(origImage, quantizedImg):
    start = plt.imread(origImage)
    start = np.array(start, dtype='float')
    end = quantizedImg
    diff = np.subtract(start,end)
    diff = diff.flatten()
    square = np.power(diff,2)
    return np.sum(square)
    

