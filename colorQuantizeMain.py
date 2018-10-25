import matplotlib.pyplot as plt 
import numpy as np
import PIL 
import PIL.ImageOps  
from quantizeRGB import quantizeRGB
from quantizeHSV import quantizeHSV
from computeQuantizationError import computeQuantizationError
from getHueHists import getHueHists

fig,axis = plt.subplots(2,2, figsize =(15,10))
fig.suptitle('Image Transformations')
images =[]

k = 3
rgb_quantized, mean_rgb = quantizeRGB('fish.jpg',k)
images.append(axis[0,0].imshow(rgb_quantized, aspect ='auto'))
axis[0,0].set_title('RGB Quantized')

hsv_quantized, mean_hsv = quantizeHSV('fish.jpg',k)
plt.imread('fish.jpg').tofile("fish.txt", sep=',')
images.append(axis[0,1].imshow(hsv_quantized, aspect ='auto'))
axis[0,1].set_title('HSV Quantized')

print(computeQuantizationError('fish.jpg', rgb_quantized))
print(computeQuantizationError('fish.jpg', hsv_quantized))

histEqual, histClustered= getHueHists('fish.jpg', k)

equal_hist, equal_bin_edges = histEqual
images.append(axis[1,0].bar(equal_bin_edges[:-1], equal_hist, width=0.1))
axis[1,0].set_title('Histogram Equally Spaced Bins')

cluster_hist, cluster_bin_edges = histClustered
images.append(axis[1,1].bar(cluster_bin_edges[:-1], cluster_hist, width=0.1))
axis[1,1].set_title("Histogram Based on Clusters")
#f1 = plt.figure()
#f2 = plt.figure()
#ax1 = f1.add_subplot(111)
#ax1.plot(values, bins=bins)
#ax2 = f2.add_subplot(111)
#axis[1,1].hist(values_clust, bins=bins_clust)
plt.show()
#images.append(axis[1,0].imshow(histEqual, aspect ='auto'))
#axis[1,0].set_title('Histogram equal bins')
#images.append(axis[1,1].imshow(histClustered, aspect ='auto'))
#axis[1,1].set_title('Histogram clustered bins')

