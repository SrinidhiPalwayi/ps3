from skimage import feature
import scipy
import matplotlib.pylab as plt
import numpy as np
from skimage import color
from matplotlib.patches import Circle
import scipy.misc

def nonMaximalSuppresion(x,y, houghSpace,ax, radius):
    prev_x = x[0]
    prev_y = y[0]
    max_x = x[0]
    max_y = y[0]
    max_val = 0
    for x_val, y_val in zip(x,y):
        if(abs(x_val-prev_x)<10 and abs(y_val-prev_y)<8):
            #print("came in ", x_val, y_val)

            if(max_val < houghSpace[y_val][x_val]):
                max_x = x_val
                max_y = y_val
                max_val = houghSpace[y_val][x_val]
        else:
            circle = Circle((max_x, max_y),radius,fill=False,edgecolor='red' )
            ax.add_patch(circle)
            max_val=0
        prev_x = x_val
        prev_y= y_val
    circle = Circle((max_x, max_y),radius,fill=False,edgecolor='red' )
    ax.add_patch(circle)
    plt.show()

def multipleCcircles(x,y, houghSpace, ax, radius):
    for x_val, y_val in zip(x,y):
        circle = Circle((x_val, y_val),radius,fill=False,edgecolor='red' )
        ax.add_patch(circle)

def detectCircles(im, radius, useGradient):
    image = color.rgb2gray(plt.imread(im))
    
    grad_x = np.gradient(image, axis=0)
    grad_y = np.gradient(image, axis=1)

    width, height = image.shape
    houghSpace = np.zeros((height, width))
    
    edges = feature.canny(image, sigma=10)
    plt.imshow(edges)
    plt.show()

    for i in range(len(edges)):
        for j in range(len(edges[0])):
            if edges[i][j]:
                if(useGradient):
                    theta = np.arctan2(-grad_y[i][j], grad_x[i][j])
                    a = int(j - radius*np.cos(theta)) #x

                    b = int(i + radius*np.sin(theta)) #y
                    d = int(i + radius*np.sin(-theta)) 
                    

                    if(a < width and b < height):
                        houghSpace[b][a] +=2
                        houghSpace[d][a]+=1
    
                else:
                    if(-radius+i >= 0):
                        min_y = -radius+i
                    else:
                        min_y = 0
                    if(radius+1+i <= len(houghSpace)):
                        max_y = radius+1+i
                    else:
                        max_y = len(houghSpace)
                    for y in range(min_y, max_y):
                        range_x = int((radius**2 - (y-i)**2) **0.5)
                        if(-range_x+j>=0):
                            min_x = -range_x+j
                        else:
                            min_x = 0
                        if(range_x+j+1 <= len(houghSpace[0])):
                            max_x = range_x+j+1
                        else:
                            max_x = len(houghSpace[0])
                        for x in range(min_x, max_x):
                            houghSpace[y][x] = houghSpace[y][x]+1


                """
                if(min_y%2==1):
                    min_y+=1
                if(max_y%2==1):
                    max_y+=1 
                
                vote space quanitzation 
                """
                #add 2 to the step of the range for vote space quantization 
            

                """
                vote space quantization
                
                if(min_x%2==1):
                    min_x+=1
                if(max_x%2==1):
                    max_x+=1
                """
                   #add 2 to the step of the range for vote space quantization 

                    
    houghSpace.tofile("matrix.txt", sep=',')
    plt.imshow(houghSpace)
    plt.show()
    max_val = max(houghSpace.flatten())
    print(max_val)
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(plt.imread(im))
    y,x = np.where(houghSpace >= max_val)
    #x_sorted = [x_ind for _,x_ind in sorted(zip(y,x))]
    #y.sort()
    #print(zip(y,x))
    values = zip(y,x)
    sorted(values , key=lambda k: [k[1], k[0]])
    unzip = zip(*values)
    print(unzip)
    y = list(unzip[0])
    x = list(unzip[1])
    #nonMaximalSuppresion(x,y, houghSpace,ax, radius)
    multipleCcircles(x,y, houghSpace, ax, radius)
    plt.show()

    

   
"""
    for x_val, y_val in zip(x,y):
        circle = Circle((x_val, y_val),radius,fill=False,edgecolor='red' )
        ax.add_patch(circle)
"""
    #print(x,y)
    #plt.imshow(houghSpace)
#plt.show()

#detectCircles("egg.jpg",6, 1)
#for gradient sigma =2
detectCircles("jupiter.jpg", 110, 1)



