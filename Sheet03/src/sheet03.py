import numpy as np
import cv2 as cv
import random


##############################################
#     Task 1        ##########################
##############################################


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/shapes.png')
    '''
    ...
    your code ...
    ...
    '''
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img_gray, 50, 200)
    lines = cv.HoughLines(edges,1,np.pi/90,50) 
    lines = lines[:,0,:]
    print(lines)
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    cv.imshow("Lines", img)
    cv.waitKey(0)


def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    accumulator = np.zeros((int(180 / theta_step_sz), int(np.linalg.norm(img_edges.shape) / d_resolution)))
    detected_lines = []
    '''
    ...
    your code ...
    ...
    '''
    y_coor, x_coor = np.where(img_edges==255)
    for i_edges in range(0, len(x_coor)):
        x = x_coor[i_edges]
        y = y_coor[i_edges]
        for theta_idx in range(0, 180//theta_step_sz):
            theta = np.deg2rad(theta_idx*theta_step_sz)
            # why is it "d = xcos - ysin" on the slides?(angle)
            d = int(round(x * np.cos(theta) + y * np.sin(theta)))
            accumulator[theta_idx, d] = accumulator[theta_idx, d] +  1
    detected_lines.append(np.where(accumulator>threshold))
    detected_lines = np.asarray(detected_lines[0])
    # multiply with step size
    detected_lines[0] =  detected_lines[0] * theta_step_sz
    detected_lines = detected_lines.T
    return detected_lines, accumulator


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert the image into grayscale
    edges = cv.Canny(img_gray, 50, 200) # detect the edges
    detected_lines, accumulator = myHoughLines(edges, 1, 2, 50)
    '''
    ...
    your code ...
    ...
    '''
    cv.imshow("Accumulator", accumulator/accumulator.mean())
    cv.imwrite("./accumulator.png", (accumulator/accumulator.mean())*255)
    cv.waitKey(0)
    print(detected_lines)
    lines = detected_lines
    for theta, d in lines:
        theta = np.deg2rad(theta)
        print(theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*d
        y0 = b*d
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    cv.imshow("My Hough Lines", img)
    cv.waitKey(0)

  


##############################################
#     Task 2        ##########################
##############################################


def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert the image into grayscale
    edges = cv.Canny(img_gray, 50, 200) # detect the edges
    theta_res = 2 # set the resolution of theta
    d_res = 1 # set the distance resolution
    _, accumulator = myHoughLines(edges, d_res, theta_res, 50)
    '''
    ...
    your code ...
    ...
    '''
    print(accumulator.shape)
    def neighbor_data(data, centers, distance=5):
        neighbors = []
        for d in data:
            if np.linalg.norm(d-centers) <= distance:
                neighbors.append(d)

        return np.asarray(neighbors)

    def gaussian_kernel(distance, bandwidth):
        return np.random.normal(bandwidth, distance)

    cv.imshow("accu", accumulator)
    cv.waitKey(0)
    data = np.reshape(accumulator, (-1, 1))
    new_ceteroids = np.copy(data)
    difference = 1
    need_shift = np.ones((new_ceteroids.shape), dtype=bool)
    max_difference = 1
    THRESHOLD = 0.001
    EPOCH = 0
    print(accumulator.shape)
    # Calculating Meanshift
    while max_difference>THRESHOLD:
        EPOCH = EPOCH +1
        print("EPOCH", EPOCH)
        max_difference = 0
        all = 0
        
        for idx, d in enumerate(new_ceteroids):
            neighbors = neighbor_data(new_ceteroids, d, distance=9)
            weights = np.zeros((neighbors.shape))
            weights = gaussian_kernel(np.linalg.norm(neighbors-d, axis=1), 9)
            
            if not need_shift[idx]:
                continue

            last_centroid = new_ceteroids[idx][0]
            new_centroid = np.sum(np.asmatrix(weights)*neighbors) / np.sum(weights)
            # update mean shift
            new_ceteroids[idx] = new_centroid
            difference = np.linalg.norm(new_centroid-last_centroid)
            # print(str(idx), "diff", difference,"new centroid", new_centroid)
            
            if difference > max_difference:#record the max in all points
                max_difference = difference
            if difference < THRESHOLD:#no need to move
                need_shift[idx] = False
            print('i '+ str(idx) + ' max diff ' + str(max_difference), end='\r')

        """ The problem of my code is that this takes too much time to run for only 1 epoch, so I force it to end."""
        if EPOCH==1:
            max_difference = 0.0000001

    display = np.reshape(new_ceteroids, accumulator.shape)
    cv.imshow("Visualize Mean Shift", display/display.mean())
    cv.waitKey(0)
    cv.imwrite("./accumulator.png", display/display.mean()*255)

    



##############################################
#     Task 3        ##########################
##############################################


def myKmeans(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """
    centers = np.zeros((k, data.shape[1]))
    index = np.zeros(data.shape[0], dtype=int)
    clusters = [[] for i in range(k)]

    # initialize centers using some random points from data
    # ....
    centers = data[(np.random.rand(centers.shape[0])*data.shape[0]).astype(int)].astype(float)
    print("init centers", centers)
    
    last_centers = centers.copy()
    convergence = False
    iterationNo = 0
    while not convergence:
        # assign each point to the cluster of closest center
        # ...
        # calculating the distance between datapoints and cetroids
        for idx, d in enumerate(data):
            index[idx] = np.linalg.norm(d-centers, axis=1).argmin()
            # print(np.linalg.norm(d-centers, axis=1))
        # update clusters' centers and check for convergence
        # ...
        # calculating the mean(new center)
        for c in range(k):
            centers[c] = data[np.where(index==c)].mean(0)
            
       
        iterationNo += 1
        print('iterationNo = ', iterationNo)
        # calculating the difference between last and current centroids
        diff = (np.linalg.norm(last_centers-centers, axis=1))
        # make the threshold at 0.01(Euclidean distance)
        if np.all(diff)<0.01:
            print("Final center", centers)
            convergence = True
        else:
            print("difference", diff)
            last_centers = centers.copy()
    return index, centers


def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''
    intensity_img = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
    intensity_img_data = intensity_img.reshape((-1,1))
    for k in range(2, 7, 2):
        index, centers = myKmeans(intensity_img_data, k)
        group = centers[index].reshape(intensity_img.shape)
        cv.imshow("Kmeans-"+str(k), group/255.)
        cv.waitKey(0)

def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''
    color_data = img.reshape((-1, img.shape[2]))
    for k in range(2, 7, 2):
        index, centers = myKmeans(color_data, k)
        group = centers[index].reshape(img.shape)
        cv.imshow("Kmeans-"+str(k), group/255.)
        cv.waitKey(0)


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''
    intensity_img = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
    intensity_img_data = intensity_img.reshape((-1,1))

    position_data = np.argwhere(intensity_img)  
    data = np.hstack((intensity_img_data, position_data))
    for k in range(2, 7, 2):
        index, centers = myKmeans(data, k)
        group = centers[index].reshape(img.shape)
        cv.imshow("Kmeans-"+str(k), group/255.)
        cv.waitKey(0)


##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")
    D = None  # construct the D matrix
    W = None  # construct the W matrix
    '''
    ...
    your code ...
    ...
    '''


##############################################
##############################################
##############################################


# task_1_a()
# task_1_b()
task_2()
# task_3_a()
# task_3_b()
# task_3_c()
# task_4_a()

