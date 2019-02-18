"""
Title: CV I Sheet09
Author: Yung-Yu Chen(1 man group)
Stud ID: 3192698
"""



import cv2
import numpy as np
import random

#   =======================================================
#                   Task1
#   =======================================================

img1 = cv2.imread('../images/building.jpeg')
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray) 


# compute structural tensor

Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

Ixy = Iy*Ix
Ixx = Ix**2
Iyy = Iy**2

Ixy = cv2.GaussianBlur(Ixy,(0,0), 2)
Ixx = cv2.GaussianBlur(Ixx,(0,0), 2)
Iyy = cv2.GaussianBlur(Iyy,(0,0), 2)


#Harris Corner Detection

Mdet = (Ixx*Iyy) - Ixy**2
Mtrace =Ixx + Iyy
k=.05
R = Mdet - k*(Mtrace)**2
Rmax = np.amax(R)

width = gray.shape[1]
height = gray.shape[0]

img_harris = np.copy(img1)
img_foestner = np.copy(img1)

harris_response = []

for y in range(height-1):
    for x in range(width-1):
        if R[y,x] > 0.01 * Rmax and R[y,x] > R[y-1, x-1] and R[y,x] > R[y-1, x+1] and R[y,x] > R[y+1, x-1] and R[y,x] > R[y+1, x+1]:
                img_harris[y,x] = [0,0,255]

cv2.imshow("harris corner", img_harris)
cv2.waitKey(0)  

#Forstner Corner Detection

w = Mdet / Mtrace
q = 4*Mdet / (Mtrace**2)

w_min = 1.5
q_min = .75


for y in range(1, height-1):
    for x in range(1, width-1):
        if w[y,x] > w_min and q[y,x]>q_min and w[y,x] > w[y-1, x-1] and w[y,x] > w[y-1, x+1] and w[y,x] > w[y+1, x-1] and w[y,x] > w[y+1, x+1] and w[y,x] > w[y, x+1] and w[y,x] > w[y, x-1] and w[y,x] > w[y+1, x] and w[y,x] > w[y-1, x]:
                img_foestner[y,x] = [0,0,255]

## Problem: Response detected on the cloud and lawn???
cv2.imshow("foestner corner", img_foestner)
cv2.waitKey(0)
#   =======================================================
#                   Task2
#   =======================================================

img1 = cv2.imread('../images/mountain1.png')
img2 = cv2.imread('../images/mountain2.png')

#extract sift keypoints and descriptors
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# own implementation of matching
bf = cv2.BFMatcher()
good_matches = bf.knnMatch(des1,des2, k=2)

good = []
for m,n in good_matches:
    if m.distance/n.distance < 0.4:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)


# display matched keypoints
cv2.imshow("matches", img3)
cv2.waitKey(0)


#  =======================================================
#                          Task-3                         
#  =======================================================

nSamples = 4;
nIterations = 50;
thresh = .1;
minSamples = 4;

max_inliers = 0
M_best = []

kp_img1 = []
kp_img2 = []

for all_matches in good_matches:
    img1_idx = all_matches[0].queryIdx
    img2_idx = all_matches[0].trainIdx
    kp_img1.append(kp1[img1_idx].pt)
    kp_img2.append(kp2[img2_idx].pt)
kp_img1 = np.array(kp_img1)
kp_img2 = np.array(kp_img2)


#  /// RANSAC loop
for i in range(nIterations):

    print('iteration '+str(i))
    
    #randomly select 4 pairs of keypoints
    matches = np.array(random.sample(good_matches, nSamples))[:,0]
    #compute transofrmation and warp img2 using it
    rand_kp1 = []
    rand_kp2 = []
    for m in matches:
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx
        rand_kp1.append(kp1[img1_idx].pt)
        rand_kp2.append(kp2[img2_idx].pt)
    
    rand_kp1 = np.array(rand_kp1, dtype=np.float32)
    rand_kp2 = np.array(rand_kp2, dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(rand_kp2, rand_kp1)
    Hp = cv2.perspectiveTransform(kp_img2[None,:,:], M)[0]

    #count inliers and keep transformation if it is better than the best so far
    diff = Hp - kp_img1
    residuals = np.linalg.norm(diff, axis=1) 

    inliers_count = np.count_nonzero((residuals<4.5).astype(np.bool)) 
    if inliers_count > max_inliers:
        max_inliers = inliers_count
        # check if max inliers is larger than threshold 
        if max_inliers > kp_img1.shape[0]*thresh:
            M_best = M


print("Max inliers:", max_inliers)
# Check
if M_best == []:
        print("Best Model not found!, Please run again.")
        quit()
#apply best transformation to transform img2 
warped = cv2.warpPerspective(img2, M_best, (img1.shape[1], img1.shape[0]))

#display stitched images
black_index = np.where(np.all(warped==[0,0,0], axis=-1))
warped[black_index] = img1[black_index]
cv2.imshow("Warp", warped)
cv2.waitKey(0)  

