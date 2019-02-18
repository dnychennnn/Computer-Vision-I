"""
Information

Content: Computer Vision I Sheet00
Author: Yung-Yu Chen
Matrikelnummer: 3192698
E-mail: yung-yu.chen@uni-bonn.de
Group member: not found yet(only me)


"""

import cv2 as cv
import numpy as np
import random
import sys

if __name__ == '__main__':
    img_path = sys.argv[1]

    # 2a: read and display the image
    img = cv.imread(img_path)
    cv.imshow('image', img)
    cv.waitKey()

    # 2b: display the intenstity image
    intensity_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Intensity Image', intensity_img)
    cv.waitKey(0)
    
    # 2c: for loop to perform the operation
    img = np.array(img)
    intensity_img = np.array(intensity_img) * 0.5
    after_img = np.zeros(img.shape, dtype=np.uint8)

    """ for-loop operation """
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            for channel in range(img.shape[2]):
                after_img[row][col][channel] = max(img[row][col][channel] - intensity_img[row][col], 0)
    print(after_img.dtype)

    cv.imshow('After', after_img)
    cv.waitKey(0)
        

    # 2d: one-line statement to perfom the operation above
    after_img_oneline = np.zeros(img.shape, dtype=np.uint8)

    """ one line operation """
    after_img_oneline = np.maximum(img - np.expand_dims(intensity_img, axis=3), 0).astype(np.uint8)

    cv.imshow("After(One Line)", after_img_oneline)
    cv.waitKey(0)

    # 2e: Extract a random patch
    image = img.copy()
    """ extract patch """
    patch_center = np.array([215, 150])
    patch_size = 16
    print(patch_size)
    patch_x = int(patch_center[0] - patch_size/2)
    patch_y = int(patch_center[1] - patch_size/2)
    patch_image = image[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size]
    """ copy to image at random position """
    x = random.randint(0, image.shape[0]-patch_size)
    y = random.randint(0, image.shape[1]-patch_size)
    image[x:x+patch_size, y:y+patch_size] = patch_image

    cv.imshow("Copy Patch Image", image)
    cv.waitKey(0)


    # 2f: Draw random rectangles and ellipses
    width = 100 
    height = 50
    random = random.Random()

    def random_color(random):
        """
        Return a random color
        """
        icolor = random.randint(0, 0xFFFFFF)
        return (icolor & 0xff, (icolor >> 8) & 0xff, (icolor >> 16) & 0xff)
    """ Draw 10 random rectangles """
    for i in range(10):
        pt1 =  (random.randrange(-width, 2 * width),
                          random.randrange(-height, 2 * height))
        pt2 =  (random.randrange(-width, 2 * width),
                          random.randrange(-height, 2 * height))
        cv.rectangle(img, pt1, pt2,
                        random_color(random),
                        -1)
    cv.imshow("Draw", img)
    cv.waitKey(0)

    # draw ellipses
    """ Draw 10 random ellipses """
    width = 215  
    height = 150
    for i in range(10):
        pt1 =  (random.randrange(width, 2 * width),
                            random.randrange(height, 2 * height))
        sz =  (random.randrange(0, 200),
                        random.randrange(0, 200))
        angle = random.randrange(0, 1000) * 0.180
        cv.ellipse(img, pt1, sz, angle, angle-100 , angle+280,
                        random_color(random),
                        -1)
            
    cv.imshow("Draw", img)
    cv.waitKey(0)

    # destroy all windows
    cv.destroyAllWindows()
















