import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import time
import sys


if __name__ == '__main__':
    img_path = sys.argv[1]

    img = cv.imread(img_path)
    intensity_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Grey", intensity_img)
    cv.waitKey(0)
#    =========================================================================    
#    ==================== Task 1 =================================
#    =========================================================================    
    print('Task 1:');
    # a
    Integralimage_1 = intensity_img.copy()  
    for row in range(intensity_img.ndim):
        Integralimage_1 = Integralimage_1.cumsum(axis=row)
    cv.imshow("Integral Image",Integralimage_1);
    cv.waitKey(0);

    # b
    row = Integralimage_1.shape[0]
    col = Integralimage_1.shape[1]
    pixel_num = row * col
    Integralimage = intensity_img.copy()
    ## i
    def b_i(img, row, col):
        sum_of_grey = 0
        for r in range(0,row):
            for c in range(0,col):
                sum_of_grey += img[r,c]
        pixel_num = row*col  
        mean_grey = sum_of_grey // pixel_num  
        return mean_grey
    print("(i)Mean Grey Value: ", b_i(Integralimage, row, col))
    ## ii
    def b_ii(img, row, col):
        Integralimage = cv.integral(img)
        pixel_num = row*col
        mean_grey = Integralimage[-1][-1] // pixel_num
        return mean_grey
    print("(ii)Mean Grey Value: ", b_ii(Integralimage, row, col))
    ## iii
    def b_iii(img, row, col):
        Integralimage = img
        for r in range(img.ndim):
            Integralimage = Integralimage.cumsum(axis=r)
        pixel_num = row*col
        mean_grey = Integralimage[-1][-1] // pixel_num
        return mean_grey
    print("(iii)Mean Grey Value: ", b_iii(Integralimage, Integralimage.shape[0], Integralimage.shape[1]))

    # c 
    ## crop 10 random images
    random_crop = np.zeros([10,100,100])
    for i in range(0, 10):
        square_x, square_y = random.randint(0, 200), random.randint(0, 330)
        random_crop[i] = intensity_img[square_x:square_x+100, square_y:square_y+100]
    
    ## runtime i
    start = time.time()
    for i in random_crop:
        b_i(i, i.shape[0], i.shape[1])
    end = time.time()
    print("(i)duration: ", end-start, "(s)")

    ## runtime ii
    start = time.time()
    for i in random_crop:
        b_ii(i, i.shape[0], i.shape[1])
    end = time.time()
    print("(ii)duration: ", end-start, "(s)")

    ## runtime iii
    start = time.time()
    for i in random_crop:
        b_iii(i, i.shape[0], i.shape[1])
    end = time.time()
    print("(iii)duration: ", end-start, "(s)")




#    =========================================================================    
#    ==================== Task 2 =================================
#    =========================================================================    
    print('Task 2:');
    
    equalHist = cv.equalizeHist(intensity_img)
    res = np.hstack((intensity_img,equalHist))
    # n, bins, patches = plt.hist(equalHist)
    # plt.show()
    cv.imshow("Equalized Histogram", res)
    cv.waitKey(0)





#    =========================================================================    
#    ==================== Task 4 =================================
#    =========================================================================    
    print('Task 4:');
    ## a
    blur_1 = cv.GaussianBlur(intensity_img,(5, 5), 2**(3/2))
    kernel_1 = cv.getGaussianKernel(ksize=5, sigma=2**(3/2))
    # print(blur_1)
    ## b
    def gkern(l=5, sig=1.):
        """
        creates gaussian kernel with side length l and a sigma of sig
        """

        ax = np.arange(-l // 2 + 1., l // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

        return kernel / np.sum(kernel)

    kernel_2 = gkern(l=5, sig=2**(3/2))
    blur_2 = cv.filter2D(intensity_img, -1, kernel_2)
    # print(np.array_equal(kernel_1*kernel_1.T, kernel_2))
    # print(np.subtract(blur_1, blur_2).max())  
    ## c 
    def get1dkernel(filter_length=5,sigma=1.):
        result = np.zeros( filter_length )
        mid = filter_length//2
        result=[(1/(sigma*np.sqrt(2*np.pi)))*(1/(np.exp((i**2)/(2*sigma**2)))) for i in range(-mid,mid+1)]  
        return result / np.sum(result)
    kernel_3 = get1dkernel(sigma=2**(3/2))
    blur_3 = cv.sepFilter2D(intensity_img,-1, kernel_3, kernel_3)


    print("Max difference btwn a, b", np.absolute(np.subtract(blur_1, blur_2)).max())
    print("Max difference btwn b, c", np.absolute(np.subtract(blur_2, blur_3)).max())
    print("Max difference btwn c, a", np.absolute(np.subtract(blur_1, blur_3)).max())



    blur = np.vstack((blur_1, blur_2, blur_3))
    cv.imshow("Blur", blur)
    cv.waitKey(0)

#    =========================================================================    
#    ==================== Task 6 =================================
#    =========================================================================    
    print('Task 6:');
    
    filter_twice = cv.GaussianBlur(intensity_img,(5, 5), 2)
    filter_twice = cv.GaussianBlur(filter_twice,(5, 5), 2)
    filter_once = cv.GaussianBlur(intensity_img,(5, 5), 2**(3/2))
    comparison = np.hstack((filter_twice, filter_once))
    print("Maximum Difference", np.absolute(np.subtract(filter_once , filter_twice)).max())
    cv.imshow("Multiple Gaussian Filters", comparison)
    cv.waitKey(0)


#    =========================================================================    
#    ==================== Task 7 =================================
#    =========================================================================    
    print('Task 7:');
    img = cv.imread(img_path)
    intensity_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    def sp_noise(image,prob):
        '''
        Add salt and pepper noise to image
        prob: Probability of the noise
        '''
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output
    img_noise = sp_noise(intensity_img, 0.3)
    cv.imshow("Salt-and-Pepper", img_noise)
    cv.waitKey(0)

    img_filtGau = cv.GaussianBlur(img_noise,(5, 5), 2)
    img_filtMedian = cv.medianBlur(img_noise, 5)
    img_filtBi = cv.bilateralFilter(img_noise, 5, 75,75)
    cv.imshow("Filter Comparison", np.vstack((img_filtGau, img_filtMedian, img_filtBi)))
    cv.waitKey(0)


#    =========================================================================    
#    ==================== Task 8 =================================
#    =========================================================================    
    print('Task 8:');
    img = cv.imread(img_path)
    intensity_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel_1 = np.array([[0.0113, 0.0838, 0.0113],
                         [0.0838, 0.6193, 0.0838],
                         [0.0113, 0.0838, 0.0113]])
    
    kernel_2 = np.array([[-0.8984, 0.1472, 1.1410],
                         [-1.9075, 0.1566, 2.1359],
                         [-0.8659, 0.0537, 1.0337]])

    filt_1 = cv.filter2D(intensity_img, -1, kernel_1)
    filt_2 = cv.filter2D(intensity_img, -1, kernel_2)

    cv.imshow("Given Filter Comparison", np.hstack((filt_1, filt_2)))
    cv.waitKey(0)


    _, U_1, _ = cv.SVDecomp(kernel_1)
    _, U_2, _ = cv.SVDecomp(kernel_2)
    
    filt_1 = cv.filter2D(intensity_img, -1, U_1)
    filt_2 = cv.filter2D(intensity_img, -1, U_2)
    
    cv.imshow("SVD Filter Comparison", np.hstack((filt_1, filt_2)))
    cv.waitKey(0)
    print("Maximum Difference", np.absolute(np.subtract(filt_1, filt_2)).max())