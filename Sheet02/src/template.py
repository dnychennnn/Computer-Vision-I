"""
Information

Content: Computer Vision I Sheet02
Author: Yung-Yu Chen
Matrikelnummer: 3192698
E-mail: yung-yu.chen@uni-bonn.de
Group member: not found yet(only me)

"""


import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

def get_convolution_using_fourier_transform(image, kernel):
	fr = np.fft.fft2(image)
	delta_w = image.shape[1] - kernel.shape[1]
	delta_h = image.shape[0] - kernel.shape[0]
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)

	# top, bottom = 0, delta_h
	# left, right = 0, delta_w

	fr2 = np.fft.fft2((cv2.copyMakeBorder(kernel, top, bottom, left, right, cv2.BORDER_CONSTANT)))
	m,n = fr.shape
	cc = np.real(np.fft.ifft2(fr*fr2))
	
	# adjust to the center
	cc = np.roll(cc, int(-m/2+1),axis=0)
	cc = np.roll(cc, int(-n/2+1),axis=1)
	return cc

def task1():

	image = cv2.imread('../data/einstein.jpeg', 0)
	kernel = cv2.getGaussianKernel(ksize=7, sigma=1) #calculate kernel
	kernel = np.matrix(kernel*kernel.T)
	print("Gaussian kernel", kernel)
	
	conv_result = cv2.filter2D(image, cv2.CV_64F, kernel) #calculate convolution of image and kernel
	print("conv_result", conv_result)

	fft_result = get_convolution_using_fourier_transform(image, kernel)
	print("ftt_result", fft_result)

	#compare results
	print("mean absolute difference", np.abs(cv2.subtract(conv_result,fft_result)).mean())


def normalized_cross_correlation(image, template):

	k, l = template.shape
	ncc = np.zeros((image.shape[0]-template.shape[0]+1,image.shape[1]-template.shape[1]+1))

	xcorr_template = template-np.mean(template)
	sum_norm_template = np.sum(np.square(xcorr_template))
	
	for row in range(0, image.shape[0]-k):
		for col in range(0, image.shape[1]-l):
			ncc[row, col] = np.sum(np.multiply(xcorr_template , image[row:row+k, col:col+l]-np.mean(image[row:row+k, col:col+l]))) 
			# Normalized
			ncc[row, col] = ncc[row, col] / np.sqrt(sum_norm_template * np.sum(np.square(image[row:row+k, col:col+l]-np.mean(image[row:row+k, col:col+l]))))
			
	return ncc
	

def task2():
	image = cv2.imread('../data/lena.png', 0).astype('float32')
	template = cv2.imread('../data/eye.png', 0).astype('float32')
	result_ncc = normalized_cross_correlation(image, template)

	#draw rectangle around found location in all four results
	#show the results
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_ncc)
	top_left = max_loc
	bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
	cv2.rectangle(image,top_left, bottom_right, 255, 2)
	plt.subplot(121),plt.imshow(result_ncc,cmap = 'gray')
	plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(image,cmap = 'gray')
	plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	fig = plt.figure(1)
	fig.canvas.set_window_title('Template Matching - normalized cross correlation')
	plt.show()
	
	
def build_gaussian_pyramid_opencv(image, num_levels):
	pyramid = [image]
	last_layer = image
	for scale in range(0, num_levels):
		last_layer = cv2.pyrDown(last_layer)
		pyramid.append(last_layer)
		print("gauss_layer_"+str(scale+1), last_layer.shape,last_layer)
		cv2.imshow("(OpenCV)pyramid - layer" + str(scale+1), last_layer)
		cv2.waitKey(0)
	
	return pyramid

def build_gaussian_pyramid(image, num_levels, sigma):
	pyramid = [image]
	last_layer = image
	
	for scale in range(0, num_levels):
		last_layer = cv2.GaussianBlur(last_layer, (0, 0), sigma)
		last_layer = cv2.resize(last_layer, (int(np.ceil(last_layer.shape[1]/2)), int(np.ceil(last_layer.shape[0]/2))))
		pyramid.append(last_layer)
		print("gauss_level_"+str(scale+1), last_layer.shape,last_layer)
		cv2.imshow("(Mine)pyramid - level" + str(scale+1), last_layer)
		cv2.waitKey(0)

	return pyramid

def template_matching_multiple_scales(pyramid_image, pyramid_template, threshold):
	results = []
	level = len(pyramid_image)
	print("# of Level:", level)
	for idx in range(0, level):
		refimg = pyramid_image[-idx]
		tplimg = pyramid_template[-idx]

		# On the first level performs regular template matching.
		# On every other level, perform pyramid transformation and template matching
		# on the predefined ROI areas, obtained using the result of the previous level.
		# Uses contours to define the region of interest and perform TM on the areas.
		if idx == 0:
			result = cv2.matchTemplate(refimg, tplimg, cv2.TM_CCORR_NORMED)
		else:
			mask = cv2.pyrUp(threshed)
			mask8u = cv2.inRange(mask, 0, 255)
			_,contours,_ = cv2.findContours(mask8u, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)

			tH, tW = tplimg.shape[:2]
			for cnt in contours:
				x, y, w, h = cv2.boundingRect(cnt)
				src = refimg[y:y+h+tH, x:x+w+tW]
				result = cv2.matchTemplate(src, tplimg, cv2.TM_CCORR_NORMED)
				# Draw Rectangle
				# (_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)
				# print(minLoc, maxLoc)
				# top_left = maxLoc
				# bottom_right = (top_left[0] + tplimg.shape[1], top_left[1] + tplimg.shape[0])
				# cv2.rectangle(refimg,top_left, bottom_right, 255, 2)
				# plt.subplot(121),plt.imshow(result,cmap = 'gray')
				# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
				# plt.subplot(122),plt.imshow(refimg,cmap = 'gray')
				# plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
				# plt.canvas.set_window_title('Pyramids - level', idx)
				# plt.show()
		T, threshed = cv2.threshold(result, threshold, 1., cv2.THRESH_TOZERO)
		results.append(threshed)

	return threshed


def task3():
	image = cv2.imread('../data/traffic.jpg', 0)
	template = cv2.imread('../data/traffic-template.png', 0)

	cv_pyramid = build_gaussian_pyramid_opencv(image, 4)
	mine_pyramid = build_gaussian_pyramid(image, 4, 1)
	
	#compare and print mean absolute difference at each level
	for i in range(0,4):
		mean_abs_diff = np.mean(np.absolute(cv2.subtract(cv_pyramid[i], mine_pyramid[i])))
		print("mean absolute diﬀerence at layer" + str(i+1), mean_abs_diff)

	# Mine implementation of template matching
	print("Start Calculating NCC")
	start = time.time()
	result_ncc = normalized_cross_correlation(image, template)
	end = time.time()
	print("Time for my normalized cross correlation: ", end - start)
	
	#draw rectangle around found location in all four results
	#show the results

	# Do the template matching by using my implementation of normalized crosscorrelation
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_ncc)
	top_left = max_loc
	bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
	cv2.rectangle(image,top_left, bottom_right, 255, 2)
	plt.subplot(121),plt.imshow(result_ncc,cmap = 'gray')
	plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(image,cmap = 'gray')
	plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	fig = plt.figure(1)
	fig.canvas.set_window_title('Template Matching - normalized cross correlation')
	plt.show()
	
	#  Use pyramids to make template matching faster
	pyramid_template = build_gaussian_pyramid(template, 4, 2)
	start = time.time()
	result = template_matching_multiple_scales(mine_pyramid, pyramid_template, 0.7)
	end = time.time()
	print("Time for pyramids: ", end - start)
	
	#show result
	"""Question: another incomplete rectangle shows up"""
	(_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)
	top_left = maxLoc
	bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
	cv2.rectangle(image,top_left, bottom_right, 255, 2)
	plt.subplot(121),plt.imshow(result,cmap = 'gray')
	plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(image,cmap = 'gray')
	plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	fig = plt.figure(1)
	fig.canvas.set_window_title('Template Matching - pyramids')
	plt.show()
	

def get_derivative_of_gaussian_kernel(size, sigma):

	kernel = cv2.getGaussianKernel(size, sigma)

	kernel2d = kernel * kernel.T
	dx, dy = np.gradient(kernel2d)	
	print("dx", dx)
	print("dy", dy)

	return dx, dy

def task4():
	image = cv2.imread('../data/einstein.jpeg', 0)

	kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)

	edges_x = cv2.filter2D(image, cv2.CV_64F, kernel_x) #convolve with kernel_x
	edges_y = cv2.filter2D(image, cv2.CV_64F, kernel_y) #convolve with kernel_y

	magnitude = np.sqrt(np.square(edges_x) + np.square(edges_y)) #compute edge magnitude
	direction = np.arctan(np.divide(edges_y, edges_x)) #compute edge direction

	cv2.imshow('Magnitude', magnitude)
	cv2.imshow('Direction', direction)
	cv2.waitKey(0)

def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):

	return None
	
def _upscan(f):
	for i, fi in enumerate(f):
		if fi == np.inf: continue
		for j in range(1,i+1):
			x = fi+j*j
			if f[i-j] < x: break
			f[i-j] = x

def distance_transform(bitmap):
    f = np.where(bitmap, 0.0, np.inf)
    for i in range(f.shape[0]):
        _upscan(f[i,:])
        _upscan(f[i,::-1])
    for i in range(f.shape[1]):
        _upscan(f[:,i])
        _upscan(f[::-1,i])
    np.sqrt(f,f)
    return f

def task5():
	image = cv2.imread('../data/traffic.jpg', 0)
	
	edges = cv2.Canny(image, 50, 150) #compute edges
	cv2.imshow('edge', edges)
	cv2.waitKey(0)

	edge_function = None #prepare edges for distance transform


	# dist_transfom_mine = l2_distance_transform_2D(edge_function, positive_inf, negative_inf)
	
	# dist_transform using cv2
	dist_transfom_cv = cv2.distanceTransform(edges, cv2.DIST_L2, 5) #compute using opencv
	print("dist_transfom_cv", dist_transfom_cv)
	
	#compare and print mean absolute difference


task1()
task2()
task3()
task4()
task5()



