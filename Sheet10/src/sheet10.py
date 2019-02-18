
import numpy as np
import os
import cv2 as cv


MAX_ITERATIONS = 1000 # maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
EPSILON = 0.002 # the stopping criterion for the difference when performing the Horn-Schuck algorithm
EIGEN_THRESHOLD = 0.01 # use as threshold for determining if the optical flow is valid when performing Lucas-Kanade

def load_FLO_file(filename):
    
    assert os.path.isfile(filename), 'file does not exist: ' + filename   
    flo_file = open(filename,'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    assert magic == 202021.25,  'Magic number incorrect. .flo file is invalid'
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    #the float values for u and v are interleaved in row order, i.e., u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...,
    # in total, there are 2*w*h flow values
    data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    flo_file.close()
    return flow

#***********************************************************************************
#implement Lucas-Kanade Optical Flow 
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# window_size: the number of points taken in the neighborhood of each pixel
# returns the Optical flow based on the Lucas-Kanade algorithm
def Lucas_Kanade_flow(frames, Ix, Iy, It, window_size):
	im1 = frames[0]
	im2 = frames[1]
	width = im1.shape[1]
	height = im1.shape[0]
	opfl = np.zeros((height, width, 2))

	for y in range(window_size//2, height - window_size//2  ):
		for x in range(window_size//2 , width - window_size//2 ):

			neighbours = [(Y, X) for X in range(x - window_size//2, x + window_size//2 +1 ) for Y in range(y - window_size//2, y + window_size//2 +1)]
			
			A = np.matrix([[
				Ix[p],
				Iy[p]
				] for p in neighbours
			])

			b = np.matrix([[
				It[p]
				] for p in neighbours
			])
			
			"""[unsolved]my threhold create a back result"""
			# sort out the value by eigen threshold
			# eig_val, _ = np.linalg.eig(A.T@A)
			# if max(eig_val) - min(eig_val) < EIGEN_THRESHOLD: 
				
			v = np.linalg.inv(A.T@A) @ A.T @ b
			opfl[y,x][0] = v[0,0]
			opfl[y,x][1] = v[1,0]
			
	return opfl
#***********************************************************************************
#implement Horn-Schunck Optical Flow 
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direc9tion
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# window_size: the number of points taken in the neighborhood of each pixel
# returns the Optical flow based on the Horn-Schunck algorithm
def Horn_Schunck_flow(Ix, Iy, It):
    i = 0
    diff = 1
    while i<MAX_ITERATIONS and diff > EPSILON: #Iterate until the max number of iterations is reached or the difference is less than epsilon
        i += 1
        pass

#calculate the angular error here
def calculate_angular_error(estimated_flow, groundtruth_flow):
	estimated_flow_u = estimated_flow[...,0]
	estimated_flow_v = estimated_flow[...,1]
    
	groundtruth_flow_u = groundtruth_flow[...,0]
	groundtruth_flow_v = groundtruth_flow[...,1]

	top = 1.0 + estimated_flow_u*groundtruth_flow_u + estimated_flow_v*groundtruth_flow_v
	bottom = np.sqrt(1.0 +estimated_flow_u**2 + estimated_flow_v**2)*np.sqrt(1.0 +groundtruth_flow_u**2 + groundtruth_flow_v**2)
	return np.mean(np.arccos(top / bottom)) 


#function for converting flow map to to BGR image for visualisation
def flow_map_to_bgr(flow_map):
	height, width = flow_map.shape[:2]

	u = flow_map[...,0]
	v = flow_map[...,1]

	ang = np.arctan2(u, v) + np.pi
	mag = np.sqrt(u*u+v*v)
	hsv = np.zeros((height,width,3), np.uint8) # having hsv channel
	hsv[...,0] = ang*(180/np.pi/2)
	hsv[...,1] = 255
	hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
	bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
	cv.imshow('opticalhsv',bgr)
	cv.waitKey(0)

	pass


if __name__ == "__main__":
    # read your data here and then call the different algorithms, then visualise your results
	WINDOW_SIZE = [15, 15]  #the number of points taken in the neighborhood of each pixel when applying Lucas-Kanade
	gt_flow = load_FLO_file('../data/groundTruthOF.flo')
	
	frame1 = cv.imread("../data/frame1.png")

	frame2 = cv.imread("../data/frame2.png")
	frame1 = cv.cvtColor(frame1, cv.CV_64F).astype(np.float64)
	frame2 = cv.cvtColor(frame2, cv.CV_64F).astype(np.float64)
	
	frames = []
	frames.append(frame1)
	frames.append(frame2)
	frames = np.array(frames)

	Ix = np.zeros(frame1.shape)
	Iy = np.zeros(frame1.shape)
	It = np.zeros(frame1.shape)

	Ix = cv.Sobel(frame1, cv.CV_64F, 1, 0) 
	Iy = cv.Sobel(frame1, cv.CV_64F, 0, 1) 
	It = (frame1 - frame2).astype(np.float64) 

	norm_Ix = cv.normalize(Ix, None,0,1,cv.NORM_MINMAX)
	norm_Iy = cv.normalize(Iy, None,0,1,cv.NORM_MINMAX)
	norm_It = cv.normalize(It, None,0,1,cv.NORM_MINMAX)


	

	# perform Lucas Kanadeflow
	print('--------------Lucas Kanade flow--------------')
	op_flow = Lucas_Kanade_flow(frames, norm_Ix, norm_Iy, norm_It, 15)
	flow_map_to_bgr(op_flow)
	print("Angualr Error: ", calculate_angular_error(op_flow, gt_flow))


	# visualize ground truth
	print('--------------ground truth flow--------------')
	flow_map_to_bgr(gt_flow)
	pass
