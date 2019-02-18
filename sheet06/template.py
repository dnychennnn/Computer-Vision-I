#!/usr/bin/python3.5

import numpy as np
from scipy import misc


'''
    read the usps digit data
    returns a python dict with entries for each digit (0, ..., 9)
    dict[digit] contains a list of 256-dimensional feature vectores (i.e. the gray scale values of the 16x16 digit image)
'''
def read_usps(filename):
    data = dict()
    with open(filename, 'r') as f:
        N = int( np.fromfile(f, dtype = np.uint32, count = 1, sep = ' ') )
        for n in range(N):
            c = int( np.fromfile(f, dtype = np.uint32, count = 1, sep = ' ') )
            tmp = np.fromfile(f, dtype = np.float64, count = 256, sep = ' ') / 1000.0
            data[c] = data.get(c, []) + [tmp]
    for c in range(len(data)):
        data[c] = np.stack(data[c])
    return data

'''
    load the face image and foreground/background parts
    image: the original image
    foreground/background: numpy arrays of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
'''
def read_face_image(filename):
    image = misc.imread(filename) / 255.0
    bounding_box = np.zeros(image.shape)
    bounding_box[50:100, 60:120, :] = 1
    foreground = image[bounding_box == 1].reshape((50 * 60, 3))
    background = image[bounding_box == 0].reshape((40000 - 50 * 60, 3))
    return image, foreground, background



'''
    implement your GMM and EM algorithm here
'''
class GMM(object):

    '''
        fit a single gaussian to the data
        @data: an N x D numpy array, where N is the number of observations and D is the dimension (256 for usps digits, 3 for the skin color RGB pixels)
    '''
    def fit_single_gaussian(self, data):
        #TODO
        self.data = data
        self.lamda = [1]
        self.mu = np.mean(data, axis=0)
        self.mu = self.mu.reshape(len(self.mu), 1)
        self.mu = [self.mu]
        self.std = np.std(data, axis=0)
        self.std = [self.std.reshape(len(self.std), 1)]
        self.covariance = np.cov(data.T, rowvar=True)
        # Make the covariance matrix to be diagonal
        length = len(self.covariance)
        self.covariance = [np.eye(length, length) * np.diagonal(self.covariance)]
        self.covariance = np.expand_dims(np.diag(np.var(data, axis=0)), axis=0)
        
        print(self.covariance.shape)


    '''
        implement the em algorithm here
        @data: an N x D numpy array, where N is the number of observations and D is the dimension (256 for usps digits, 3 for the skin color RGB pixels)
        @n_iterations: the number of em iterations
    '''
    def em_algorithm(self, data, n_iterations = 10):
        #TODO
        k_num = len(self.mu)
        r = np.zeros((len(data), k_num))
        for iter in range(n_iterations):
            # E-step
            for k in range(k_num):
                data_mu = data.T-self.mu[k]
                cov_inv = np.linalg.pinv(self.covariance[k])
                # svd = -0.5 * np.dot(np.dot(data_mu.T, cov_inv), data_mu)
                # because I have memory problem for face, so I split the above to several steps
                svd_1 = np.dot(data_mu.T, cov_inv)
                svd_2 = np.multiply(svd_1, data_mu.T)
                print(svd_2.shape)
                svd_2 = np.sum(svd_2, axis=1)
                print(np.where(np.diag(self.covariance[k]) == 0))
                # print("det", np.exp(np.sum(np.log(np.diag(self.covariance[k])+0.01))))

                # Use log here
                r_k = np.divide(self.lamda[k], np.sqrt(2*np.pi)) + svd_2
                r[:, k] = r_k

            r = r / np.sum(r, axis=1).reshape(len(data), 1)

            # M-step
            # Updata lambda
            self.lamda = np.sum(r, axis=0)/np.sum(r)

            for k in range(k_num):
                # Updatae mean
                curr_mu = np.zeros_like(self.mu[0])
                for i in range(len(data)):
                    temp = r[i, k] * data[i].reshape(len(data[i]), 1)
                    curr_mu += temp

                self.mu[k] = curr_mu/ np.sum(r[:, k])

                # Update covariance matrix
                cov = np.zeros_like(self.covariance[k])
                for i in range(len(data)):
                    x_mu = data[i].reshape(len(data[i]), 1) - self.mu[k]
                    cov += r[i, k] * np.dot(x_mu, x_mu.T)

                self.covariance[k] = cov/np.sum(r[:,k])
                length = len(self.covariance[k])
                self.covariance[k] = np.eye(length, length) * np.diagonal(self.covariance[k])


    '''
        implement the split function here
        generates an initialization for a GMM with 2K components out of the current (already trained) GMM with K components
        @epsilon: small perturbation value for the mean
    '''
    def split(self, epsilon = 0.1):
        #TODO
        new_lamda = []
        new_mu = []
        new_cov = []
        k = len(self.mu)
        for i in range(k):
            new_lamda.append(self.lamda[i])
            new_lamda.append(self.lamda[i])
            new_mu.append(self.mu[i] + epsilon * self.std[i])
            new_mu.append(self.mu[i] - epsilon * self.std[i])
            new_cov.append(self.covariance[i])
            new_cov.append(self.covariance[i])

        self.mu = new_mu
        self.covariance = new_cov
        self.lamda = np.divide(new_lamda, 2)
        self.std = 2 * k * [self.std[0]]

    
    '''
        sample a D-dimensional feature vector from the GMM
    '''
    def sample(self):
        #TODO
        k_num = len(self.mu)
        new_mu = np.sum(self.mu, axis=0)/k_num
        new_cov = np.sum(self.covariance, axis=0)/k_num

        # The shape of new_mu is (256, 1), we have to input 1-D array to multivatiate_normal,
        # so it become new_mu[:, 0]
        sample = np.random.multivariate_normal(new_mu[:,0], new_cov, 1)

        return sample

    def compute_p(self, x):
        # Compute p(x | w )
        k_num = len(self.mu)
        r = np.zeros(k_num)
        for k in range(k_num):
            data_mu = x.reshape(len(x),1) - self.mu[k]
            cov_inv = np.linalg.pinv(self.covariance[k])
            svd = -0.5 * np.dot(np.dot(data_mu.T, cov_inv), data_mu)
            r_k = np.divide(self.lamda[k], np.sqrt(2 * np.pi)) * np.exp(svd)
            r[k] = np.squeeze(r_k)

        p = np.sum(r)
        return p


'''
    Task 2d: synthesizing handwritten digits
    if you implemeted the code in the GMM class correctly, you should not need to change anything here
'''
data = read_usps('usps.txt')
gmm = [ GMM() for _ in range(10) ] # 10 GMMs (one for each digit)
for split in [0, 1, 2]:
    result_image = np.zeros((160, 160))
    for digit in range(10):
        # train the model
        if split == 0:
            gmm[digit].fit_single_gaussian(data[digit])
        else:
            gmm[digit].em_algorithm(data[digit])
        # sample 10 images for this digit
        for i in range(10):
            x = gmm[digit].sample()
            x = x.reshape((16, 16))
            x = np.clip(x, 0, 1)
            result_image[digit*16:(digit+1)*16, i*16:(i+1)*16] = x
        # save image
        misc.imsave('digits.' + str(2 ** split) + 'components.png', result_image)
        # split the components to have twice as many in the next run
        gmm[digit].split(epsilon = 0.1)


'''
    Task 2e: skin color model
'''
image, foreground, background = read_face_image('face.jpg')

'''
    TODO: compute p(x|w=foreground) / p(x|w=background) for each image pixel and manipulate image such that everything below the threshold is black, display the resulting image
'''
# Train GMM for foreground
fore_GMM = GMM()
for split in [0, 1, 2]:
    if split == 0:
        fore_GMM.fit_single_gaussian(foreground)
    else:
        fore_GMM.em_algorithm(foreground)

    fore_GMM.split(epsilon = 0.1)

# Train GMM for background
back_GMM = GMM()
for split in [0, 1, 2]:
    if split == 0:
        back_GMM.fit_single_gaussian(background)
    else:
        back_GMM.em_algorithm(background)
    back_GMM.split(epsilon=0.1)

# Compute p(x|w=foreground) / p(x|w=background) for each image pixel
height, width,_ = image.shape
result_image = image.copy()
ratio_all = np.zeros((height, width, 3))
threshold = 0.1
for i in range(height):
    for j in range(width):
        x = image[i,j,:]
        p_foreground = fore_GMM.compute_p(x)
        p_background = back_GMM.compute_p(x)
        ratio = p_foreground/p_background
        ratio_all[i, j, :] = ratio

ratio_all = np.where(ratio_all < threshold, 0, 1)
result_image = np.multiply(result_image, ratio_all)
result_image = np.clip(result_image, 0, 1)
misc.imsave('foreground.png', result_image)