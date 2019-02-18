#!/usr/bin/python3.5

import numpy as np
from scipy import misc
from scipy.stats import multivariate_normal as mvn

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

    mu = 0
    sigma = 0
    k = 1

    mu_k = []
    sigma_k = []
    lambda_k = []
 

    '''
        fit a single gaussian to the data
        @data: an N x D numpy array, where N is the number of observations and D is the dimension (256 for usps digits, 3 for the skin color RGB pixels)
    '''
    def fit_single_gaussian(self, data):
        #TODO

        # avoid singular matrix
        data = data
        self.mu = np.mean(data, axis=0)
        self.sigma = np.std(data, axis=0)
        self.cov = np.cov(data, rowvar=False)
        # self.cov = np.diagflat(np.var(data, axis=0))
        # self.cov = np.eye(data.shape[1])
        # print(np.linalg.det(self.cov))
        # self.cov = np.diagflat(self.sigma)
        self.mu_k = np.array(self.mu)
        self.cov_k = np.array(self.cov)
        self.lambda_k =np.ones((self.k)) / self.k
       
    '''
        implement the em algorithm here
        @data: an N x D numpy array, where N is the number of observations and D is the dimension (256 for usps digits, 3 for the skin color RGB pixels)
        @n_iterations: the number of em iterations
    '''
   
    def em_algorithm(self, data, n_iterations = 10):   

        #TODO
        lambda_k = np.expand_dims(np.ones((self.k)) / self.k, axis=1)
        r_ik =np.zeros((self.k, data.shape[0]))
        mu_k = data[np.random.choice(data.shape[0], self.k, False), :]
        cov_k = [np.eye(data.shape[1])] * self.k

        print(mu_k.shape)

        np.seterr(divide='ignore', invalid='ignore')
        for i in range(n_iterations):
            print("........................", i)
            # E-step
            a = np.einsum('ij, ij -> i',\
                        (data - mu_k[0]), np.dot(np.linalg.inv(cov_k[0]) , (data - mu_k[0]).T).T )
            print("a", a.shape)

            for k in range(self.k):     
                part1 = np.exp(2. * np.sum(np.log(np.diag(cov_k[k])))) ** -.5 * ( 2*np.pi) ** (-data.shape[1]/2.)
                print(part1)
                for idx in range(data.shape[0]):
                    part2 = -.5 * ( (data[idx]-mu_k_k[k]).dot(np.linalg.inv(cov_k[k]))).dot((data[idx]-mu_k[k]).T)
                    r_ik[k][idx] = part1 * np.exp(part2)
            # print(mu_k[0][0], mu_k[1][0])
            print(r_ik[0][0])
            print(r_ik[1][0])
            # print(-.5 * ( (data[0]-mu_k[0]).dot(np.linalg.inv(cov_k[0]))).dot((data[0]-mu_k[0]).T)) #/ 1/( np.sqrt(2*np.pi) * (2. * np.sum(np.log(np.diag(cov_k[0]))))))
            # print(-.5 * ( (data[0]-mu_k[1]).dot(np.linalg.inv(cov_k[1]))).dot((data[0]-mu_k[1]).T))
            print(lambda_k)
            # pint(np.sum( lambda_k * r_ik, axis=0 ).shape)
            # print(r_ik[0][0] /np.sum( lambda_k * r_ik, axis=0 ))
            # print(r_ik[1][0] /np.sum( lambda_k * r_ik, axis=0 ))

            r_ik = lambda_k * r_ik / np.sum( lambda_k * r_ik, axis=0 )
            # print(numpy.isnan(np.nansum(r_ik)).any())
            # print( lambda_k[0] * r_ik[0])
            # print( lambda_k[1] * r_ik[1])
            # M-step
            r_ik_ex = np.expand_dims(r_ik, axis=2)
            # update lambda_k
            lambda_k_1 = np.nansum(r_ik, axis=1) / np.nansum(r_ik)
            
            # update mu_k
            mu_k_1 = np.sum(r_ik_ex * data, axis=1) / np.nansum(r_ik_ex, axis=1)
            # update cov_k
            for k in range(self.k):
                # numerator size:(256, 256)   denominator size:(1(k),256)
                cov_k[k]= np.nansum(np.expand_dims(r_ik_ex[k], axis=1) * np.dot((data - mu_k[k]).T,(data-mu_k[k])), axis=0) / np.nansum(r_ik[k], axis=0)
            # update the iterate value, expand dimsion for matrix calculation
            lambda_k = np.expand_dims(lambda_k_1, axis=1)
        

        self.mu_k = mu_k_1
        self.lambda_k = lambda_k_1
        self.cov_k = cov_k
        # print(self.mu_k, self.lambda_k, self.cov_k)
        
    '''
        implement the split function here
        generates an initialization for a GMM with 2K components out of the current (already trained) GMM with K components
        @epsilon: small perturbation value for the mean
    '''
   
    
    
    def split(self, epsilon = 0.1):
        #TODO
        temp_mu_k = []
        temp_cov_k = []
        for k in range(0, self.k):
            mu_k1 = self.mu_k[k] + epsilon * self.sigma
            mu_k2 = self.mu_k[k] - epsilon * self.sigma
            temp_mu_k.append(mu_k1)
            temp_mu_k.append(mu_k2)
            temp_cov_k.append(self.cov)
            temp_cov_k.append(self.cov)

        self.k = self.k*2
        self.mu_k = np.asarray(temp_mu_k)
        self.cov_k = np.asarray(temp_cov_k)
        print(self.mu_k[0][0], self.mu_k[1][0])
        

    '''
        sample a D-dimensional feature vector from the GMM
    '''
    def sample(self):
        #TODO
        np.seterr(divide='ignore', invalid='ignore')
        mu = np.sum(np.expand_dims(self.lambda_k, axis=1) * self.mu_k, axis=0)
        
        if self.k ==1:
            cov_k = self.cov_k
            cov_k = np.expand_dims(cov_k, axis=0)
        else:
            cov_k = self.cov_k
        
        cov = cov_k[0]

        # print(cov, mu)
        sample = np.random.multivariate_normal(mu, cov)
        
        return sample 



'''
    Task 2d: synthesizing handwritten digits
    if you implemeted the code in the GMM class correctly, you should not need to change anything here
'''
data = read_usps('usps.txt')
gmm = [ GMM() for _ in range(10) ] # 10 GMMs (one for each digit)
for split in [0, 1, 2]:
    result_image = np.zeros((160, 160))
    print(split)
    for digit in range(10):
        # train the model
        if split == 0:
            gmm[digit].fit_single_gaussian(data[digit])
        else:
            gmm[digit].em_algorithm(data[digit])
        # sample 10 images for this digit
        for i in range(1,10):
            x = gmm[digit].sample()
            x = x.reshape((16, 16))
            x = np.clip(x, 0, 1)
            result_image[digit*16:(digit+1)*16, i*16:(i+1)*16] = x
        # save image
        misc.imsave('digits.' + str(2 ** split) + 'components.png', result_image)
        # split the components to have twice as many in the next run
        gmm[digit].split(epsilon = 0.01)


'''
    Task 2e: skin color model
'''
image, foreground, background = read_face_image('face.jpg')

'''
    TODO: compute p(x|w=foreground) / p(x|w=background) for each image pixel and manipulate image such that everything below the threshold is black, display the resulting image
'''
