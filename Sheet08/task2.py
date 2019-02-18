import numpy as np
import utils

# ======================= PCA =======================
def pca(centered_hands ,covariance, preservation_ratio=0.9):

    # Happy Coding! :)

    eigen_val, eigen_vec = np.linalg.eig(covariance)
    eigen_vec = eigen_vec.T
    eigen_val = eigen_val
    
    # 90% preservation
    _,S,_ = np.linalg.svd(centered_hands)
    normS = np.sum(np.square(S))
    K = np.min(np.where((np.cumsum(np.square(S)) / normS)> .9))

    # get top K principal components
    K_index = np.argsort(eigen_val)[-1:-K-1:-1]
    K_weights = eigen_val[K_index] / covariance.shape[0]
    K_eigen_vecs = eigen_vec[K_index]



    return K_eigen_vecs, K_weights




# ======================= Covariance =======================

def create_covariance_matrix(kpts, mean_shape):
    # ToDO
    W = kpts - mean_shape
    WW = np.cov(W.T)
    return WW





# ======================= Visualization =======================

def visualize_impact_of_pcs(mean, pcs, pc_weights):
    # your part here

    utils.visualize_hands(utils.convert_samples_to_xy(np.expand_dims(mean, axis=0)), "mean", delay=1)

    # get positive and negative weights
    v = np.sqrt(5)
    positive_K_weights = pc_weights * v
    negative_K_wegihts = pc_weights * -v

    A = mean + np.expand_dims(np.sum(pcs * np.expand_dims(positive_K_weights, axis=1), axis=0), axis=0)
    utils.visualize_hands(utils.convert_samples_to_xy(A), "Mean with Sum of positive weighted PCs", delay=1)

    B = mean + np.expand_dims(np.sum(pcs * np.expand_dims(negative_K_wegihts, axis=1), axis=0), axis=0)
    utils.visualize_hands(utils.convert_samples_to_xy(B), "Mean with Sum of negative weighted PCs", delay=1)

   
    A = mean + np.expand_dims(pcs * np.expand_dims(positive_K_weights, axis=1), axis=0)
    
    utils.visualize_hands(utils.convert_samples_to_xy(A[0]), "Difference between each positive weighted PCs ", delay=.4)

    B = mean + np.expand_dims(pcs * np.expand_dims(negative_K_wegihts, axis=1), axis=0)
    utils.visualize_hands(utils.convert_samples_to_xy(B[0]), "Difference between each negative weighted PCs", delay=.4)

    
    return None





# ======================= Training =======================
def train_statistical_shape_model(kpts):
    # Your code here
    mean_shape = np.mean(kpts, axis=0)
    W = kpts - mean_shape

    ## scatter matrix normalized = covariance matrix
    WW_cov = W.T @ W / (W.shape[0]-1)
    W_cov = create_covariance_matrix(kpts, mean_shape)
    K_pcs, K_weights = pca(W, W_cov)
    

    # visualization 
    visualize_impact_of_pcs(mean_shape, K_pcs, K_weights)
    

    return mean_shape, K_pcs, K_weights




# ======================= Reconstruct =======================
def reconstruct_test_shape(kpts, mean, pcs, pc_weight):
    #ToDo
    N = 56
    print("Hik: ", pc_weight)

    weighted_pc =pcs*np.expand_dims(pc_weight, axis=1)
    X = kpts + weighted_pc
    utils.visualize_hands(utils.convert_samples_to_xy(X), "reconstruction")

    pass
