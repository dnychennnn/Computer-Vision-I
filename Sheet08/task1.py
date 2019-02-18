import numpy as np
import utils



# ========================== Mean =============================
def calculate_mean_shape(kpts):
    # ToDO
    mean_shape = np.expand_dims(np.mean(kpts, axis=0), axis=0)
    return mean_shape



# ====================== Main Step ===========================
def procrustres_analysis_step(kpts, reference_mean):
    # Happy Coding
    reference_mean = reference_mean[0]
    for i, kpt in enumerate(kpts):
        N = 56
        # centered the shapes
        centroid_kpt = np.mean(kpt, axis=0)
        centroid_mean = np.mean(reference_mean, axis=0)
        AA = kpt - np.tile(centroid_kpt, (N, 1))
        BB = reference_mean - np.tile(centroid_mean, (N, 1))
        # implement svd decomposition for affine transformation matrix
        H = AA.T @ BB
        U,D,Vt = np.linalg.svd(H)
        R =  Vt@U.T
        t = -R@centroid_kpt.T + centroid_mean.T
        t = t.reshape(2,1)

        kpts[i] = ((R@kpt.T+t)).T

    return kpts



# =========================== Error ====================================

def compute_avg_error(kpts, mean_shape):
    # ToDo
    
    rms = np.sqrt(np.square(np.linalg.norm(kpts - mean_shape, axis=2)).mean(axis=1))
    rms = rms.mean()
    print(rms)
    return rms

# ============================ Procrustres ===============================

def procrustres_analysis(kpts, max_iter=int(1e3), min_error=1e-5):

    aligned_kpts = kpts.copy()
    aligned_kpts = utils.convert_samples_to_xy(aligned_kpts)
    start_mean = calculate_mean_shape(aligned_kpts)
    for iter in range(max_iter):
        print(iter)
        ##################### Your Part Here #####################
        reference_mean = calculate_mean_shape(aligned_kpts)
        # align shapes to mean shape
        aligned_kpts = procrustres_analysis_step(aligned_kpts, reference_mean)
        
        new_mean = calculate_mean_shape(aligned_kpts)
        err = compute_avg_error(aligned_kpts, new_mean)
        if err < min_error:
            break
        ##########################################################


    # visualize
    utils.visualize_hands(aligned_kpts, "After Aligned", delay=.1)  
    # visualize mean shape
    utils.visualize_hands(calculate_mean_shape(aligned_kpts), "Mean Shape After Aligned")
    return aligned_kpts
