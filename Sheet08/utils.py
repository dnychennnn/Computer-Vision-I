import numpy as np
import matplotlib.pyplot as plt
import csv

# ======================= Data Loading =======================
def load_data(path):
    data_dim = 0
    num_data = 0
    samples = []
    with open(path, 'r') as f:
        data_reader = csv.reader(f, delimiter='\t')

        for line_idx, row in enumerate(data_reader):
            if line_idx == 0:
                data_info = row
                data_dim = int(data_info[0])
                num_data = int(data_info[1])

            else:
                # fix row since data has some issues
                sample = np.asarray(row)
                sample = sample[:num_data].astype(np.float32)
                samples.append(sample)

    return {
        'data_dim': int(data_dim),
        'num_data': int(num_data),
        'samples': np.asarray(samples).transpose()
    }


# ======================= Conversion =======================
def convert_samples_to_xy(data_matrix):
    num_points = int(data_matrix.shape[1] / 2)

    X = data_matrix[:, :num_points]
    Y = data_matrix[:, num_points:]

    kpts = np.asarray([X, Y]).transpose([1, 2, 0])

    return kpts




# ======================= Visualization =======================

def visualize_hands(kpts, title, delay=0.5, ax=None, clear=False):
    """
        kpts: shape keypoints with dim [num_samples x num_keypoints x 2]
    """
    if ax is None:
        fig = plt.figure(figsize=(5, 4))
        fig.canvas.set_window_title(title)
        ax = fig.add_subplot(111)
        ax.invert_yaxis()
        plt.axis('off')

    for sample_idx in range(kpts.shape[0]):
        if clear:
            ax.clear()
            ax.invert_yaxis()
            plt.axis('off')

        ax.plot(kpts[sample_idx, :, 0], kpts[sample_idx, :, 1])
        plt.pause(delay)
    return ax
