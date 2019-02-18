import numpy as np
import numpy.linalg as la
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# --- this are all the imports you are supposed to use!
# --- please do not add more imports!

n_views = 101
n_features = 215

# --- add your code here ---


# --- how to plot 3d ---
pts3d = np.random.randint(0, 10, (100, 3))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])

plt.show()
