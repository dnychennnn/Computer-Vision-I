import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)  # if you do not have latex installed simply uncomment this line + line 75


def load_data():
    """ loads the data for this task
    :return:
    """
    fpath = 'images/ball.png'
    radius = 70
    Im = cv2.imread(fpath, 0).astype('float32')/255  # 0 .. 1

    # we resize the image to speed-up the level set method
    Im = cv2.resize(Im, dsize=(0, 0), fx=0.5, fy=0.5)

    height, width = Im.shape

    centre = (width // 2, height // 2)
    Y, X = np.ogrid[:height, :width]
    phi = radius - np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    return Im, phi


def get_contour(phi):
    """ get all points on the contour
    :param phi:
    :return: [(x, y), (x, y), ....]  points on contour
    """
    eps = 1
    A = (phi > -eps) * 1
    B = (phi < eps) * 1
    D = (A - B).astype(np.int32)
    D = (D == 0) * 1
    Y, X = np.nonzero(D)
    return np.array([X, Y]).transpose()

# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here
# Calculate the norm for each point, the shape of gradient is 2 * height * width
def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))
# ------------------------


if __name__ == '__main__':

    n_steps = 20000
    plot_every_n_step = 100

    Im, phi = load_data()
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # ------------------------
    # your implementation here
    # Initialize the parameters
    epslon = 1 / np.power(10, 4)
    gamma = 0.5
    dIm = np.array(np.gradient(Im))
    omega =1./np.sqrt(1. + norm(dIm)**2)

    # ------------------------

    for t in range(n_steps):
        # ------------------------
        # your implementation here
        # Calculate the gradient of phi
        height, width = Im.shape
        dphi = np.zeros((height, width))
        for h in range(1, height-1):
            for w in range(1, width-1):
                phi_y = (phi[h + 1, w] - phi[h - 1, w]) / 2
                phi_yy = phi[h + 1, w] - 2 * phi[h, w] + phi[h - 1, w]
                phi_xy = (phi[h + 1, w + 1] - phi[h - 1, w + 1] - phi[h + 1, w - 1] + phi[h - 1, w - 1]) / 4
                phi_x = (phi[h, w+1] - phi[h, w-1])/2
                phi_xx = phi[h, w + 1] - 2 * phi[h, w] + phi[h, w - 1]

                dphi[h, w] = (phi_xx*np.square(phi_y) - 2*phi_x*phi_y*phi_xy + phi_yy*np.square(phi_x))/(np.square(phi_x) + np.square(phi_y) + epslon)

        # Update phi
        phi = phi + gamma * omega * dphi
        # ------------------------

        if t % plot_every_n_step == 0:
            ax1.clear()
            ax1.imshow(Im, cmap='gray')
            ax1.set_title('frame ' + str(t))

            contour = get_contour(phi)
            if len(contour) > 0:
                ax1.scatter(contour[:, 0], contour[:, 1], color='red', s=1)

            ax2.clear()
            ax2.imshow(phi)
            #ax2.set_title(r'$\phi$', fontsize=22)
            plt.pause(0.01)
    plt.show()


