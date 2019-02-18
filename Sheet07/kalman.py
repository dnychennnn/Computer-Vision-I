import numpy as np
import matplotlib.pylab as plt

observations = np.load('observations.npy')


def get_observation(t):
    return observations[t]


class KalmanFilter(object):
    def __init__(self, Lambda, sigma_p, Phi, sigma_m):
        self.Lambda = Lambda
        self.sigma_p = sigma_p
        self.Phi = Phi
        self.sigma_m = sigma_m
        self.state = None
        self.convariance = None

    def init(self, init_state):
        self.state = init_state
        self.convariance = np.eye(init_state.shape[0]) * 0.01

    def track(self, xt):
        
        pred_state =  self.Lambda @ self.state.T
        pred_covariance = self.sigma_p + self.Lambda @ self.convariance @ self.Lambda.T
        kalman_gain = pred_covariance @ self.Phi.T @ np.linalg.inv(self.sigma_m + self.Phi @ (pred_covariance @ self.Phi.T))
        update_state = pred_state + kalman_gain @ (xt - self.Phi @ pred_state)
        update_covariance = (np.identity(kalman_gain.shape[0]) - kalman_gain @ self.Phi) @ pred_covariance
        self.state = update_state
        self.convariance = update_covariance
        pass

    def get_current_location(self):
        return self.Phi @ self.state


def main():
    init_state = np.array([0, 1, 0, 0])

    dt = 1
    Lambda = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0,1]])

    sp = 0.01
    sigma_p = np.array([[sp, 0, 0, 0],
                        [0, sp, 0, 0],
                        [0, 0, sp * 4, 0],
                        [0, 0, 0, sp * 4]])

    Phi = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) 

    sm = 0.05
    sigma_m = np.array([[sm, 0], [0, sm]])

    tracker = KalmanFilter(Lambda, sigma_p, Phi, sigma_m)
    tracker.init(init_state)

    track = []
    for t in range(len(observations)):
        tracker.track(get_observation(t))
        track.append(tracker.get_current_location())

    plt.figure()
    plt.plot([x[0] for x in observations], [x[1] for x in observations])
    plt.plot([x[0] for x in track], [x[1] for x in track])
    plt.show()


if __name__ == "__main__":
    main()
