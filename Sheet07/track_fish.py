import cv2
import numpy as np
import os

IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

INIT_X = 448
INIT_Y = 191
INIT_WIDTH = 38
INIT_HEIGHT = 33

INIT_BBOX = [INIT_X, INIT_Y, INIT_WIDTH, INIT_HEIGHT]


def load_frame(frame_number):
    """
    :param frame_number: which frame number, [1, 32]
    :return: the image
    """
    image = cv2.imread(os.path.join(IMAGES_FOLDER, '%02d.png' % frame_number))
    return image


def crop_image(image, bbox):
    """
    crops an image to the bounding box
    """
    x, y, w, h = tuple(bbox)
    return image[y: y + h, x: x + w]


def draw_bbox(image, bbox, thickness=2, no_copy=False):
    """
    (optionally) makes a copy of the image and draws the bbox as a black rectangle.
    """
    x, y, w, h = tuple(bbox)
    if not no_copy:
        image = image.copy()
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), thickness)

    return image


def compute_histogram(image):
    # implement here
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_hist = cv2.calcHist(image,[2], None, [256], [0, 256])
    return image_hist


def compare_histogram(hist1, hist2):
    # implement here

    hist_comp_val = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    likelihood = np.exp(-hist_comp_val * 20.0)
    return likelihood



class Position(object):
    """
    A general class to represent position of tracked object.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_bbox(self):
        """
        since the width and height are fixed, we can do such a thing.
        """
        return [self.x, self.y, INIT_WIDTH, INIT_HEIGHT]

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Position(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Position(self.x * other, self.y * other)

    def __repr__(self):
        return "[%d %d]" % (self.x, self.y)

    def make_ready(self, image_width, image_height):
        # convert to int
        self.x = int(round(self.x))
        self.y = int(round(self.y))

        # make sure inside the frame
        self.x = max(self.x, 0)
        self.y = max(self.y, 0)
        self.x = min(self.x, image_width)
        self.y = min(self.y, image_height)


class ParticleFilter(object):
    def __init__(self, du, sigma, num_particles=200):
        self.template = None  # the template (histogram) of the object that is being tracked.
        self.position = None  # we don't know the initial position still!
        self.particles = []  # we will store list of particles at each step here for displaying.
        self.fitness = []  # particle's fitness values
        self.du = du
        #self.mu = np.array([0, 0])#.reshape(1,2)
        self.mu_x = INIT_X
        self.mu_y = INIT_Y
        self.sigma = sigma
        self.num_particles = num_particles
        self.image_width = 0
        self.image_height = 0

        self.count = 2

    def init(self, frame, bbox):
        self.image_width = frame.shape[1]
        self.image_height = frame.shape[0]

        self.position = Position(x=bbox[0], y=bbox[1])  # initializing the position
        # implement here ...
        object_frame = crop_image(frame, bbox)
        self.template = compute_histogram(object_frame)
        x = np.random.normal(self.mu_x, self.sigma, self.num_particles)
        y = np.random.normal(self.mu_y, self.sigma, self.num_particles)
        self.list_2_box(x, y)
        self.fitness = np.zeros([200, 1])
        self.fitness += 1/self.num_particles


    def track(self, new_frame):
        # implement here ...
        curr_mean = np.sum(np.multiply(self.fitness, self.box_2_list()), axis=0)
        #print('curr mean', curr_mean)
        x = np.random.normal(self.mu_x, self.sigma, self.num_particles)
        y = np.random.normal(self.mu_y, self.sigma, self.num_particles)
        self.list_2_box(x, y)
        for i in range(self.num_particles):
            curr_box = self.particles[i].get_bbox()
            object_frame = crop_image(new_frame, curr_box)

            self.fitness[i, 0] = compare_histogram(self.template, compute_histogram(object_frame))

        self.fitness = self.fitness/np.sum(self.fitness)


        next_mean = np.sum(self.fitness * self.box_2_list(), axis=0)
        
        if next_mean[0] > self.mu_x:
            self.du[0] = 1.05
        else:
            self.du[0] = 0.95
        self.mu_x = self.du[0] * self.mu_x + (1-self.du[0]) * (next_mean[0] - curr_mean[0])

        if next_mean[1] > self.mu_y:
            self.du[1] = 1.05
        else:
            self.du[1] = 0.95
        self.mu_y = self.du[1] * self.mu_y + (1-self.du[1]) * (next_mean[1] - curr_mean[1])
        #print(self.mu_x, self.mu_y)

        self.count += 1

    def box_2_list(self):
        particles_list = []
        for i in range(self.num_particles):
            particles_list.append(self.particles[i].get_bbox()[:2])

        return particles_list

    def list_2_box(self, x_list, y_list):
        self.particles = []
        for i in range(self.num_particles):
            pos = Position(x=x_list[i], y=y_list[i])
            pos.make_ready(self.image_width, self.image_height)
            self.particles.append(pos)


    def display(self, current_frame):
        #cv2.imshow('frame', current_frame)
        frame_copy = current_frame.copy()
        print(len(self.particles))
        for i in range(len(self.particles)):
            draw_bbox(frame_copy, self.particles[i].get_bbox(), thickness=1, no_copy=True)

        cv2.imshow('particles', frame_copy)
        cv2.waitKey(100)
        #cv.destroyAllWindows()


def main():
    np.random.seed(0)
    DU = np.array([1.05, 1.05])
    SIGMA = 15

    cv2.namedWindow('particles')
    #cv2.namedWindow('frame')
    frame_number = 1
    frame = load_frame(frame_number)

    tracker = ParticleFilter(du=DU, sigma=SIGMA)
    tracker.init(frame, INIT_BBOX)
    tracker.display(frame)

    for frame_number in range(2, 33):
        frame = load_frame(frame_number)
        tracker.track(frame)
        tracker.display(frame)


if __name__ == "__main__":
    main()
