import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2


def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0,:]).reshape((-1, 2))
    ax.plot(V_plt[:,0], V_plt[:,1], color=line, alpha=alpha)
    ax.scatter(V[:,0], V[:,1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))


def load_data(fpath, radius):
    """
    :param fpath:
    :param radius:
    :return:
    """
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    n = 20  # number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return Im, V


# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here
def find_best_shift(start_snake, external):
    snakelength = start_snake.shape[0]
    minimal_way_sum_prev = np.zeros((snakelength, 9), dtype=int)  ###of previous layer
    E = np.zeros((snakelength, 9))
    snake_neighbors = []
    for current_layer in range(0, snakelength):
        curr = start_snake[current_layer%snakelength]

         #get snake neighbor
        snake_neighbors.append( get_neighbor(start_snake[current_layer]) )
        for current_possibility in range(0, 9):
            # calculate energy from previous layer to current point
            e_prev_to_curr = external[current_layer-1] + external[current_layer][current_possibility] + get_internal(start_snake, current_layer, snake_neighbors[current_layer][current_possibility])
            # get the min path from point from last layer
            e_total_curr_possibility = min(e_prev_to_curr)
            # record the path from last layer
            minimal_prev_index =list(e_prev_to_curr).index(e_total_curr_possibility)
            
            minimal_way_sum_prev[current_layer][current_possibility] = minimal_prev_index

            E[current_layer][current_possibility] = e_total_curr_possibility
    min_path = []
    min_coor = []
    current_layer = snakelength-1
    min_path_start = np.argmin(E[current_layer]) #min position at the end of the energy matrix
    while True:        
        min_path_start=minimal_way_sum_prev[current_layer][min_path_start]
        min_path.append((min_path_start))
        current_layer = current_layer-1
        min_coor.append(snake_neighbors[current_layer][min_path_start])
        if current_layer < 0:
            break
    print("min path:", min_path)
    new_snake = np.array(min_coor)
    return new_snake
             

def gradient(Im):
    return np.gradient(Im)

def getExternal(G, snake, gamma):
    #return "minus the squared gradient" at 9 positons for each point of the snake.
    gx, gy = G
    G = np.array(G)
    snake = np.array(snake)
    External = []

    for idx,s in enumerate(snake):
        neighbors = get_neighbor(s)
        e_neighbors = []
        for n in neighbors:
            e_neighbors.append(-(gx[tuple(n)]**2 + gy[tuple(n)]**2) *gamma)
        External.append(e_neighbors)
    
    return External

def get_neighbor(snake):
    neighbors = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            neighbors.append([snake[0]+dx, snake[1]+dy])
    
    return np.array(neighbors)

def get_internal(snake, layer, curr):
    """return the m=9 values of prev layer to current point"""
    n_points = len(snake)
    elastic = []
    alpha = 0.000001
    
    penalize_dev = np.sum([np.linalg.norm(snake[i]-snake[(i+1)% n_points] ) for i in range( 0, n_points ) ] ) / n_points
    
    prev = get_neighbor(snake[layer-1]) 
    
    for p in prev:
        elastic.append(np.linalg.norm(curr-p))
    return alpha*(penalize_dev-elastic)**2

# ------------------------


def run(fpath, radius):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    Im, V = load_data(fpath, radius) #V is snake

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 200    
    # ------------------------
    # your implementation here
    G = gradient(Im)
    gamma = 0.0001
    # ------------------------

    for t in range(n_steps):
        # ------------------------
        # your implementation here
        External_energies = getExternal(G, V, gamma)
        V = find_best_shift(V, External_energies)
        # ------------------------

        ax.clear()
        ax.imshow(Im, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, V)
        plt.pause(0.01)

    plt.pause(2)


if __name__ == '__main__':
    run('images/ball.png', radius=120)
    run('images/coffee.png', radius=100)
