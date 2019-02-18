import cv2
import numpy as np
import maxflow
from maxflow.fastmin import aexpansion_grid

import matplotlib.pyplot as plt
import networkx as nx

def plot_graph_2d(graph, nodes_shape, plot_weights=True, plot_terminals=True, font_size=7):
    X, Y = np.mgrid[:nodes_shape[0], :nodes_shape[1]]
    aux = np.array([Y.ravel(), X[::-1].ravel()]).T
    positions = {i: v for i, v in enumerate(aux)}
    positions['s'] = (-1, nodes_shape[0] / 2.0 - 0.5)
    positions['t'] = (nodes_shape[1], nodes_shape[0] / 2.0 - 0.5)

    nxgraph = graph.get_nx_graph()
    if not plot_terminals:
        nxgraph.remove_nodes_from(['s', 't'])

    plt.clf()
    nx.draw(nxgraph, pos=positions)

    if plot_weights:
        edge_labels = {}
        for u, v, d in nxgraph.edges(data=True):
            edge_labels[(u,v)] = d['weight']
        nx.draw_networkx_edge_labels(nxgraph,
                                     pos=positions,
                                     edge_labels=edge_labels,
                                     label_pos=0.3,
                                     font_size=font_size)

    plt.axis('equal')
    plt.show()

def question_3(I,rho=0.6,pairwise_cost_same=0.01,pairwise_cost_diff=0.5):

    ### 1) Define Graph
    g = maxflow.Graph[float]()
    ### 2) Add pixels as nodes
    row, col = I.shape
    nodeids = g.add_grid_nodes((row, col))
    
    ### 3) Compute Unary cost
    U = I.copy()
    U_I = 255 -U
    U = U.astype(float) / 255  
    U_I = U_I/255
    
    U[U==1] = 1-rho
    U[U==0] = rho
    U_I[U_I==1] = 1-rho
    U_I[U_I==0] = rho    
    U = -np.log(U)
    U_I = -np.log(U_I)
    # print(U, U_I)
      
    ### 4) Add terminal edges
    g.add_grid_tedges(nodeids, U, U_I )

    ### 5) Add Node edges

    # g.add_grid_edges(nodeids, pairwise_cost_diff, symmetric=True)
    ### Vertical Edges
    for c in range(col):
        for r in range(row-1):
            if U[r, c] == U[r+1, c]:
                g.add_edge(nodeids[r,c], nodeids[r+1, c], pairwise_cost_same, pairwise_cost_same)
            else:
                g.add_edge(nodeids[r,c], nodeids[r+1, c], pairwise_cost_diff, pairwise_cost_diff)

    ### Horizontal edges
    for r in range(row):
        for c in range(col-1):
            if U[r, c]==U[r, c+1]:
                g.add_edge(nodeids[r, c], nodeids[r, c+1], pairwise_cost_same, pairwise_cost_same)
            else:
                g.add_edge(nodeids[r,c], nodeids[r, c+1], pairwise_cost_diff, pairwise_cost_diff)

    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)

    ### 6) Maxflow
    g.maxflow()

    segments = g.get_grid_segments(nodeids)
    Denoised_I = np.int_(np.logical_not(segments))
    Denoised_I = Denoised_I.astype(np.float)

    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I*255), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def question_4(I,rho=0.6):
    labels = np.unique(I).tolist()

    Denoised_I = np.zeros_like(I)
    Denoised_I = I.copy()
    ### Use Alpha expansion binary image for each label
    #I = I/I.max()
    num_labels = len(labels)
    D = np.abs(I.reshape(I.shape + (1,)) - np.array(labels).reshape((1, 1, -1)))
    D = np.where(D == 0, rho, (1-rho)/2)
    V = 255 * np.eye(3)

    max_iter = 50
    better_energy = np.inf
    # Stop when energy is not changed or reached max iteration.
    for i in range(max_iter):
        improved = False
        # Iterate through the labels.
        for alpha in labels:
            # Create graph and Caculate the energy.
            energy, Denoised_I = calculate_energy(alpha, D, V, Denoised_I, rho)
            # Check if the better energy has been improved.
            if energy < better_energy:
                better_energy = energy
                improved = True
        # Finish the minimization when energy is not decreased.
        if not improved:
            break

    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()

    return

def calculate_energy(alpha, D, V, Denoised_I, rho):
    ### 1) Define Graph
    g = maxflow.Graph[float]()

    row, col = Denoised_I.shape
    ### 2) Add pixels as nodes
    nodeids = g.add_grid_nodes((row, col))
    ### 3) Compute Unary cost
    ### 4) Add terminal edges
    #g.add_grid_tedges(nodeids, rho, (1-rho)/2)
    label_dict = {0:0, 128:1, 255:2}

    ### 5) Add Node edges
    ### Vertical Edges
    for c in range(col):
        for r in range(row-1):
            curr_state = Denoised_I[r, c]
            if curr_state == alpha:
                # Add unary cost from label to source and sink
                g.add_tedge(nodeids[r, c], D[r,c,label_dict[alpha]], np.inf)
                if Denoised_I[r+1, c] != alpha: # for alpha and else
                    dist_state_alpha = V[label_dict[Denoised_I[r+1, c]], label_dict[alpha]]
                    g.add_edge(nodeids[r, c], nodeids[r+1, c], 0, dist_state_alpha)
            else:
                g.add_tedge(nodeids[r, c], D[r,c,label_dict[alpha]], D[r,c,label_dict[curr_state]])
                if Denoised_I[r+1, c] == alpha: # for else and alpha
                    dist_state_alpha = V[label_dict[Denoised_I[r, c]], label_dict[alpha]]
                    g.add_edge(nodeids[r, c], nodeids[r+1, c], dist_state_alpha, 0)
                else: # for else and else
                    if Denoised_I[r, c] == Denoised_I[r+1, c]:
                        dist_state_alpha = V[label_dict[Denoised_I[r, c]], label_dict[Denoised_I[r+1, c]]]
                        g.add_edge(nodeids[r, c], nodeids[r + 1, c], dist_state_alpha, dist_state_alpha)
                    else:
                        dist = V[label_dict[Denoised_I[r, c]], label_dict[Denoised_I[r+1, c]]]
                        curr = V[label_dict[Denoised_I[r, c]], label_dict[alpha]]
                        next = V[label_dict[Denoised_I[r+1, c]], label_dict[alpha]]
                        # Add an extra node between two nodes that both are not alpha
                        extra_node = g.add_nodes(1)
                        g.add_tedge(extra_node, 0, dist)
                        g.add_edge(nodeids[r, c], extra_node, curr, np.inf)
                        g.add_edge(nodeids[r+1, c], extra_node, np.inf, next)
    ### Horizontal edges
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)
    for r in range(row):
        for c in range(col - 1):
            curr_state = Denoised_I[r, c]
            if curr_state == alpha:
                if Denoised_I[r, c+1] != alpha:
                    dist_state_alpha = V[label_dict[alpha], label_dict[Denoised_I[r, c+1]]]
                    g.add_edge(nodeids[r, c], nodeids[r, c+1], 0, dist_state_alpha)
            else:
                if Denoised_I[r, c+1] == alpha:
                    dist_state_alpha = V[label_dict[Denoised_I[r, c]], label_dict[alpha]]
                    g.add_edge(nodeids[r, c], nodeids[r, c+1], dist_state_alpha, 0)
                else:
                    if Denoised_I[r, c] == Denoised_I[r, c+1]:
                        dist_state_alpha = V[label_dict[Denoised_I[r, c]], label_dict[Denoised_I[r, c+1]]]
                        g.add_edge(nodeids[r, c], nodeids[r, c+1], dist_state_alpha, dist_state_alpha)
                    else:
                        dist = V[label_dict[Denoised_I[r, c]], label_dict[Denoised_I[r, c+1]]]
                        curr = V[label_dict[Denoised_I[r, c]], label_dict[alpha]]
                        next = V[label_dict[Denoised_I[r, c+1]], label_dict[alpha]]
                        extra_node = g.add_nodes(1)
                        g.add_tedge(extra_node, 0, dist)
                        g.add_edge(nodeids[r, c], extra_node, curr, np.inf)
                        g.add_edge(nodeids[r, c+1], extra_node, np.inf, next)

    ### 6) Maxflow
    energy = g.maxflow()

    segments = g.get_grid_segments(nodeids)
    segments = np.logical_not(segments)
    for i in range(row):
        for j in range(col):
            if segments[i,j]:
                Denoised_I[i, j] = alpha

    return energy, Denoised_I

def main():
    image_q3 = cv2.imread('./images/noise.png', cv2.IMREAD_GRAYSCALE)
    image_q4 = cv2.imread('./images/noise2.png', cv2.IMREAD_GRAYSCALE)

    ### Call solution for question 3
    # question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.15)
    # question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.3)
    # question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.6)

    ### Call solution for question 4
    question_4(image_q4, rho=0.8)
    return

if __name__ == "__main__":
    main()



