# =============================================================================
# Finite Element Solver for a big G grid
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt

# =============================================================================
# Grid Generation
# =============================================================================

def find_node_index_of_location(nodes, location):
    """
    Given all the nodes and a location (that should be the location of
    *a* node), return the index of that node.

    Parameters
    ----------

    nodes : array of float
        (Nnodes, 2) array containing the x, y coordinates of the nodes
    location : array of float
        (2,) array containing the x, y coordinates of location
    """
    dist_to_location = np.linalg.norm(nodes - location, axis=1)
    return np.argmin(dist_to_location)


def generate_g_grid(side_length):
    """
    Generate a 2d triangulation of the letter G. All triangles have the same
    size (right triangles, short length side_length)

    Parameters
    ----------

    side_length : float
        The length of each triangle. Should be 1/N for some integer N

    Returns
    -------

    nodes : array of float
        (Nnodes, 2) array containing the x, y coordinates of the nodes
    IEN : array of int
        (Nelements, 3) array linking element number to node number
    ID : array of int
        (Nnodes,) array linking node number to equation number; value is -1 if
        node should not appear in global arrays.
    """
    x = np.arange(0, 4 + 0.5 * side_length, side_length)
    y = np.arange(0, 5 + 0.5 * side_length, side_length)
    X, Y = np.meshgrid(x, y)
    potential_nodes = np.zeros((X.size, 2))
    potential_nodes[:, 0] = X.ravel()
    potential_nodes[:, 1] = Y.ravel()
    xp = potential_nodes[:, 0]
    yp = potential_nodes[:, 1]
    nodes_mask = np.logical_or(
            np.logical_and(xp >= 2, np.logical_and(yp >= 2,yp <= 3)),
                np.logical_or(np.logical_and(xp >= 3,yp <= 3),
                    np.logical_or(xp<=1, np.logical_or(yp<=1, yp>=4))))
    nodes = potential_nodes[nodes_mask, :]

    ID = np.zeros(len(nodes), dtype=np.int)
    n_eq = 0
    for nID in range(len(nodes)):
        if np.allclose(nodes[nID, 0], 4):
            ID[nID] = -1
        else:
            ID[nID] = n_eq
            n_eq += 1

    inv_side_length = int(1/side_length)
    Nelements_per_block = inv_side_length ** 2
    Nelements = 2 * 14 * Nelements_per_block
    IEN = np.zeros((Nelements, 3), dtype=np.int)
    block_corners = [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [3, 1], [0, 2],
                     [2, 2], [3, 2], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]]
    current_element = 0
    for block in block_corners:
        for i in range(inv_side_length):
            for j in range(inv_side_length):
                node_locations = np.zeros((4, 2))
                for a in range(2):
                    for b in range(2):
                        node_locations[a + 2 * b, 0] = block[0] + \
                            (i + a) * side_length
                        node_locations[a + 2 * b, 1] = block[1] + \
                            (j + b) * side_length
                index_lo_l = find_node_index_of_location(nodes,
                                                         node_locations[0, :])
                index_lo_r = find_node_index_of_location(nodes,
                                                         node_locations[1, :])
                index_hi_l = find_node_index_of_location(nodes,
                                                         node_locations[2, :])
                index_hi_r = find_node_index_of_location(nodes,
                                                         node_locations[3, :])
                IEN[current_element, :] = [index_lo_l, index_lo_r, index_hi_l]
                current_element += 1
                IEN[current_element, :] = [index_lo_r, index_hi_r, index_hi_l]
                current_element += 1

    return nodes, IEN, ID


# =============================================================================
# Functions
# =============================================================================

nodes, IEN, ID = generate_g_grid(1/2)

plt.figure(figsize=(10, 6))
plt.triplot(nodes[:, 0], nodes[:, 1], triangles=IEN, lw=1)
plt.axis('equal')
plt.xlim(-1, 5)
plt.ylim(-1, 6)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()
