# =============================================================================
# A 2D finite element solver using triangular elements
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12, 6)


# =============================================================================
# Grid Generation - provided by Dr Ian Hawke
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
# Finite Element Helper Functions
# =============================================================================

def shape_function(xi):
    """
    Compute the value of the shape functions for a reference element with
    coordinates (xi, eta).

    Parameters
    ----------
    xi: 1 x 2 array of floats. The local coordinates of the  element at the
        reference coordinates (xi, eta).

    Returns
    -------
    N: 3 x 1 array of floats. The value of the shape functions at the reference
        coordinates (xi, eta)
    """

    N = np.array([[1 - xi[1] - xi[0]], [xi[0]], [xi[1]]])

    return N


def shape_function_dN():
    """
    Compute the differential of the shape functions at a reference element
    with coordinates (xi, eta). As linear shape functions will always be used,
    the differentials do not depend on the coordinates, hence there is no
    argument for this function.

    Returns
    -------
    dN: 1 x 3 array of ints. An array containing the values of the
        differential at the reference coordinates (xi, eta) for an element.
    """

    dN = np.array([[-1, -1], [1, 0], [0, 1]])

    return dN


def local_to_global(nodes, xi):
    """
    Converts the local coordinates for a reference element to the global
    coordinates for an element.

    Parameters
    ----------
    nodes: 3 x 2 array. The global locations of the nodes of the triangular
        element.
    xi: 1 x 2 array. The local coordinates of the  element at the reference
        coordinates (xi, eta).

    Returns
    -------
    global_coords: 1 x 2 array. The global coordinates of the element.

    """
    N = shape_function(xi)

    x = nodes[0, 0] * N[0] + nodes[1, 0] * N[1] + nodes[2, 0] * N[2]
    y = nodes[0, 1] * N[0] + nodes[1, 1] * N[1] + nodes[2, 1] * N[2]

    global_coords = np.hstack((x, y))

    return global_coords


def jacobian(nodes, xi):
    """
    Compute the Jacobian matrix at a node (xi, eta).

    Parameters
    ----------
    nodes: 3 x 2 array. The global locations of the nodes of the triangular
        element.
    xi: 1 x 2 array. The local coordinates of the  element at the reference
        coordinates (xi, eta).
    Returns
    -------
    """

    dN = shape_function_dN()    # calculate the derivatives of the shape funcs
    J = np.dot(dN.T, nodes)     # costruct the Jacobian matrix

    return J


def det_jacobian(nodes, xi):
    """
    Compute the determinant of the Jacobian matrix at a node (xi, eta).

    Parameters
    ----------
    nodes: 3 x 2 array. The global locations of the nodes of the triangular
        element.
    xi: 1 x 2 array. The local coordinates of the  element at the reference
        coordinates (xi, eta).
    Returns
    -------
    """

    return np.abs(np.linalg.det(jacobian(nodes, xi)))


def global_dN(nodes, xi):
    """
    Compute the global derivatives using the Jacobian matrix.

    Parameters
    ----------
    nodes: 3 x 2 array. The global locations of the nodes of the triangular
        element.
    xi: 1 x 2 array. The local coordinates of the  element at the reference
        coordinates (xi, eta).

    Returns
    -------
    """

    dN = shape_function_dN()
    J = jacobian(nodes, xi)
    global_dN = np.zeros((3, 2))

    for i in range(3):
        global_dN[i, :] = np.linalg.solve(J, dN[i, :])

    return global_dN


def reference_quad(psi):
    """
    Compute the quadrature over the reference triangle.

    Parameters
    ----------
    psi: function.

    Returns
    -------
    reference_quad: float.
    """

    xi1 = [1/6, 1/6]
    xi2 = [4/6, 1/6]
    xi3 = [1/6, 4/6]

    reference_quad = (1/6) * (psi(xi1) + psi(xi2) + psi(xi3))

    return reference_quad


def element_quad(phi, nodes):
    """
    Compute the quadrature over the element.

    Parameters
    ----------
    nodes: 3 x 2 array. The global locations of the nodes of the triangular
        element.
    """

    def psi(xi):
        return det_jacobian(nodes, xi) * phi(local_to_global(nodes, xi),
                                             shape_function(xi),
                                             global_dN(nodes, xi))

    return reference_quad(psi)


def local_stiffness(nodes):
    """
    Compute the local stiffness matrix.

    Parameters
    ----------
    nodes: 3 x 2 array. The global locations of the nodes of the triangular
        element.
    """

    N = nodes.shape[1] + 1  # N = number of dimensions + 1
    k_ab = np.zeros((N, N))

    # loop over the node reference points
    for a in range(N):
        for b in range(N):
            # define the vector phi
            def phi_k(x, N, dN):
                return dN[a, 0] * dN[b, 0] + dN[a, 1] * dN[b, 1]

            # compute the quadrature for that element
            k_ab[a, b] = element_quad(phi_k, nodes)

    return k_ab


def local_force(nodes, f):
    """
    Compute the local force vector.

    Parameters
    ----------
    nodes: 3 x 2 array. The global locations of the nodes of the triangular
        element.
    """

    N = nodes.shape[1] + 1  # N = number of dimensions + 1
    f_b = np.zeros(N)

    # loop over the b reference points
    for b in range(N):
        # define the vector phi
        def phi_f(x, N, dN):
            return N[b] * f(x)

        # compute the quadrature for that element
        f_b[b] = element_quad(phi_f, nodes)

    return f_b


# =============================================================================
# Finite Element Algorithm
# =============================================================================

def finite_element_2d(nodes, IEN, ID, f):
    """
    The finite element algorithm to solve for T.

    Parameters
    ----------
    nodes: N x N array. An array containing the global x, y cooridnates of the
        nodes for the element mesh.
    IEN: n_elements x n_dim + 1 array. An array containing how each element in
        the mesh is related to the nodes.
    ID: 1 x n_elements array. An array linking the global node locations to the
        global equation number in the global stiffness and force matrices.
    f: function. The force function used to calculate the local force vector
        for each element. The function should accept one argument which is an
        array containing the global x and y coordinates for a node.

    Returns
    -------
    T: 1 x nodes array of floats. The array of temperatures at each global
        node, where nodes is the number of nodes.
    """

    # get the number of the equations, elements and nodes
    n_equations = np.max(ID) + 1
    n_elements = IEN.shape[0]
    n_nodes = nodes.shape[0]
    n_dim = nodes.shape[1]
    # generate the location  matrix
    LM = np.zeros_like(IEN.T)
    for e in range(n_elements):
        for a in range(n_dim + 1):
            LM[a, e] = ID[IEN[e, a]]

    # create arrays for global stiffness matrix and global force vector
    K = np.zeros((n_equations, n_equations))
    F = np.zeros((n_equations,))

    # Loop over each element
    for e in range(n_elements):
        # calculate the local stiffness and force was each element
        k_e = local_stiffness(nodes[IEN[e, :], :])
        f_e = local_force(nodes[IEN[e, :], :], f)
        # loop over the reference nodes
        for a in range(n_dim + 1):
            A = LM[a, e]
            for b in range(n_dim + 1):
                B = LM[b, e]
                if (A >= 0) and (B >= 0):
                    # calculate the global stiffness
                    K[A, B] += k_e[a, b]
            if (A >= 0):
                # calculate the global force
                F[A] += f_e[a]

    # Solve the linear equation
    T_A = np.linalg.solve(K, F)
    T = np.zeros(n_nodes)
    for n in range(n_nodes):
        # if ID < 0, don't add the temperature array
        if ID[n] >= 0:
            T[n] = T_A[ID[n]]

    return T


# =============================================================================
# Define Simulation Parameters
# =============================================================================

def force_func(x):
    return np.exp(-(x[0] ** 2 + x[1] ** 2))


# =============================================================================
#
# =============================================================================

nodes, IEN, ID = generate_g_grid(1/12)
T = finite_element_2d(nodes, IEN, ID, force_func)

plt.figure()
plt.axis('equal')
plt.tripcolor(nodes[:, 0], nodes[:, 1], T, triangles=IEN)
plt.colorbar()
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()
