# =============================================================================
# A 2D finite element solver using triangular elements
# =============================================================================

import timeit
import pytest
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

    if side_length > 1:
        raise ValueError('side_length should be 1/N, where N is an integer, \
                         hence sidelength should be <= 1.')
    elif side_length == 0:
        raise ValueError('The value of side_length cannot be zero, as the \
                         elements will have no length.')
    elif side_length < 0:
        raise ValueError('side_length is < 0. I\'m not sure what a negative \
                         length is!')

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
    Compute the value of the shape functions at an element's location given
    the reference locations of the nodes, (xi, eta).


    Parameters
    ----------
    xi: 1 x 2 array of floats. The reference coordinates of an element
        (xi, eta).

    Returns
    -------
    N: 3 x 1 array of floats. The value of the shape functions at the reference
        coordinates (xi, eta).
    """

    if len(xi) != 2:
        raise ValueError('Only two reference coordinates should be provided.')

    N = np.array([[1 - xi[1] - xi[0]], [xi[0]], [xi[1]]])

    return N


def shape_function_dN():
    """
    Compute the derivatives of the shape functions at the reference points
    of an element with coordinates (xi, eta).

    As linear shape functions will always be used, the derivatives will either
    be 0, -1 or 1.

    Returns
    -------
    dN: 1 x 3 array of ints. An array containing the values of the
        differential at the reference coordinates (xi, eta) for an element.
    """

    dN = np.array([[-1, -1], [1, 0], [0, 1]])

    return dN


def local_to_global(nodes, xi):
    """
    Converts the local coordinates of an elements reference coordinates to the
    global coordinates for an element on the mesh.

    Parameters
    ----------
    nodes: 3 x 2 array of floats. The global locations of the nodes of the
        triangular element.
    xi: 1 x 2 array of floats. The reference coordinates of an element
        (xi, eta).

    Returns
    -------
    global_coords: 1 x 2 array. The global coordinates of the element on the
    mesh.

    """

    if len(xi) != 2:
        raise ValueError('Only two reference coordinates should be provided.')
    assert(nodes.shape == (3, 2)), \
        'nodes needs to be an array of shape (3, 2).'

    N = shape_function(xi)

    x = nodes[0, 0] * N[0] + nodes[1, 0] * N[1] + nodes[2, 0] * N[2]
    y = nodes[0, 1] * N[0] + nodes[1, 1] * N[1] + nodes[2, 1] * N[2]

    global_coords = np.hstack((x, y))

    return global_coords


def jacobian(nodes):
    """
    Compute the Jacobian matrix of an element with a global location defined
    by nodes and reference locations defined by the reference coordinates
    (xi, eta) in xi.

    Parameters
    ----------
    nodes: 3 x 2 array of floats. The global locations of the nodes of the
        triangular element.

    Returns
    -------
    J: 2x2 matrix of floats. The Jacobian matrix, i.e. the matrix of
        derivatives, at the global location of an element.
    """

    assert(nodes.shape == (3, 2)), \
        'nodes needs to be an array of shape (3, 2).'

    dN = shape_function_dN()    # calculate the derivatives of the shape funcs
    J = np.dot(dN.T, nodes)     # costruct the Jacobian matrix

    return J


def det_jacobian(nodes, xi):
    """
    Compute the deterimnant of the Jacobian matrix of an element with a
    global location defined by nodes and reference locations defined by the
    reference coordinates (xi, eta) in xi.

    Parameters
    ----------
    nodes: 3 x 2 array of floats. The global locations of the nodes of the
        triangular element.
    xi: 1 x 2 array of floats. The local coordinates of the  element at the
        reference coordinates (xi, eta).

    Returns
    -------
    The determinant of the Jacobian matrix for an element with global location
    nodes and reference coordiante (xi, eta).
    """

    if len(xi) != 2:
        raise ValueError('Only two reference coordinates should be provided.')
    assert(nodes.shape == (3, 2)), \
        'nodes needs to be an array of shape (3, 2).'

    return np.abs(np.linalg.det(jacobian(nodes)))


def global_dN(nodes, xi):
    """
    Computes the deriatives of the shape functions for an element given its
    global position nodes and reference coordinates (xi, eta).

    Parameters
    ----------
    nodes: 3 x 2 array of floats. The global locations of the nodes of the
        triangular element.
    xi: 1 x 2 array of floats. The local coordinates of the  element at the
        reference coordinates (xi, eta).

    Returns
    -------
    global_dN: 3 x 2 array of ints. The value of the derivatives of the shape
        functions for an element with global location nodes and reference
        locations (xi, eta).
    """

    if len(xi) != 2:
        raise ValueError('Only two reference coordinates should be provided.')
    assert(nodes.shape == (3, 2)), \
        'nodes needs to be an array of shape (3, 2).'

    # create the local derivs and the Jacobian matrix
    dN = shape_function_dN()
    J = jacobian(nodes)
    global_dN = np.zeros_like(dN)

    # solve the linear equation global_dN * J = dN
    for i in range(3):
        global_dN[i, :] = np.linalg.solve(J, dN[i, :])

    return global_dN


def reference_quad(psi):
    """
    Computes the volume of the reference triangle for an element using Gauss
    quadrature given a function psi.

    Parameters
    ----------
    psi: function, with arguments nodes, N, dN. Where nodes is the global x, y
        locations of the element, N is the value of the shape functions at
        the reference nodes in the element and dN is the derivitive of the
        shape functions at the reference nodes in the element. This function
        is used to compute the volume of an element.

    Returns
    -------
    reference_quad: float. The volume of the reference element given by the
    Gauss quadrature of the reference points xi1, xi2 and xi3.
    """

    xi1 = [1/6, 1/6]
    xi2 = [4/6, 1/6]
    xi3 = [1/6, 4/6]

    reference_quad = (1/6) * (psi(xi1) + psi(xi2) + psi(xi3))

    return reference_quad


def element_quad(phi, nodes):
    """
    Compute the volume of the element given a function phi using Gauss
    quadrature.

    Parameters
    ----------
    nodes: 3 x 2 array of floats. The global locations of the nodes of the
        triangular element.
    phi: function, with arguments nodes, N, dN. Where nodes is the global x, y
        locations of the element, N is the value of the shape functions at
        the reference nodes in the element and dN is the derivitive of the
        shape functions at the reference nodes in the element. This function
        is used to compute the volume of an element.

    Returns
    -------
    quad: float. The volume of the element given by the Gauss quadrature.
    """

    assert(nodes.shape == (3, 2)), \
        'nodes needs to be an array of shape (3, 2).'

    def psi(xi):
        return det_jacobian(nodes, xi) * phi(local_to_global(nodes, xi),
                                             shape_function(xi),
                                             global_dN(nodes, xi))
    quad = reference_quad(psi)

    return quad


def local_stiffness(nodes):
    """
    Computes the local stiffness matrix for an element given ithe global node
    locations of the element.

    Parameters
    ----------
    nodes: 3 x 2 array of floats. The global locations of the nodes of the
        triangular element.
    heat_source: function, requires one argument which will be a 1 x 2 array
        containing the x, y coordinates of the node location for a reference
        point in an element. The force function used to calculate the local
        force vector for each element. The function should accept one argument
        which is an array containing the global x and y coordinates for a
        reference point of an element.

    Returns
    -------
    k_ab: n_dim + 1 x n_dim + 1 array of floats. The local stiffness matrix for
        an element, where n_dim is the number of dimensions of the element.
    """

    assert(nodes.shape == (3, 2)), \
        'nodes needs to be an array of shape (3, 2).'

    N_eq = nodes.shape[1] + 1  # N_eq = number of dimensions + 1
    k_ab = np.zeros((N_eq, N_eq))

    # loop over the node reference points
    for a in range(N_eq):
        for b in range(N_eq):
            # define the vector phi
            def phi_k(nodes, N, dN):
                return dN[a, 0] * dN[b, 0] + dN[a, 1] * dN[b, 1]

            # compute the quadrature for that element
            k_ab[a, b] = element_quad(phi_k, nodes)

    return k_ab


def local_force(nodes, heat_source):
    """
    Computes the local force vector of an element given the global node
    locations of the element.

    Parameters
    ----------
    nodes: 3 x 2 array of floats. The global locations of the nodes of the
        triangular element.
    heat_source: function, requires one argument which will be a 1 x 2 array
        containing the x, y coordinates of the node location for a reference
        point in an element. The force function used to calculate the local
        force vector for each element. The function should accept one argument
        which is an array containing the global x and y coordinates for a
        reference point of an element.

    Returns
    -------
    f_b: 1 x n_dim + 1 array of floats. The local force vector for an element,
        where n_dim is the number of dimensions of the element.
    """

    assert(nodes.shape == (3, 2)), \
        'nodes needs to be an array of shape (3, 2).'

    N_eq = nodes.shape[1] + 1  # N_eq = number of dimensions + 1
    f_b = np.zeros(N_eq)

    # loop over the b reference points
    for b in range(N_eq):
        # define the vector phi
        def phi_f(nodes, N, dN):
            return N[b] * heat_source(nodes)

        # compute the quadrature for that element
        f_b[b] = element_quad(phi_f, nodes)

    return f_b


# =============================================================================
# Finite Element Algorithm
# =============================================================================

def finite_element_2d(nodes, IEN, ID, heat_source):
    """
    The finite element algorithm to solve for T. This function works by
    consturcting a location matrix to link elements to node. Each element then
    has its local stiffness matrix Kab and local force vector Fb calculated.
    The global stiffness and force matrices are then computed using Kab and Fb
    and used to solve the linear equation K * T = F, which returns the values
    of the temperatures at each node.

    Parameters
    ----------
    nodes: N x N array of floats. An array containing the global x, y
        cooridnates of the nodes for the element mesh.
    IEN: n_elements x n_dim + 1 array of ints. An array containing how each
        element in the mesh is related to the nodes.
    ID: 1 x n_elements array of ints. An array linking the global node
        locations to the global equation number in the global stiffness and
        force matrices.
    heat_source: function, requires one argument which will be a 1 x 2 array
        containing the x, y coordinates of the node location for a reference
        point in an element. The force function used to calculate the local
        force vector for each element. The function should accept one argument
        which is an array containing the global x and y coordinates for a
        reference point of an element.

    Returns
    -------
    T: 1 x nodes array of floats. The array of temperatures at each global
        node, where nodes is the number of nodes.
    """

    assert(ID.shape[0] == nodes.shape[0]), 'The ID array has incorrect \
        dimensions, it should have as many elements as there are nodes.'
    assert(nodes.shape[1] == 2), \
        'nodes needs to be an array of 2 columns for the x and y coordinates.'
    assert(IEN.shape[1] == 3), \
        'Each element has 3 node numbers associated to it, hence there should \
        be 3 columns per element.'

    # get the number of the equations, elements and nodes
    n_eq = np.max(ID) + 1
    n_e = IEN.shape[0]
    n_nodes, n_dim = nodes.shape

    # generate the location matrix which links the node locations to an element
    LM = np.zeros_like(IEN.T)
    for e in range(n_e):
        for a in range(n_dim + 1):
            LM[a, e] = ID[IEN[e, a]]

    # create arrays for global stiffness matrix and global force vector
    K_global = np.zeros((n_eq, n_eq))
    F_global = np.zeros(n_eq)

    # Loop over each element
    for e in range(n_e):
        # calculate the local stiffness and force for each element
        k_ab = local_stiffness(nodes[IEN[e, :], :])
        f_b = local_force(nodes[IEN[e, :], :], heat_source)
        # loop over the reference coords
        for a in range(n_dim + 1):
            A = LM[a, e]
            for b in range(n_dim + 1):
                B = LM[b, e]
                if (A >= 0) and (B >= 0):
                    # calculate the global stiffness
                    # if A or B < 0, then the node will be ignored
                    K_global[A, B] += k_ab[a, b]
            if (A >= 0):
                # calculate the global force
                F_global[A] += f_b[a]

    # Solve the linear equation
    T_A = np.linalg.solve(K_global, F_global)

    # now construct the array to contain T at the nodes
    T = np.zeros(n_nodes)
    for node in range(n_nodes):
        # if ID < 0, this node should not appear in the global temperature
        if ID[node] >= 0:
            T[node] = T_A[ID[node]]

    return T


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_temperature_triplot(nodes, IEN, tri_size, T, cmap='hot'):
    """
    Plot the temperature of the grid using the tripcolor function in matplotlib
    to plot the temperature as a colourmap.

    Parameters
    ----------
    nodes: N x N array of floats. An array containing the global x, y
        cooridnates of the nodes for the element mesh.
    IEN: n_elements x n_dim + 1 array of ints. An array containing how each
        element in the mesh is related to the nodes.
    tri_size: The length of the short side of the triangular elements.
    T: 1 x nodes array of floats. The array of temperatures at each global
        node, where nodes is the number of nodes.
    cmap: string. The name of the desired colourmap to be used when plotting
        the temperature. By default the colourmap is hot.

    Returns
    -------
    A colour plot showing the temperature of each element. The plot is printed
    to the console and also saved to the working directory with the filename
    temperature_tri_size={} where {} is the length of the short side of the
    triangular element.
    """

    assert(nodes.shape[1] == 2), \
        'The nodes array is in the wrong format. It should be n_nodes x n_dims'

    plt.tripcolor(nodes[:, 0], nodes[:, 1], T, triangles=IEN, cmap=cmap)
    colourbar = plt.colorbar()
    colourbar.set_label(r'Temperature, $T$', labelpad=25, rotation=270)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.axis('equal')
    plt.savefig('temperature_tri_size={:4.3f}.pdf'.format(tri_size))
    plt.show()


def plot_temperature_tri_surf(nodes, IEN, tri_size, T):
    """
    Plot the temperature of the grid using the tri_surf function in matplotlib
    to plot the temperature as the z-direction.

    Parameters
    ----------
    nodes: N x N array of floats. An array containing the global x, y
        cooridnates of the nodes for the element mesh.
    IEN: n_elements x n_dim + 1 array of ints. An array containing how each
        element in the mesh is related to the nodes.
    tri_size: The length of the short side of the triangular elements.
    T: 1 x nodes array of floats. The array of temperatures at each global
        node, where nodes is the number of nodes.

    Returns
    -------
    A surface plot showing the temperature of each element. The plot is printed
    to the console and also saved to the working directory with the filename
    temperature_tri_surf={} where {} is the length of the short side of the
    triangular element.
    """

    assert(nodes.shape[1] == 2), \
        'The nodes array is in the wrong format. It should be n_nodes x n_dims'

    fig = plt.figure()

    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_trisurf(nodes[:, 0], nodes[:, 1], T, triangles=IEN)
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    ax1.set_zlabel(r'Temperature, $T$')
    ax1.view_init(elev=30, azim=40)

    plt.savefig('temperature_tri_surf={:4.3f}.pdf'.format(tri_size))
    plt.show()


# =============================================================================
# Define Heat Soure
# =============================================================================

def heat_source(x):
    """
    The function used to define the heat source.

    Parameters
    ----------
    x: 1 x 2 array of floats. The x, y coordinates of the global location for
        a reference point of an element.

    Returns
    -------
    Float. The value of the heat source at the reference point for an element.
    """

    assert(len(x) == 2), 'The array does not contain two coordinate locations.'

    return np.exp(-(x[0] ** 2 + x[1] ** 2))


# =============================================================================
# Run the functions and plot the results
# =============================================================================

if __name__ == '__main__':

    # call pytest to test the different fucntions
    pytest.main(['-v'])

    start = timeit.default_timer()

    # define the size of the triangular elements, this must be < 1
    tri_size = 1/2
    # construct the grid
    nodes, IEN, ID = generate_g_grid(tri_size)
    # solve for the temperature using finite elements and plot the results
    T = finite_element_2d(nodes, IEN, ID, heat_source)
    plot_temperature_triplot(nodes, IEN, tri_size, T)
    plot_temperature_tri_surf(nodes, IEN, tri_size, T)

    stop = timeit.default_timer()

    print('Run time: {:6.2f} seconds.'.format(stop - start))
