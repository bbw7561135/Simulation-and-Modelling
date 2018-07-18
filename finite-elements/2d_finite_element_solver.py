# =============================================================================
# 2D finite element solver for computing the temperature of a 2D system
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (10, 5)

# =============================================================================
#
# =============================================================================


def shape_function(xi):
    """
    Compute the value of the shape functions at an element's location given
    the reference locations of the nodes, (xi, eta).


    Parameters
    ----------
    xi: 1 x 2 array of floats.
        The reference coordinates of an element (xi, eta).

    Returns
    -------
    N: 3 x 1 array of floats.
        The value of the shape functions at the reference coordinates
        (xi, eta).
    """

    assert(len(xi) == 2), \
        'Only two reference coordinates should be provided.'

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
    dN: 1 x 3 array of ints.
        An array containing the values of the differential at the reference
        coordinates (xi, eta) for an element.
    """

    dN = np.array([[-1, -1], [1, 0], [0, 1]])

    return dN


def local_to_global(nodes, xi):
    """
    Converts the local coordinates of an elements reference coordinates to the
    global coordinates for an element on the mesh.

    Parameters
    ----------
    nodes: 3 x 2 array of floats.
        The global locations of the nodes of the triangular element.
    xi: 1 x 2 array of floats.
        The reference coordinates of an element (xi, eta).

    Returns
    -------
    global_coords: 1 x 2 array.
        The global coordinates of the element on the mesh.
    """

    assert(len(xi) == 2), \
        'Only two reference coordinates should be provided.'
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
    by nodes.

    Parameters
    ----------
    nodes: 3 x 2 array of floats.
        The global locations of the nodes of the triangular element.

    Returns
    -------
    J: 2x2 matrix of floats.
        The Jacobian matrix, i.e. the matrix of derivatives, at the global
        location of an element.
    """

    assert(nodes.shape == (3, 2)), \
        'nodes needs to be an array of shape (3, 2).'

    dN = shape_function_dN()
    J = np.dot(dN.T, nodes)  # costruct the Jacobian matrix

    return J


def det_jacobian(nodes):
    """
    Compute the deterimnant of the Jacobian matrix of an element with a
    global location defined by nodes.

    Parameters
    ----------
    nodes: 3 x 2 array of floats.
        The global locations of the nodes of the triangular element.

    Returns
    -------
    The determinant of the Jacobian matrix for an element with global location
    nodes and reference coordiante (xi, eta).
    """

    assert(nodes.shape == (3, 2)), \
        'nodes needs to be an array of shape (3, 2).'

    J = jacobian(nodes)

    return np.abs(np.linalg.det(J))


def global_dN(nodes):
    """
    Computes the deriatives of the shape functions for an element given its
    global position nodes and reference coordinates (xi, eta).

    Parameters
    ----------
    nodes: 3 x 2 array of floats.
        The global locations of the nodes of the triangular element.

    Returns
    -------
    global_dN: 3 x 2 array of ints.
        The value of the derivatives of the shape functions for an element with
        global location nodes and reference locations (xi, eta).
    """

    assert(nodes.shape == (3, 2)), \
        'nodes needs to be an array of shape (3, 2).'

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
    psi: function, with argument xi.
        This function is used to calculate the volume of the reference element.
        It is applied at each point of the reference element, (xi, eta).

    Returns
    -------
    reference_quad: float.
        The volume of the reference element given by the Gauss quadrature of
        the reference points xi1, xi2 and xi3.
    """

    # the reference points for the quadrature will always be the same, hence
    # define them here
    xi1 = [1/6, 1/6]
    xi2 = [4/6, 1/6]
    xi3 = [1/6, 4/6]

    reference_quad = (1/6) * (psi(xi1) + psi(xi2) + psi(xi3))

    # this quadrature should always be a float, and not an array or etc, so
    # check to make sure nothing weird has happened somewhere
    assert(type(reference_quad) == float or int), \
        'The reference quadrature is being calculated in the wrong format.'

    return reference_quad


def element_quad(phi, nodes):
    """
    Compute the volume of the element given a function phi using Gauss
    quadrature.

    Parameters
    ----------
    nodes: 3 x 2 array of floats.
        The global locations of the nodes of the triangular element.
    phi: function, with arguments nodes, N, dN.
        Where nodes is the global x, y locations of the element, N is the value
        of the shape functions at the reference nodes in the element and dN is
        the derivitive of the shape functions at the reference nodes in the
        element. This function is used to compute the volume of an element.

    Returns
    -------
    quad: float.
        The volume of the element given by the Gauss quadrature.
    """

    assert(nodes.shape == (3, 2)), \
        'nodes needs to be an array of shape (3, 2).'

    def psi(xi):
        return det_jacobian(nodes) * phi(
            local_to_global(nodes, xi), shape_function(xi),
            global_dN(nodes))

    quad = reference_quad(psi)

    return quad


def local_stiffness(nodes):
    """
    Computes the local stiffness matrix for an element given ithe global node
    locations of the element.

    Parameters
    ----------
    nodes: 3 x 2 array of floats.
        The global locations of the nodes of the triangular element.

    Returns
    -------
    k_ab: n_dim + 1 x n_dim + 1 array of floats.
        The local stiffness matrix for an element, where n_dim is the number of
        dimensions of the element.
    """

    assert(nodes.shape == (3, 2)), \
        'nodes needs to be an array of shape (3, 2).'

    # The number of equations is equal to the number of dimensions + 1
    N_eq = nodes.shape[1] + 1
    k_ab = np.zeros((N_eq, N_eq))

    # loop over the node reference points
    for a in range(N_eq):
        for b in range(N_eq):
            # define the vector phi for the element quadrature
            def phi_k(nodes, N, dN):
                return dN[a, 0] * dN[b, 0] + dN[a, 1] * dN[b, 1]

            k_ab[a, b] = element_quad(phi_k, nodes)

    return k_ab


def local_force(nodes, force_func):
    """
    Computes the local force vector of an element given the global node
    locations of the element.

    Parameters
    ----------
    nodes: 3 x 2 array of floats.
        The global locations of the nodes of the triangular element.
    force_func: function.
        Requires one argument which will be a 1 x 2 array containing the x, y
        coordinates of the node location for a reference point in an element.
        The force function used to calculate the local force vector for each
        element. The function should accept one argument which is an array
        containing the global x and y coordinates for a reference point of an
        element.

    Returns
    -------
    f_b: 1 x n_dim + 1 array of floats.
        The local force vector for an element, where n_dim is the number of
        dimensions of the element.
    """

    assert(nodes.shape == (3, 2)), \
        'nodes needs to be an array of shape (3, 2).'

    # The number of equations is equal to the number of dimensions + 1
    N_eq = nodes.shape[1] + 1
    f_b = np.zeros(N_eq)

    # loop over the b reference points
    for b in range(N_eq):
        # define the vector phi
        def phi_f(nodes, N, dN):
            return N[b] * force_func(nodes)

        f_b[b] = element_quad(phi_f, nodes)

    return f_b


# =============================================================================
# Function to generate a 2D mesh - provided by Dr Ian Hawke
# =============================================================================

def generate_2d_grid(Nx):
    """
    Generates a 2D grid and returns the positions of the nodes as well as the
    IEN and ID matrices.
    """

    Nnodes = Nx + 1
    x = np.linspace(0, 1, Nnodes)
    y = np.linspace(0, 1, Nnodes)
    X, Y = np.meshgrid(x, y)
    nodes = np.zeros((Nnodes ** 2, 2))
    nodes[:, 0] = X.ravel()
    nodes[:, 1] = Y.ravel()
    ID = np.zeros(len(nodes), dtype=np.int)
    n_eq = 0
    for nID in range(len(nodes)):
        if nID % Nnodes == Nx:
            ID[nID] = -1
        else:
            ID[nID] = n_eq
            n_eq += 1
    IEN = np.zeros((2 * Nx ** 2, 3), dtype=np.int)
    for i in range(Nx):
        for j in range(Nx):
            IEN[2 * i + 2 * j * Nx, :] = i + j * Nnodes, i + 1 + j * Nnodes, \
                i + (j + 1) * Nnodes
            IEN[2 * i + 1 + 2 * j * Nx, :] = i + 1 + j * Nnodes, \
                i + 1 + (j + 1) * Nnodes, i + (j + 1) * Nnodes

    return nodes, IEN, ID


# =============================================================================
# Finite elements algorithm
# =============================================================================

def finite_element_2d(nodes, IEN, ID, force_func):
    """
    The finite element algorithm to solve for T. This function works by
    consturcting a location matrix to link elements to node. Each element then
    has its local stiffness matrix Kab and local force vector Fb calculated.
    The global stiffness and force matrices are then computed using Kab and Fb
    and used to solve the linear equation K * T = F, which returns the values
    of the temperatures at each node.

    Parameters
    ----------
    nodes: N x 2 array of floats.
        An array containing the global x, y cooridnates of the nodes for the
        element mesh.
    IEN: n_elements x n_dim + 1 array of ints.
        An array containing how each element in the mesh is related to the
        nodes.
    ID: 1 x n_elements array of ints.
        An array linking the global node locations to the global equation
        number in the global stiffness and force matrices.
    force_func: function.
        Requires one argument which will be a 1 x 2 array containing the x, y
        coordinates of the node location for a reference point in an element.
        The force function used to calculate the local force vector for each
        element. The function should accept one argument which is an array
        containing the global x and y coordinates for a reference point of an
        element.

    Returns
    -------
    T: 1 x nodes array of floats.
        The array of temperatures at each global node, where nodes is the
        number of nodes.
    """

    assert(ID.shape[0] == nodes.shape[0]), 'The ID array has incorrect \
        dimensions, it should have as many elements as there are nodes.'
    assert(nodes.shape[1] == 2), \
        'nodes needs to be an array of 2 columns for the x and y coordinates.'
    assert(len(nodes) > 2), 'There are not enough nodes to define a triangular \
        element.'
    assert(IEN.shape[1] == 3), \
        'Each element has 3 node numbers associated to it, hence there should \
        be 3 columns per element.'

    # get the number of the equations, elements and nodes
    N_eq = np.max(ID) + 1
    N_elements = IEN.shape[0]
    N_nodes, N_dim = nodes.shape

    # generate the location matrix which links the node locations to an element
    LM = np.zeros_like(IEN.T)
    for e in range(N_elements):
        for a in range(N_dim + 1):
            LM[a, e] = ID[IEN[e, a]]

    # create arrays for global stiffness matrix and global force vector
    K_global = np.zeros((N_eq, N_eq))
    F_global = np.zeros(N_eq)

    for e in range(N_elements):
        # calculate the local stiffness and force for each element
        k_ab = local_stiffness(nodes[IEN[e, :], :])
        f_b = local_force(nodes[IEN[e, :], :], force_func)

        # loop over the reference coords
        for a in range(N_dim + 1):
            A = LM[a, e]
            for b in range(N_dim + 1):
                B = LM[b, e]
                if (A >= 0) and (B >= 0):
                    # if A or B < 0, then the node will be ignored
                    K_global[A, B] += k_ab[a, b]
            if (A >= 0):
                # if A, then the node will be ignored
                F_global[A] += f_b[a]

    # Solve the linear equation, k * T = F
    T_A = np.linalg.solve(K_global, F_global)

    # now construct the array to contain T at the nodes
    T = np.zeros(N_nodes)
    for node in range(N_nodes):
        # if ID < 0, this temperature should not appear in the global array
        if ID[node] >= 0:
            T[node] = T_A[ID[node]]

    return T


# =============================================================================
# Run the 2D code
# =============================================================================

# simple function
def f1(x):
    return 1.0


nodes1, IEN1, ID1 = generate_2d_grid(10)
T1 = finite_element_2d(nodes1, IEN1, ID1, f1)
x = np.linspace(0, 1)
T1_exact = (1 - x ** 2)/2


# a bit more complicated
def f2(x):
    return x[0] ** 2 * (x[0] - 1) * (
        x[1] ** 2 + 4 * x[1] * (x[1] - 1) + (x[1] - 1) ** 2) + \
            x[1] ** 2 * (3 * x[0] - 1) * (x[1] - 1) ** 2


def f2_sol(x):
    return x[:, 0] ** 2 * (1 - x[:, 0]) * x[:, 1] ** 2 * (1 - x[:, 1]) ** 2/2


nodes2, IEN2, ID2 = generate_2d_grid(40)
T2 = finite_element_2d(nodes2, IEN2, ID2, f2)

# =============================================================================
# Plot the results
# =============================================================================

fig = plt.figure(figsize=(12, 12))

ax1 = fig.add_subplot(211)
ax1.plot(x, T1_exact, '--', label='Exact solution')
ax1.plot(nodes1[:, 0], T1, 'x', label='Finite element')
ax1.set_xlim(nodes1[0, 0], nodes1[-1, 0])
ax1.set_ylim(0)
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$T$")
ax1.legend()

ax2 = fig.add_subplot(212, projection='3d')
ax2.plot_trisurf(nodes2[:, 0], nodes2[:, 1], T2, triangles=IEN2)
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
ax2.set_zlabel(r"$T$")

plt.show()
