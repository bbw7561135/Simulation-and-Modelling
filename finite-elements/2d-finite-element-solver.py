# =============================================================================
# 2D finite element solver for computing the temperature of a 2D system
#
# =============================================================================

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12, 6)

# =============================================================================
#
# =============================================================================


def shape_function(xi):
    """
    Compute the shape functions at a node (xi, eta).

    Parameters
    ----------
    x: array. The coordinates of the reference elements.
    """

    return np.array([[1 - xi[1] - xi[0]], [xi[0]], [xi[1]]])


def shape_function_dN(xi):
    """
    Compute the differential at a node (xi, eta).
    """

    return np.array([[-1, -1], [1, 0], [0, 1]])


def local_to_global(nodes, xi):
    """
    Convert local coordinates into global coordinates.
    """
    N = shape_function(xi)

    x = nodes[0, 0] * N[0] + nodes[1, 0] * N[1] + nodes[2, 0] * N[2]
    y = nodes[0, 1] * N[0] + nodes[1, 1] * N[1] + nodes[2, 1] * N[2]

    return x, y


def jacobian(nodes, xi):
    """
    Compute the Jacobian matrix at a node (xi, eta).
    """
    dN = shape_function_dN(xi)

    return np.dot(dN.T, nodes)


def det_jacobian(nodes, xi):
    """
    Compute the determinant of the Jacobian matrix at a node (xi, eta).
    """

    return np.linalg.det(jacobian(nodes, xi))


def global_dN(xi, nodes):
    """
    Compute the global derivatives using the Jacobian matrix.
    """

    dN = shape_function_dN(xi)
    J = jacobian(nodes, xi)
    global_dN = np.zeros((3, 2))

    for i in range(3):
        global_dN[i, :] = np.linalg.solve(J, dN[i, :])

    return global_dN


# =============================================================================
#
# =============================================================================

nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
xi = [1/6, 1/6]













IEN = np.array([[0, 1, 2], [1, 3, 2]])  # local node number
ID = np.array([0, -1, 1, -1])  # destination array, links the global node
# number to the global equation number

# the location matrix links the nodes to the equation number
LM = np.zeros_like(IEN.T)
for e in range(IEN.shape[0]):
    for a in range(IEN.shape[1]):
        LM[a, e] = ID[IEN[e, a]]


