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

    return np.hstack((x, y))


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

    return np.abs(np.linalg.det(jacobian(nodes, xi)))


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


def reference_quad(psi):
    """
    Compute the quadrature over the reference triangle.
    """

    xi1 = [1/6, 1/6]
    xi2 = [4/6, 1/6]
    xi3 = [1/6, 4/6]

    return (1/6) * (psi(xi1) + psi(xi2) + psi(xi3))


def element_quad(phi, nodes):
    """
    Compute the quadrature over the element.
    """

    def psi(xi):
        return det_jacobian(nodes, xi) * phi(local_to_global(nodes, xi),
                                             shape_function(xi),
                                             global_dN(xi, nodes))

    return reference_quad(psi)


def local_stiffness(nodes):
    """
    Compute the local stiffness matrix.
    """

    N = nodes.shape[1] + 1  # N = number of dimensions + 1
    k_ab = np.zeros((N, N))

    # loop over the node reference points
    for a in range(N):
        for b in range(N):
            # define the vector psi
            def psi_k(x, N, dN):
                return dN[a, 0] * dN[b, 0] + dN[a, 1] * dN[b, 1]

            # compute the quadrature for that element
            k_ab[a, b] = element_quad(psi_k, nodes)

    return k_ab


def local_force(nodes, f):
    """
    Compute the local force vector.
    """

    N = nodes.shape[1] + 1  # N = number of dimensions + 1
    f_b = np.zeros(N)

    # loop over the b reference points
    for b in range(N):
        # define the vector psi
        def psi_f(x, N, dN):
            return N[b] * f(x)

        # compute the quadrature for that element
        f_b[b] = element_quad(psi_f, nodes)

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

def finite_element_2d(nodes, IEN, ID, f):
    """
    The finite element algorithm to solve for T.
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
# Run the 2D code
# =============================================================================

def f1(x):
    return 1.0


nodes1, IEN1, ID1 = generate_2d_grid(10)
T1 = finite_element_2d(nodes1, IEN1, ID1, f1)
x = np.linspace(0, 1)
T1_exact = (1 - x ** 2)/2


def f2(x):
    return x[0] ** 2 * (x[0] - 1) * (x[1] ** 2 + 4 * x[1] *
                                     (x[1] - 1) + (x[1] - 1) ** 2) + \
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
