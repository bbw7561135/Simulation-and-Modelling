import pytest
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


# =============================================================================
# Hartree-Fock Algorithm Functions
# =============================================================================

def transformation_matrix(S):
    """
    Compute the transformation matrix given the overlap matrix S.

    Parameters
    ----------
    S: (n orbitals x n orbitals) array of floats.
        An array containing the coefficents for the overlap matrix.

    Returns
    -------
    X: (n orbitals x n orbitals) array of floats.
        The transformation matrix given by the eigenvectors and eigenvalues
        of the overlap matrix S.
    """

    # for each matrix:
    # check that the elements are finite, that there are elements and that
    # the matrix is in the correct form
    assert(np.all(np.isfinite(S))), 'The entries of S are not finite.'
    assert(S.shape[0] > 0), 'S has no entries.'
    assert(S.shape[0] == S.shape[1]), 'S is not square.'

    eigvals, eigvecs = np.linalg.eig(S)
    X = np.dot(eigvecs, np.dot(np.diag(eigvals ** (-0.5)),
                               np.conj(np.transpose(eigvecs))))

    return X


def density_matrix(C, n_electrons):
    """
    Compute the density matrix given the coefficents of the molecular orbit
    basis coefficents.

    Parameters
    ----------
    C: (n orbitals x n orbitals) array of floats.
        The coefficents for the molecular orbit basis functions.
    n_electrons: int.
        The number of electrons in the system.

    Returns
    -------
    D: (n orbitals x n orbitals) array of floats.
        The density matrix of the system.
    """

    # for each matrix:
    # check that the elements are finite, that there are elements and that
    # the matrix is in the correct form
    assert(np.all(np.isfinite(C))), 'The entries of C are not finite.'
    assert(C.shape[0] > 0), 'C has no entries.'
    assert(C.shape[0] == C.shape[1]), 'C is not square.'

    assert(np.isfinite(n_electrons)), 'n_electrons is not finite.'
    assert(type(n_electrons) == int), \
        'The number of electrons is an int amount.'
    if n_electrons <= 0:
        raise ValueError('The number of electrons is positive and non-zero.')

    D = np.zeros_like(C)
    n = C.shape[0]

    for mu in range(n):
        for nu in range(n):
            for j in range(n_electrons//2):  # use int division
                D[mu, nu] += 2 * C[mu, j] * C[nu, j]

    return D


def flock_matrix(H, G, D):
    """
    Compute the Flock matrix of the system given the one-electron integral
    matrix H, the two-electron integrals matrix G and the density matrix D.

    Parameters
    ----------
    H: (n orbitals x n orbitals) array of floats.
        The one-electron integral coefficent matrix.
    G: (n orbitals x n orbialts x n orbitals x n orbitals) array of floats.
        The two-electron integral coefficent matrix.
    D: (n orbitals x n orbitals) array of floats.
        The density matrix for the system.

    Returns
    -------
    F: (n orbitals x n orbitals) array of floats.
        The Flock matrix for the system.
    """

    # for each matrix:
    # check that the elements are finite, that there are elements and that
    # the matrix is in the correct form, kind of a weak check as the rows
    # could be correct, but the columns incorrect :-(
    assert(H.shape[0] == G.shape[0] == D.shape[0]), \
        'The array shapes are not compatiable.'

    assert(np.all(np.isfinite(H))), 'The entries of H are not finite.'
    assert(H.shape[0] > 0), 'H has no entries.'
    assert(H.shape[0] == H.shape[1]), 'H is not square.'

    assert(np.all(np.isfinite(G))), 'The entries of G are not finite.'
    assert(G.shape[0] > 0), 'G has no entries.'
    assert(G.shape[0] == G.shape[1] == G.shape[2] == G.shape[3]), \
        'G has incorrect dimensions.'

    assert(np.all(np.isfinite(D))), 'The entries of D are not finite.'
    assert(D.shape[0] > 0), 'D has no entries.'
    assert(D.shape[0] == D.shape[1]), 'D is not square.'

    F = np.zeros_like(H)
    n = F.shape[0]

    # set the elements of F to be H, makes the next bit of the code smaller
    # as F = H + G * D
    F[:, :] = H[:, :]
    assert(np.all(F) == np.all(H)), \
        'Value of H have not been assigned to F properly.'

    for mu in range(n):
        for nu in range(n):
            for alpha in range(n):
                for beta in range(n):
                    F[mu, nu] += (G[mu, nu, alpha, beta] - 0.5 *
                                  G[mu, beta, alpha, nu]) * D[alpha, beta]

    return F


def orbital_values(F, X):
    """
    Diagonalise the Flock matrix and compute the oribal energies (eigenvalues),
    and the coefficents of the molecular orbitals basis functions
    (eigenvectors).

    Parameters
    ----------
    F: (n orbitals x n orbitals) array of floats.
        The Flock matrix for the system.
    X: (n orbitals x n orbitals) array of floats.
        The transformation matrix.

    Returns
    -------
    orb_eng: (n orbials x 1) array of floats.
        The eigenvalues of the diagonalised Flock matrix.
    C: (n orbitals x n orbitals) array of floats.
        The eigenvectors of the diagonalised Flock matrix, which are the
        coefficents for the molecular orbits basis functions.
    """

    # for each matrix:
    # check that the elements are finite, that there are elements and that
    # the matrix is in the correct form
    assert(np.all(np.isfinite(F))), 'The entries of F are not finite.'
    assert(F.shape[0] > 0), 'F has no entries.'
    assert(F.shape[0] == F.shape[1]), 'F is not square.'

    assert(np.all(np.isfinite(X))), 'The entries of G are not finite.'
    assert(X.shape[0] > 0), 'X has no entries.'
    assert(X.shape[0] == X.shape[1]), 'X is not square.'

    # diagonalise the Flock matrix
    Fprime = np.dot(np.conj(np.transpose(X)), np.dot(F, X))
    eigvals, eigvecs = np.linalg.eigh(Fprime)

    # putting the eigenvalues and eigenvectors into the correct order by
    # sorting the eigenvalues from smallest to largest and remembering to
    # rearrange the eigenvectors as well
    idx = eigvals.argsort()
    orb_eng = eigvals[idx]
    orb_coefs = eigvecs[:, idx]

    assert(orb_eng.shape == eigvals.shape), 'Eigenvalue have been lost!'
    assert(orb_coefs.shape == eigvecs.shape), 'Eigenvectors have been lost!'

    C = np.dot(X, orb_coefs)

    return orb_eng, C


def HF_step(X, H, G, C, D, n_electrons):
    """
    The set of steps required to compute a step using the Hartree-Fock method.
    The function will take in the number of electrons, the density matrix,
    the transformation matrix, the molecular orbit coefficent matrix and the
    electron integral matrcies.

    These are used to compute the Fock matrix of the system and then update the
    molecular orbit coefficents and new density matrix.

    Parameters
    ----------
    X: (n orbitals x n orbitals) array of floats.
        The transformation matrix.
    H: (n orbitals x n orbitals) array of floats.
        The one-electron integral coefficent matrix.
    G: (n orbitals x n orbialts x n orbitals x n orbitals) array of floats.
        The two-electron integral coefficent matrix.
    D: (n orbitals x n orbitals) array of floats.
        The density matrix for the system.
    C: (n orbitals x n orbitals) array of floats.
        The coefficents for the molecular orbit basis functions.
    n_electrons: int.
        The number of electrons in the system.

    Returns
    -------
    new_D: (n orbitals x n orbitals) array of floats.
        The updated density matrix of the system.
    orb_eng: (n orbitals x 1) array of floats.
        The eigenvalues of the diagonalised Flock matrix.
    new_C: (n orbitals x n orbitals) array of floats.
        The updated coefficents for the molecular orbit basis functions.
    """

    # for each matrix:
    # check that the elements are finite, that there are elements and that
    # the matrix is in the correct form
    assert(np.all(np.isfinite(X))), 'The entries of G are not finite.'
    assert(X.shape[0] > 0), 'X has no entries.'
    assert(X.shape[0] == X.shape[1]), 'X is not square.'

    assert(np.all(np.isfinite(H))), 'The entries of H are not finite.'
    assert(H.shape[0] > 0), 'H has no entries.'
    assert(H.shape[0] == H.shape[1]), 'H is not square.'

    assert(np.all(np.isfinite(G))), 'The entries of G are not finite.'
    assert(G.shape[0] > 0), 'G has no entries.'
    assert(G.shape[0] == G.shape[1] == G.shape[2] == G.shape[3]), \
        'G has incorrect dimensions.'

    assert(np.all(np.isfinite(D))), 'The entries of D are not finite.'
    assert(D.shape[0] > 0), 'D has no entries.'
    assert(D.shape[0] == D.shape[1]), 'D is not square.'

    assert(np.all(np.isfinite(C))), 'The entries of C are not finite.'
    assert(C.shape[0] > 0), 'C has no entries.'
    assert(C.shape[0] == C.shape[1]), 'C is not square.'

    assert(np.isfinite(n_electrons)), 'n_electrons is not finite.'
    assert(type(n_electrons) == int), \
        'The number of electrons is an int amount.'
    if n_electrons <= 0:
        raise ValueError('The number of electrons is positive and non-zero.')

    F = flock_matrix(H, G, D)
    orb_eng, new_C = orbital_values(F, X)
    new_D = density_matrix(new_C, n_electrons)

    return new_D, orb_eng, new_C


def total_energy(D, H, F, Vnn):
    """
    Compute the total energy of the system using the density matrix D, the
    flock matrix F, the one-electron integral H and the nucleon repulsion
    energy.

    Parameters
    ----------
    D: (n orbitals x n orbitals) array of floats.
        The density matrix for the system.
    H: (n orbitals x n orbitals) array of floats.
        The one-electron integral coefficent matrix.
    F: (n orbitals x n orbitals) array of floats.
        The Flock matrix of the system.
    Vnn: float.
        The value of the nucleon repulsion energy of the system.

    Returns
    -------
    Etot: float.
        The total energy of the system.
    """

    # for each matrix:
    # check that the elements are finite, that there are elements and that
    # the matrix is in the correct form
    assert(np.all(np.isfinite(D))), 'The entries of D are not finite.'
    assert(D.shape[0] > 0), 'D has no entries.'
    assert(D.shape[0] == D.shape[1]), 'D is not square.'

    assert(np.all(np.isfinite(H))), 'The entries of H are not finite.'
    assert(H.shape[0] > 0), 'H has no entries.'
    assert(H.shape[0] == H.shape[1]), 'H is not square.'

    assert(np.all(np.isfinite(F))), 'The entries of F are not finite.'
    assert(F.shape[0] > 0), 'F has no entries.'
    assert(F.shape[0] == F.shape[1]), 'F is not square.'

    assert(np.isfinite(Vnn)), 'Vnn is not finite.'
    assert(type(Vnn) is float or int), 'Vnn is of type float or int.'

    n = D.shape[0]

    # set Etot as Vnn as the total energy of the system includes the nucelon
    # repulsion energy
    Etot = Vnn
    assert(Etot == Vnn), 'Vnn is not being added to Etot properly.'
    for mu in range(n):
        for nu in range(n):
            Etot += 0.5 * D[mu, nu] * (H[mu, nu] + F[mu, nu])

    return Etot


def matrix_diff(D, DNew):
    """
    Calculate the difference between two density matrices. In this case
    the difference between an old and updated density matrix.

    Parameters
    ----------
    D: (n orbitals x n orbitals) array of floats.
        The old density matrix of the system.
    DNew: (n orbitals x n orbitals) array of floats.
        The updated density matrix of the system.

    Returns
    -------
    difference: float.
        The difference between the two density matrices.
    """

    assert(D.shape == DNew.shape), \
        'The densitry matrices are different sizes.'

    n = D.shape[0]

    # loop over each element of the matrices and subtract the elements
    difference = 0
    for mu in range(n):
        for nu in range(n):
            difference += (D[mu, nu] - DNew[mu, nu]) ** 2

    difference = np.sqrt(difference)

    return difference


def HF_iteration(S, H, G, C, Vnn, n_electrons, tol=1e-6):
    """
    Iterates the given matrices using the Hatree-Fock method until the
    density matrix converges to a given tolerence. Converge is calculated by
    calculating the difference between the old and updated density matrix
    for each iteration.

    Parameters
    ----------
    S: (n orbitals x n orbitals) array of floats.
        An array containing the coefficents for the overlap matrix.
    H: (n orbitals x n orbitals) array of floats.
        The one-electron integral coefficent matrix.
    G: (n orbitals x n orbialts x n orbitals x n orbitals) array of floats.
        The two-electron integral coefficent matrix.
    C: (n orbitals x n orbitals) array of floats.
        The coefficents for the molecular orbit basis functions.
    Vnn: float.
        The value of the nucleon repulsion energy of the system.
    tol: float.
        The tolerance used to determine when the system has converged, by
        default this is 1e-6.

    Returns
    -------
    Etot: float.
        The total energy of the system, including the nuclear repulsion energy.
    C: (n orbitals x n orbitals) array of floats.
        The coefficents for the molecular orbits once the system has converged.
    OE: (1 x n orbitals) array of floats.
        The energies of each molecular orbit.
    """

    # for each matrix:
    # check that the elements are finite, that there are elements and that
    # the matrix is in the correct form
    assert(np.all(np.isfinite(S))), 'The entries of S are not finite.'
    assert(S.shape[0] > 0), 'C has no entries.'
    assert(S.shape[0] == S.shape[1]), 'S is not square.'

    assert(np.all(np.isfinite(H))), 'The entries of H are not finite.'
    assert(H.shape[0] > 0), 'H has no entries.'
    assert(H.shape[0] == H.shape[1]), 'H is not square.'

    assert(np.all(np.isfinite(G))), 'The entries of G are not finite.'
    assert(G.shape[0] > 0), 'G has no entries.'
    assert(G.shape[0] == G.shape[1] == G.shape[2] == G.shape[3]), \
        'G has incorrect dimensions.'

    assert(np.all(np.isfinite(C))), 'The entries of C are not finite.'
    assert(C.shape[0] > 0), 'C has no entries.'
    assert(C.shape[0] == C.shape[1]), 'C is not square.'

    # check there are the correct number of electrons, i.e. there has to be
    # at least one and that it is an int amount
    assert(np.isfinite(n_electrons)), 'n_electrons is not finite.'
    assert(type(n_electrons) == int), \
        'The number of electrons is an int amount.'
    if n_electrons <= 0:
        raise ValueError('The number of electrons is positive and non-zero.')

    assert(np.isfinite(Vnn)), 'Vnn is not finite.'
    assert(type(Vnn) is float or int), 'Vnn is of type float or int.'



    X = transformation_matrix(S)
    D = density_matrix(C, n_electrons)

    # set diff to be large otherwise the while loop would stop immediately
    diff = 10 * tol
    current_iteration = 0
    max_iteration = 100

    # continue to iterate the system until it converges, i.e. when the
    # difference between D and DNew is less than or equal to the tolerence
    while (diff > tol) and (current_iteration < max_iteration):
        current_iteration += 1
        DNew, orb_engs, C = HF_step(X, H, G, C, D, n_electrons)
        diff = matrix_diff(D, DNew)
        D = DNew

    if (diff < tol):
        F = flock_matrix(H, G, D)
        Etot = total_energy(D, H, F, Vnn)
        print('Total energy: {:3.3f} Hartrees\nIterations: {}'
              .format(Etot, current_iteration))

    else:
        print('Ruh-roh, failed to converge!')

        # if the sytem doesn't converge, return None
        Etot = C = orb_engs = None

    return Etot, C, orb_engs


# =============================================================================
# Constructing the Basis Fuctions & Molecular Orbits
# =============================================================================

def basis(R, a, c):
    """
    Calculate the value of the STO-3G basis function at a given distance
    r from a nucelus.

    Parameters
    ----------
    R: float.
        The distance between a point in space and the nucleus of the orbit.
    a: (1 x 3) array of floats.
        The a coefficents for the orbit for the STO-3G basis functions.
    c: (1 x 3) array of floats.
        The c coefficents for the orbit for the ST0-3G basis functions.

    Returns
    -------
    chi: float.
        The value of the basis function at the distance R from the nucelus.
    """

    assert(type(R) is float or int), 'R is of type float or int.'
    assert(a.size == 3), 'Incorrect number of a coefficents supplied.'
    assert(c.size == 3), 'Incorrect number of c coefficents supplied.'

    n_phi = 3  # number of gaussians making up the basis function

    chi = 0  # iterate through each value of a, c, and xi
    for i in range(n_phi):
        # if orb > 1 and orb < 5:
        #    chi += R * (c[i] * (((2 * a[i])/np.pi) ** (3/4)) *
        #                np.exp(-(a[i]) * R ** 2))
        # else:
        chi += c[i] * (((2 * a[i])/np.pi) ** (3/4)) * \
            np.exp(-(a[i]) * R ** 2)

    return chi


def basis_functions(R, C):
    """
    Construct the basis functions for an orbital for a given distance R from
    the nucelus. The function will output the orbital basis functions as:
        Entry 0: Oxygen 1s
        Entry 1: Oxygen 2s
        Entry 2: Oxygen 2p (x)
        Entry 3: Oxygen 2p (y)
        Entry 4: Oxygen 2p (z)
        Entry 5: Hydrogen 1 1s
        Entry 6: Hydorgen 2 1s

    Parameters
    ----------
    R: (1 x n orbitals) array of floats.
        The distance from a point in space to the nucleus of the orbital.
    C: (n orbitals x n orbitals) array of floats.
        The coefficents for the molecular orbit basis functions. Used to figure
        out the number of orbitals.

    Returns
    -------
    basis_functions: (1 x n_orbitals) array of floats.
        The value of the basis functions for the orbitals at a position in
        space R.
    """

    n_orbitals = C.shape[0]
    assert(R.size == n_orbitals), \
        'There are not the same amount of orbitals and distances'

    # the values for xi for the atom orbitals
    xi1_H = 1.24 ** 2
    xi1_O = 7.66 ** 2
    xi2_O = 2.25 ** 2

    # the a coefficents in order of the orbitals given
    a = np.array(
        [[0.1098180 * xi1_O, 0.405771 * xi1_O, 2.227660 * xi1_O],
         [0.7513860 * xi2_O, 0.231031 * xi2_O, 0.994203 * xi2_O],
         [0.0751386 * xi2_O, 0.231031 * xi2_O, 0.994203 * xi2_O],
         [0.0751386 * xi2_O, 0.231031 * xi2_O, 0.994203 * xi2_O],
         [0.0751386 * xi2_O, 0.231031 * xi2_O, 0.994203 * xi2_O],
         [0.1098180 * xi1_H, 0.405771 * xi1_H, 2.227660 * xi1_H],
         [0.1098180 * xi1_H, 0.405771 * xi1_H, 2.227660 * xi1_H]])

    # the c coefficents in order of the orbitals given
    c = np.array(
        [[0.444635, 0.535328, 0.1543290],
         [0.700115, 0.399513, -0.999672],
         [0.391957, 0.607684, 0.1559163],
         [0.391957, 0.607684, 0.1559163],
         [0.391957, 0.607684, 0.1559163],
         [0.444635, 0.535328, 0.1543290],
         [0.444635, 0.535328, 0.1543290]])

    # construct the basis functions for an orbital
    basis_functions = np.zeros(n_orbitals)
    for mu in range(n_orbitals):
        basis_functions[mu] = basis(R[mu], a[mu, :], c[mu, :])

    return basis_functions


def molecular_orbits(C, r):
    """
    Compute the molecular orbit value at a point r in the x-y plane. This
    is done for all the orbitals of the system. The function will assumes
    and outputs the molecular orbits functions as:
        Entry 0: Oxygen 1s
        Entry 1: Oxygen 2s
        Entry 2: Oxygen 2p (x)
        Entry 3: Oxygen 2p (y)
        Entry 4: Oxygen 2p (z)
        Entry 5: Hydrogen 1 1s
        Entry 6: Hydorgen 2 1s

    Parameters
    ----------
    C: (n orbitals x n orbitals) array of floats.
        The coefficents for the molecular orbital basis functions.
    r: (1 x 3) array of floats.
        A position in space on the x-y plane. The z-coordinate is required,
        but this should be set to 0.

    Returns
    -------
    MO: (1 x n orbitals) array of floats.
        The value of the molecular orbit function at a point r in the x-y
        plane.
    """

    assert(np.all(np.isfinite(C))), 'The entries of C are not finite.'
    assert(C.shape[0] > 0), 'C has no entries.'
    assert(C.shape[0] == C.shape[1]), 'C is not square.'

    assert(np.all(np.isfinite(r))), 'r is not finite.'
    assert(r.size == 3), 'r has to be a 3D vector.'

    n_orbitals = C.shape[0]

    # define the positions of the oxygen and hydrogen nuceli (in 3D)
    R_O1 = np.array([0.0, +1.809 * np.cos(104.52/180.0 * np.pi/2.0), 0.0])
    R_H1 = np.array([-1.809 * np.sin(104.52/180.0 * np.pi/2.0), 0.0, 0.0])
    R_H2 = np.array([+1.809 * np.sin(104.52/180.0 * np.pi/2.0), 0.0, 0.0])

    # calculate the separations
    rO = r - R_O1  # calculate this once
    rH1 = r - R_H1
    rH2 = r - R_H2
    coord_sep = np.array([rO, rO, rO, rO, rO, rH1, rH2])
    # use linalg.norm to calculate the distance
    R = np.linalg.norm(coord_sep, axis=1)

    # finally calculate the molecular orbits for all the orbitals
    MO = np.zeros(n_orbitals)
    basis = basis_functions(R, C)
    for orb in range(n_orbitals):
        for mu in range(n_orbitals):
            MO[orb] += C[orb, mu] * basis[mu]

    return MO


# =============================================================================
# Simulation Parameters
# =============================================================================

Vnn = 8.90770810  # nucleon repulsion
Nelectrons = 10   # the number of electrons in the orbitals
G = np.fromfile('H2O-two-electron.dat')
G = np.reshape(G, (7, 7, 7, 7))  # two electron integrals

# overlap integrals
S = np.array([[1, 0.2367039, 0,  0, -0, 0.0500137,  0.0500137],
              [0.2367039, 1,  0,  0, -0, 0.4539953,  0.4539953],
              [0,  0,  1,  0,  0, 0,  0],
              [0,  0,  0,  1,  0, 0.2927386, -0.2927386],
              [-0, -0,  0,  0,  1, 0.2455507,  0.2455507],
              [0.0500137, 0.4539953, 0, 0.2927386, 0.2455507, 1, 0.2510021],
              [0.0500137, 0.4539953, 0, -0.2927386, 0.2455507, 0.2510021, 1]])

# one electron integrals
H = np.array([[-3.26850823e+01, -7.60432270e+00, 0.00000000e+00,
               0.00000000e+00, -1.86797000e-02, -1.61960350e+00,
               -1.61960350e+00],
              [-7.60432270e+00, -9.30206280e+00, 0.00000000e+00,
               0.00000000e+00,  -2.22159800e-01, -3.54321070e+00,
               -3.54321070e+00],
              [0.00000000e+00, 0.00000000e+00, -7.43083560e+00,
               0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               0.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               -7.56702220e+00, 0.00000000e+00, -1.89085610e+00,
               1.89085610e+00],
              [-1.86797000e-02, -2.22159800e-01, 0.00000000e+00,
               0.00000000e+00, -7.52665570e+00, -1.65878930e+00,
               -1.65878930e+00],
              [-1.61960350e+00, -3.54321070e+00, 0.00000000e+00,
               -1.89085610e+00, -1.65878930e+00, -4.95649010e+00,
               -1.56026360e+00],
              [-1.61960350e+00, -3.54321070e+00, 0.00000000e+00,
               1.89085610e+00, -1.65878930e+00, -1.56026360e+00,
               -4.95649010e+00]])

# =============================================================================
# Calculate the Energy and Molecular Orbits
# =============================================================================
if __name__ == '__main__':
    pytest.main(['-v'])

    # calculate the energy of the system and MO coefficents
    E, C, OE = HF_iteration(S, H, G, np.zeros_like(H), Vnn, Nelectrons,
                            tol=1e-10)
    n_orbitals = C.shape[0]

    # create x-y plane
    n_points = 300
    x = np.linspace(-4, 4, n_points)
    y = np.linspace(-4, 4, n_points)
    X, Y = np.meshgrid(x, y)

    # iterate through each point in the plane and calculate the MO
    MO = np.zeros((n_orbitals, n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            r = np.array([X[i, j], Y[i, j], 0])
            MO[:, i, j] = molecular_orbits(C, r)

# =============================================================================
# Plotting the Molecular Orbits
# =============================================================================

    # define the nuclei positions for plotting and the subplot titles
    R_O1 = np.array([0.0, +1.809 * np.cos(104.52/180.0 * np.pi/2.0), 0.0])
    R_H1 = np.array([-1.809 * np.sin(104.52/180.0 * np.pi/2.0), 0.0, 0.0])
    R_H2 = np.array([+1.809 * np.sin(104.52/180.0 * np.pi/2.0), 0.0, 0.0])
    subplot_titles = ['Oxygen 1s E = {:1.2f}'.format(OE[0]),
                      'Oxygen 2s E = {:1.2f}'.format(OE[1]),
                      'Oxygen 2p (x) E = {:1.2f}'.format(OE[2]),
                      'Oxygen 2p (y) E = {:1.2f}'.format(OE[3]),
                      'Oxygen 2p (z) E = {:1.2f}'.format(OE[4]),
                      'Hydrogen 1 1s E = {:1.2f}'.format(OE[5]),
                      'Hydrogen 2 1s E = {:1.2f}'.format(OE[6])]

    # plot the orbitals using a contour plot
    fig, ax = plt.subplots(2, 4, figsize=(30, 15))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = ax.ravel()

    for i in range(n_orbitals):
        ax[i].plot(R_O1[0], R_O1[1], 'bx', label='Oxygen atom')
        ax[i].plot(R_H1[0], R_H1[1], 'rx', label='Hydrogen atom')
        ax[i].plot(R_H2[0], R_H1[1], 'rx')
        CS = ax[i].contour(X, Y, MO[i, :, :])
        plt.clabel(CS, ax=ax[i], fontsize=5)
        ax[i].set_xlabel(r'$x$')
        ax[i].set_ylabel(r'$y$')
        ax[i].legend(loc='lower right', prop={'size': 10})
        ax[i].set_title(subplot_titles[i], fontsize=12)

    plt.savefig('orbitals_contour.pdf')
    plt.show()

    # plot the orbitals using a colourmap
    fig, ax = plt.subplots(2, 4, figsize=(30, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = ax.ravel()

    for i in range(n_orbitals):
        im = ax[i].imshow(MO[i, :, :], cmap=plt.cm.plasma, aspect='auto',
                          origin='lower')
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes('right', size='5%', pad=0.5)
        fig.colorbar(im, cax=cax)
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('y')
        ax[i].axes.xaxis.set_ticklabels([])
        ax[i].axes.yaxis.set_ticklabels([])
        ax[i].set_title(subplot_titles[i], fontsize=12)

    plt.savefig('orbitals_colourmap.pdf')
    plt.show()
