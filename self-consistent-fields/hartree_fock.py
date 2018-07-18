import numpy as np

# =============================================================================
# Hartree-Fock Algorithm Functions
# =============================================================================

def transformation_matrix(S):
    """
    Compute the transformation matrix given the matrix S.

    Parameters
    ----------
    S: some description of the variable type.
        Some description of the variable.

    Returns
    -------
    X: some description of the variable type.
        Some description of the variable.
    """

    eigvals, eigvecs = np.linalg.eig(S)
    X = np.dot(eigvecs, np.dot(np.diag(eigvals ** (-0.5)),
                               np.conj(np.transpose(eigvecs))))

    return X


def density_matrix(C, n_electrons):
    """
    Compute the density matrix given the matrix C.

    Parameters
    ----------
    C: some description of the variable type.
        Some description of the variable.
    n_electrons: some description of the variable type.
        Some description of the variable.

    Returns
    -------
    D: some description of the variable type.
        Some description of the variable.
    """

    D = np.zeros_like(C)
    n = C.shape[0]

    for mu in range(n):
        for nu in range(n):
            for j in range(n_electrons//2):
                D[mu, nu] += 2 * C[mu, j] * C[nu, j]

    return D


def flock_matrix(H, G, D):
    """
    Compute the Flock matrix given the matrix H, G and D.

    Parameters
    ----------
    H: some description of the variable type.
        Some description of the variable.
    G: some description of the variable type.
        Some description of the variable.
    D: some description of the variable type.
        Some description of the variable.

    Returns
    -------
    F: some description of the variable type.
        Some description of the variable.
    """

    F = np.zeros_like(H)
    n = F.shape[0]

    F[:, :] = H[:, :]

    for mu in range(n):
        for nu in range(n):
            for alpha in range(n):
                for beta in range(n):
                    F[mu, nu] += (G[mu, nu, alpha, beta] - 0.5 *
                                  G[mu, beta, alpha, nu]) * D[alpha, beta]

    return F


def orbital_values(F, X):
    """
    Compute the eigenvalues and eigenvectors of the Flock matrix.

    Parameters
    ----------
    F: desc
        more desc
    X: desc
        more desc

    Returns
    -------
    eig vals: desc
        desc
    orb_coef: desc
        desc
    """

    Fprime = np.dot(np.conj(np.transpose(X)), np.dot(F, X))
    e, lamda = np.linalg.eigh(Fprime)

    # putting the eigenvalues and eigenvectors into the correct order
    idx = e.argsort()
    orb_eng = e[idx]
    orb_coefs = lamda[:, idx]

    C = np.dot(X, orb_coefs)

    return orb_eng, C


def HF_step(X, H, G, C, D, n_electrons):
    """
    Compute the new matrices.
    """

    F = flock_matrix(H, G, D)
    orb_eng, C = orbital_values(F, X)
    new_D = density_matrix(C, n_electrons)

    return new_D, orb_eng, C


def total_energy(D, H, F, Vnn):
    """
    Compute the total energy of the system.
    """

    n = D.shape[0]
    Etot = Vnn

    for mu in range(n):
        for nu in range(n):
            Etot += 0.5 * D[mu, nu] * (H[mu, nu] + F[mu, nu])

    return Etot


def HF_iteration(S, H, G, C, Vnn, n_electrons, tol=1e-6):
    """
    Compute the HF interations.
    """

    def matrix_diff(D, DNew):
        """
        Calculate the difference between the old and updated density matrix.
        """

        difference = 0
        n = D.shape[0]

        for mu in range(n):
            for nu in range(n):
                difference += (D[mu, nu] - DNew[mu, nu]) ** 2

        difference = np.sqrt(difference)

        return difference

    X = transformation_matrix(S)
    D = density_matrix(C, n_electrons)

    diff = 10 * tol
    current_iteration = 0
    max_iteration = 100

    while (diff > tol) and (current_iteration < max_iteration):
        current_iteration += 1
        DNew, C, orb_engs = HF_step(X, H, G, C, D, n_electrons)
        diff = matrix_diff(D, DNew)
        D = DNew

    if (diff < tol):
        F = flock_matrix(H, G, D)
        energy = total_energy(D, H, F, Vnn)
        print('Total energy: {}\nIterations: {}'
              .format(energy, current_iteration))

        return energy, C

    else:
        print('Ruh-roh, failed to converge!')

        return -1, -1

# =============================================================================
# Test Data
# =============================================================================

n_electrons = 2
S = np.array([[1.0, 0.434311], [0.434311, 1.0]])
H = np.array([[-1.559058, -1.111004], [-1.111004, -2.49499]])
G = np.array([[[[ 0.77460594,  0.27894304],[ 0.27894304,  0.52338927]],
             [[ 0.27894304,  0.14063907],[ 0.14063907,  0.34321967]]],
             [[[ 0.27894304,  0.14063907],[ 0.14063907,  0.34321967]],
             [[ 0.52338927,  0.34321967],[ 0.34321967,  1.05571294]]]])
Vnn = 1.3668670357

Etot, C = HF_iteration(S, H, G, np.zeros_like(H), Vnn, n_electrons, tol=1e-10)