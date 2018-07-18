import pytest
import numpy as np
from hartree_fock import transformation_matrix, density_matrix, flock_matrix, \
    orbital_values, HF_step, total_energy, matrix_diff, HF_iteration, \
    basis, basis_functions, molecular_orbits


def test_transformation_matrix(msg=False):
    """
    Testing the transformation matrix function.

    Parameters
    ----------
    msg: bool.
        If msg is True, a message will be printed to the screen if the test
        has passed. Otherwise, if it is False, no message will be printed.
    """

    # expect an error if the S array is not square or has no elements
    S = np.zeros((1, 4))
    S1 = np.array([])  # empty array
    with pytest.raises(AssertionError):
        transformation_matrix(S)
        transformation_matrix(S1)

    # the output should be the same shape as the input
    S = np.ones((7, 7))
    X = transformation_matrix(S)
    assert(X.shape == S.shape)

    if msg is True:
        print('transformation_matrix testing passed.')


def test_density_matrix(msg=False):
    """
    Testing the transformation matrix function.

    Parameters
    ----------
    msg: bool.
        If msg is True, a message will be printed to the screen if the test
        has passed. Otherwise, if it is False, no message will be printed.
    """

    n_electrons = 10
    C = np.zeros((7, 7))
    C1 = np.zeros((1, 7))
    C2 = np.array([])
    with pytest.raises(AssertionError):
        density_matrix(C, 10.5)          # n_electrons is not an int
        density_matrix(C1, n_electrons)  # array not square
        density_matrix(C2, n_electrons)  # array has no elements

    # test for bad input for n_electrons returns an error
    with pytest.raises(ValueError):
        density_matrix(C, 0)    # has to be some electrons
        density_matrix(C, -1)   # negative electrons makes no sense

    # the output has to be the same shape as the array input
    D = density_matrix(C, n_electrons)
    assert(D.shape == C.shape)

    if msg is True:
        print('density_matrix testing passed.')


def test_flock_matrix(msg=False):
    """
    Testing the transformation matrix function.

    Parameters
    ----------
    msg: bool.
        If msg is True, a message will be printed to the screen if the test
        has passed. Otherwise, if it is False, no message will be printed.
    """

    H = np.zeros((7, 7))
    G = np.zeros((7, 7, 7, 7))
    D = np.zeros_like(H)
    H1 = np.zeros((1, 7))
    G1 = np.zeros_like(H1)
    D1 = np.zeros_like(H1)

    with pytest.raises(AssertionError):
        flock_matrix(H, G, D1)  # the arrays are different sizes
        flock_matrix(H1, G, D)  # arrays are not the correct shape
        flock_matrix(H, G1, D)

    # the output should be the same size as H and D
    F = flock_matrix(H, G, D)
    assert(H.shape == F.shape)

    if msg is True:
        print('flock_matrix testing passed.')


def test_orbital_values(msg=False):
    """
    Testing the transformation matrix function.

    Parameters
    ----------
    msg: bool.
        If msg is True, a message will be printed to the screen if the test
        has passed. Otherwise, if it is False, no message will be printed.
    """

    F = np.zeros((7, 7))
    X = np.zeros_like(F)
    F1 = np.zeros((1, 7))
    X1 = np.zeros_like(F1)
    with pytest.raises(AssertionError):
        orbital_values(F1, X)  # test that an error is returned for bad shapes
        orbital_values(F, X1)

    # OE should be (1 x n orbitals (7)) and C should be the same shape as input
    OE, C = orbital_values(F, X)
    assert(OE.shape[0] == 7)
    assert(C.shape == F.shape)

    # check to make sure the orbital engeries are ordered correctly. Use random
    # values to test this
    F = np.random.rand(7, 7)
    X = np.random.rand(7, 7)
    OE, C = orbital_values(F, X)
    for i in range(6):
        assert(OE[i] < OE[i+1])

    if msg is True:
        print('orbital_values testing passed.')


def test_HF_step(msg=False):
    """
    Testing the transformation matrix function.

    Parameters
    ----------
    msg: bool.
        If msg is True, a message will be printed to the screen if the test
        has passed. Otherwise, if it is False, no message will be printed.
    """

    X = np.zeros((7, 7))
    H = np.zeros_like(X)
    G = np.zeros((7, 7, 7, 7))
    C = np.zeros_like(X)
    D = np.zeros_like(X)
    n_electrons = 11

    # some of the output should be the same size as the input arrays and one
    # should be a 1D vector
    nD, OE, nC = HF_step(X, H, G, C, D, n_electrons)
    assert(nD.shape == D.shape)
    assert(nC.shape == C.shape)
    assert(OE.shape[0] == 7)


def test_total_energy(msg=False):
    """
    Testing the transformation matrix function.

    Parameters
    ----------
    msg: bool.
        If msg is True, a message will be printed to the screen if the test
        has passed. Otherwise, if it is False, no message will be printed.
    """

    D = np.random.rand(7, 7)
    H = np.random.rand(7, 7)
    F = np.random.rand(7, 7)
    Vnn = 10

    # the output of this will be a float (or an int)
    E = total_energy(D, H, F, Vnn)
    assert(type(E) is int or float)

    # if Vnn is non-zero, the total energy should be Vnn when all the other
    # inputs are zero
    D = np.zeros((7, 7))
    H = np.zeros_like(D)
    F = np.zeros_like(D)
    E = total_energy(D, H, F, Vnn)
    assert(E == Vnn)

    if msg is True:
        print('total_energy testing passed.')


def test_matrix_diff(msg=False):
    """
    Testing the matrix difference function.

    Parameters
    ----------
    msg: bool.
        If msg is True, a message will be printed to the screen if the test
        has passed. Otherwise, if it is False, no message will be printed.
    """

    D = np.zeros((7, 7))
    nD = np.ones((7, 7))
    nD1 = np.ones((1, 7))

    # an error should be returned if the arrays are different shapes
    with pytest.raises(AssertionError):
        matrix_diff(D, nD1)

    # using the new D as an array of 1s and the old D as zeros, should expect
    # for 7 to be returned
    diff = matrix_diff(D, nD)
    assert(type(diff) is float or int)  # it has to be a float or int!
    assert(diff == 7)

    if msg is True:
        print('matrix diff testing passed.')


def test_HF_iteration(msg=False):
    """
    Testing the transformation matrix function.

    Parameters
    ----------
    msg: bool.
        If msg is True, a message will be printed to the screen if the test
        has passed. Otherwise, if it is False, no message will be printed.
    """

    # use a system which we know the value of and will converge
    n_electrons = 2
    S = np.array([[1.0, 0.434311], [0.434311, 1.0]])
    H = np.array([[-1.559058, -1.111004], [-1.111004, -2.49499]])
    G = np.array([[[[0.77460594, 0.27894304], [0.27894304, 0.52338927]],
                 [[0.27894304, 0.14063907], [0.14063907, 0.34321967]]],
                 [[[0.27894304, 0.14063907], [0.14063907, 0.34321967]],
                  [[0.52338927, 0.34321967], [0.34321967, 1.05571294]]]])
    C = np.zeros_like(S)
    Vnn = 1.3668670357

    # given a larger tolerence, it should converge
    E, C, OE = HF_iteration(S, H, G, np.zeros_like(S), Vnn, n_electrons)
    assert(np.isclose(E, -2.626133045))

    # C should be the same shape as S and OE should be 1 x n orbitals
    assert(C.shape == S.shape)
    assert(OE.shape[0] == 2)

    # use a very small tolerance on a random system where it should reach the
    # max number of iterations. If the max number of iterations are reached,
    # the function will stop and return None
#    S = np.random.rand(2, 2)
#    H = np.random.rand(2, 2)
#    G = np.random.rand(2, 2, 2, 2)
#    C = np.zeros_like(S)
#    E, C, OE = HF_iteration(S, H, G, C, Vnn, n_electrons, 10e-40)
#    assert(E is None)

    if msg is True:
        print('HF_iteration testing passed.')


def test_basis(msg=False):
    """
    Testing the basis function.

    Parameters
    ----------
    msg: bool.
        If msg is True, a message will be printed to the screen if the test has
        passed. Otherwise, it if is False, no message will be printed.
    """

    # the input will be some floats and arrays, but the output should be a
    # float. Check to make sure that is the case

    a = np.ones(3)
    c = np.ones(3)
    R = 1
    A = np.ones(4)  # incorrect formats
    C = np.ones(6)
    r = np.ones(2)

    # should be expecting errors due to incorrect format of the input
    with pytest.raises(AssertionError):
        basis(R, A, c)
        basis(R, a, C)
        basis(r, a, c)

    # the output should be a float or an int
    chi = basis(R, a, c)
    assert(type(chi) is float or int)
    # when R, a and c are all ones, chi is just the constants
    chi_calc = 3 * (1 * (((2 * 1)/np.pi) ** (3/4)) * np.exp(-(1) * 1 ** 2))
    assert(np.isclose(chi, chi_calc))

    if msg is True:
        print('basis testing passed.')


def test_basis_functions(msg=False):
    """
    Testing the transformation matrix function.

    Parameters
    ----------
    msg: bool.
        If msg is True, a message will be printed to the screen if the test
        has passed. Otherwise, if it is False, no message will be printed.
    """

    C = np.zeros((7, 7))
    R = np.zeros(7)
    R1 = np.zeros_like(C)

    # R1 is in the incorrect format so an error should be returned of this
    # is used
    with pytest.raises(AssertionError):
        basis_functions(R1, C)

    # the output should be a (1 x n orbitals) length array
    chis = basis_functions(R, C)
    assert(chis.size == C.shape[0])

    if msg is True:
        print('basis_functions testing passed.')


def test_molecular_orbits(msg=False):
    """
    Testing the transformation matrix function.

    Parameters
    ----------
    msg: bool.
        If msg is True, a message will be printed to the screen if the test
        has passed. Otherwise, if it is False, no message will be printed.
    """

    r = np.ones(3)
    C = np.zeros((7, 7))
    R = np.ones(2)

    # whilst it makes sense to give a 2D vector for R, a 3D vector is required
    # thus an error should be returned when a 2E vector is given
    with pytest.raises(AssertionError):
        molecular_orbits(C, R)

    # the output should be a (1 x n orbitals) array
    MO = molecular_orbits(C, r)
    assert(MO.size == C.shape[0])

    if msg is True:
        print('molecular_orbits testing passed.')
