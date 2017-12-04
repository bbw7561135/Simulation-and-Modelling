import pytest
import big_g
import numpy as np


def test_shape_function(msg=False):
    """
    Test the function which generates the shape functions for a reference
    triangle.
    """

    # test that given some values of the reference coordinates, the function
    # is returning the expected results with given values of xi
    N = big_g.shape_function([1/6, 4/6])
    N_ans = np.array([[1 - 4/6 - 1/6], [1/6], [4/6]])
    assert(np.all(N == N_ans)), \
        'The calculated shape functions are incorrect.'
    N = big_g.shape_function([1/6, 1/6])
    N_ans = np.array([[1 - 1/6 - 1/6], [1/6], [1/6]])
    assert(np.all(N == N_ans)), \
        'The calculated shape functions are incorrect.'

    # test to make sure that an error is returned when garbage input is given.
    # If xi is given 1, or 3 or etc then a Value Error should be raised.
    with pytest.raises(ValueError):
        big_g.shape_function([1/6, 1/6, 1/6])
        big_g.shape_function([1/6])

    if msg is True:
        print('shape_function testing passed.')


def test_shape_function_dN(msg=False):
    """
    Test the function which returns the derivatives of the shape function
    returns what it should be for linear shape functions.
    """

    # as the shape functions are linear, the derivatives of them will either be
    # 0, 1 or -1 depending on the differentiation variable. Hence, the value of
    # dN will always the same same. This test function test that the function
    # is putting out the correct array

    dN = big_g.shape_function_dN()
    dN_ans = np.array([[-1, -1], [1, 0], [0, 1]])
    assert(np.all(dN == dN_ans)), \
        'shape_function_dN is returning incorrect derivatives for the shape\
            functions.'

    if msg is True:
        print('shape_function_dN testing passed.')


def test_local_to_global(msg=False):
    """
    Test the function which converts the local triangle reference points into
    the global node reference points.
    """

    # hand calculate all of the inputs for the function and then calculate the
    # expected output. Compare this calculated value to the value which is
    # output by the function.

    xi = [1/6, 4/6]
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    N_ans = np.array([[1 - 4/6 - 1/6], [1/6], [4/6]])
    X_ans = np.array([nodes[0, 0] * N_ans[0] + nodes[1, 0] * N_ans[1] +
                      nodes[2, 0] * N_ans[2],
                      nodes[0, 1] * N_ans[0] + nodes[1, 1] * N_ans[1] +
                      nodes[2, 1] * N_ans[2]])
    # reshape the array to match the array that is output
    X_ans = np.reshape(X_ans, (1, 2))

    X_func = big_g.local_to_global(nodes, xi)

    assert(np.all(X_func == X_ans)), \
        'local_to_global is calculating incorrect global coordinates for the \
            element node.'

    # give the function some trash input to see if it returns errors as it
    # should be doing
    with pytest.raises(ValueError):
        big_g.local_to_global(nodes, [1/6])
        big_g.local_to_global(nodes, [1, 1, 1])

    nodes_bad = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    with pytest.raises(AssertionError):
        big_g.local_to_global(nodes_bad, xi)

    if msg is True:
        print('local_to_global function testing passed.')


def test_jacobian(msg=False):
    """
    Test the function which generates the Jacobian matrix using the local
    reference coordinates (xi, eta) and global node locations nodes.
    """

    # calculate the Jacobian matrix by hand to check that the function is
    # calculating the Jacobian correctly

    dN = np.array([[-1, 1, 0], [-1, 0, 1]])
    nodes1 = np.array([[0, 0], [1, 0], [0, 1]])
    nodes2 = np.array([[1, 0], [1, 1], [0, 1]])

    # calculate the Jacobian using matrix multiplication
    J = np.zeros((2, 2))
    for i in range(len(dN)):
        for j in range(len(nodes1[0])):
            for k in range(len(nodes1)):
                J[i][j] += dN[i][k] * nodes1[k][j]

    J_func = big_g.jacobian(nodes1)

    assert(np.all(J_func == J)), \
        'Jacobian matrix calculated incorrectly in function.'

    # do this again for for nodes2 to make sure it wasn't a fluke
    J = np.zeros((2, 2))
    for i in range(len(dN)):
        for j in range(len(nodes2[0])):
            for k in range(len(nodes2)):
                J[i][j] += dN[i][k] * nodes2[k][j]

    J_func = big_g.jacobian(nodes2)

    assert(np.all(J_func == J)), \
        'Jacobian matrix calculated incorrectly in function.'

    # test that the function raises an error if given wrong input
    with pytest.raises(AssertionError):
        big_g.jacobian(np.array([[0, 0], [1, 0]]))

    if msg is True:
        print('jacobian function testing passed.')


def test_det_jacobian(msg=False):
    """
    Tests the det_jacobian function which uses the numpy function to calculate
    the determinant of a matrix.3
    """

    # test the the function hasn't got a bug and is outputting the determinant
    # of a matrix.

    # calcaulte the determinant of J by hand
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    J = np.array([[1, 0], [0, 1]])
    det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    # call the function to calculate the determinant
    det_func = big_g.det_jacobian(nodes)

    assert(det_J == det_func), \
        'The determinant of the Jacobian is being calculated incorrectly.'

    if msg is True:
        print('det_jacobian function testing passed.')


def test_global_dN(msg=False):
    """
    Tests the global_dN function which calculates the derivatives at a global
    location.
    """

    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    dN = np.array([[-1, 1, 0], [-1, 0, 1]])
    dN_global = np.zeros_like(dN)

    # I shall test this function by hand calculating a global dN using the
    # linear algebra solver to solve the linear equation, but everything else
    # will be done manually.

    # calculate the Jacobian using matrix multiplication
    J = np.zeros((2, 2))
    for i in range(len(dN)):
        for j in range(len(nodes[0])):
            for k in range(len(nodes)):
                J[i][j] += dN[i][k] * nodes[k][j]

    # solving the linear system to find the global derivatives
    for i in range(3):
        dN_global[:, i] = np.linalg.solve(J, dN[:, i])

    # now compare this to what the function spits out
    dN_global_func = big_g.global_dN(nodes)

    assert(np.all(dN_global_func == dN_global.T)), \
        'The global derivatives are being calculated incorrectly.'

    # I'll do the same for a different set of nodes to make sure :-)
    nodes2 = np.array([[1, 0], [1, 1], [0, 1]])

    J = np.zeros((2, 2))
    for i in range(len(dN)):
        for j in range(len(nodes2[0])):
            for k in range(len(nodes2)):
                J[i][j] += dN[i][k] * nodes2[k][j]

    # solving the linear system to find the global derivatives
    for i in range(3):
        dN_global[:, i] = np.linalg.solve(J, dN[:, i])

    dN_global_func = big_g.global_dN(nodes2)

    assert(np.all(dN_global_func == dN_global.T)), \
        'The global derivatives are being calculated incorrectly.'

    # check that the error provides errors explaining bad input
    with pytest.raises(AssertionError):
        big_g.global_dN(np.array([[0, 0], [1, 0]]))


def test_element_refernce_quad(msg=False):
    """
    Tests the element_quad and reference_quad function which calculates the
    volume of an element and reference element respectively.
    """

    # these function are the Gaussian quadrature of a function at three
    # points defined by xi1, xi2 and xi3. Therefore feeding in simple functions
    # for f(xi) to test that the function is working as expected

    def fg(xi):
        return 1

    def gg(xi):
        return 5

    def hg(xi):
        return 0

    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    # the det jacobian is 1 for this set of nodes

    xi1 = [1/6, 1/6]
    xi2 = [4/6, 1/6]
    xi3 = [1/6, 4/6]

    # the following calculations are the exact same the reference_quad function
    # should be calculating

    # this should return 1/6 * (1 + 1 + 1) = 0.5
    gauss_quadf = 1/6 * (fg(xi1) + fg(xi2) + fg(xi3))
    # this should return 1/6 * (5 + 5 + 5) = 2.5
    gauss_quadg = 1/6 * (gg(xi1) + gg(xi2) + gg(xi3))
    # this should return 0
    gauss_quadh = 1/6 * (hg(xi1) + hg(xi2) + hg(xi3))

    # define the functions for element_quad as different arguments are needed
    def f(nodes, N, dN):
        return 1

    def g(nodes, N, dN):
        return 5

    def h(nodes, N, dN):
        return 0

    # calculate the quadrature using the element_quad function
    quadf = big_g.element_quad(f, nodes)
    quadg = big_g.element_quad(g, nodes)
    quadh = big_g.element_quad(h, nodes)

    assert(quadf == gauss_quadf), \
        'Element quadrature has been calculated incorrectly.'
    assert(quadg == gauss_quadg), \
        'Element quadrature has been calculated incorrectly.'
    assert(quadh == gauss_quadh), \
        'Element quadrature has been calculated incorrectly.'

    # define some functions to test against scipy integrate
    def psi1(nodes, N, dN):
        return nodes[0]

    def psi2(nodes, N, dN):
        return 1-nodes[0]-nodes[1]

