import pytest
import big_g
import sympy
import numpy as np


def test_shape_function(msg=False):
    """
    Test the function which generates the shape functions for a reference
    triangle.

    Parameters
    ----------
    msg: boolean. If msg is set to True, a message showing that this test
        has passed will be printed to the console.
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

    # test to make sure that an error is returned when bad input is given.
    # If xi is given 1, or 3 or etc values then a Value Error should be raised.
    with pytest.raises(AssertionError):
        big_g.shape_function([1/6, 1/6, 1/6])
        big_g.shape_function([1/6])

    if msg is True:
        print('shape_function testing passed.')


def test_shape_function_dN(msg=False):
    """
    Test the function which returns the derivatives of the shape function
    returns what it should be for linear shape functions.

    Parameters
    ----------
    msg: boolean. If msg is set to True, a message showing that this test
        has passed will be printed to the console.
    """

    # as the shape functions are linear, the derivatives of them will either be
    # 0, 1 or -1 depending on the differentiation variable. Hence, the value of
    # dN will always be the same. This test function tests that the function
    # is putting out the correct values

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

    Parameters
    ----------
    msg: boolean. If msg is set to True, a message showing that this test
        has passed will be printed to the console.
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

    # do the same again for another set of nodes, to make sure it wasn't just
    # a fluke
    nodes = np.array([[1, 0], [1, 1], [0, 1]])
    N_ans = np.array([[1 - 4/6 - 1/6], [1/6], [4/6]])
    X_ans = np.array([nodes[0, 0] * N_ans[0] + nodes[1, 0] * N_ans[1] +
                      nodes[2, 0] * N_ans[2],
                      nodes[0, 1] * N_ans[0] + nodes[1, 1] * N_ans[1] +
                      nodes[2, 1] * N_ans[2]])

    X_ans = np.reshape(X_ans, (1, 2))
    X_func = big_g.local_to_global(nodes, xi)

    assert(np.all(X_func == X_ans)), \
        'local_to_global is calculating incorrect global coordinates for the \
            element node.'

    # give the function some bad input to see if it returns errors as it
    # should be doing

    # test for bad xi's
    with pytest.raises(AssertionError):
        big_g.local_to_global(nodes, [1/6])
        big_g.local_to_global(nodes, [1, 1, 1])

    # test for bad nodes
    nodes_bad = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    with pytest.raises(AssertionError):
        big_g.local_to_global(nodes_bad, xi)

    if msg is True:
        print('local_to_global function testing passed.')


def test_jacobian(msg=False):
    """
    Test the function which generates the Jacobian matrix using the local
    reference coordinates (xi, eta) and global node locations nodes.

    Parameters
    ----------
    msg: boolean. If msg is set to True, a message showing that this test
        has passed will be printed to the console.
    """

    # calculate the Jacobian matrix by hand and compare what this
    # calculates to what the function calculates. Expecting the function
    # to compute a 2x2 array.

    dN = np.array([[-1, 1, 0], [-1, 0, 1]])
    nodes1 = np.array([[0, 0], [1, 0], [0, 1]])
    nodes2 = np.array([[1, 0], [1, 1], [0, 1]])

    # calculate the Jacobian using matrix multiplication of the derivs at
    # the node locations
    J = np.zeros((2, 2))
    for i in range(len(dN)):
        for j in range(len(nodes1[0])):
            for k in range(len(nodes1)):
                J[i][j] += dN[i][k] * nodes1[k][j]

    # calculate the jacobian using the function and compare both
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

    # test that the function raises an error if given bad input, the function
    # is expecting an array of shape (3, 2) for nodes.
    with pytest.raises(AssertionError):
        big_g.jacobian(np.array([[0, 0], [1, 0]]))

    if msg is True:
        print('jacobian function testing passed.')


def test_det_jacobian(msg=False):
    """
    Tests the det_jacobian function which uses the numpy function to calculate
    the determinant of a matrix.

    Parameters
    ----------
    msg: boolean. If msg is set to True, a message showing that this test
        has passed will be printed to the console.
    """

    # as the function uses the numpy linear algebra function to calculate the
    # matrix deteriminant, there is unlikely to be an error. But, incase a bug
    # has been added to this function, I will calculate the deteriminant of a
    # jacobian by hand and compare the two

    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    # define the Jacobian for the nodes above
    J = np.array([[1, 0], [0, 1]])
    # the function calculates the absolute value of the det, so I will so that
    # here as well
    det_J = np.abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
    # calculate the det using the function
    det_func = big_g.det_jacobian(nodes)

    assert(det_J == det_func), \
        'The determinant of the Jacobian is being calculated incorrectly.'

    if msg is True:
        print('det_jacobian function testing passed.')


def test_global_dN(msg=False):
    """
    Tests the global_dN function which calculates the derivatives at a global
    location.

    Parameters
    ----------
    msg: boolean. If msg is set to True, a message showing that this test
        has passed will be printed to the console.
    """

    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    dN = np.array([[-1, 1, 0], [-1, 0, 1]])
    dN_global = np.zeros_like(dN)

    # I shall calculate the value of the global derivatives at a set of
    # nodes by hand, apart from when solving the linear equation where I
    # will use the numpy.lingalg.solve to make my life easier.

    # first I calculate the Jacobian for the nodes provided
    J = np.zeros((2, 2))
    for i in range(len(dN)):
        for j in range(len(nodes[0])):
            for k in range(len(nodes)):
                J[i][j] += dN[i][k] * nodes[k][j]

    # solve the linear equation global_dN * J = dN
    for i in range(3):
        dN_global[:, i] = np.linalg.solve(J, dN[:, i])

    # now compare this to what the function returns
    dN_global_func = big_g.global_dN(nodes)

    # I take the transverse of the matrix here so the dimensions match
    assert(np.all(dN_global_func == dN_global.T)), \
        'The global derivatives are being calculated incorrectly.'

    # Now do the same for a different set of nodes
    nodes2 = np.array([[1, 0], [1, 1], [0, 1]])

    J = np.zeros((2, 2))
    for i in range(len(dN)):
        for j in range(len(nodes2[0])):
            for k in range(len(nodes2)):
                J[i][j] += dN[i][k] * nodes2[k][j]

    for i in range(3):
        dN_global[:, i] = np.linalg.solve(J, dN[:, i])

    dN_global_func = big_g.global_dN(nodes2)

    assert(np.all(dN_global_func == dN_global.T)), \
        'The global derivatives are being calculated incorrectly.'

    # check that the appropriate error is given for bad input, i.e.
    # where nodes is not the correct shape
    with pytest.raises(AssertionError):
        big_g.global_dN(np.array([[0, 0], [1, 0]]))

    if msg is True:
        print('global_dN function testing passed.')


def test_quad(msg=False):
    """
    Tests the element_quad and reference_quad function which calculates the
    volume of an element and reference element respectively.

    Parameters
    ----------
    msg: boolean. If msg is set to True, a message showing that this test
        has passed will be printed to the console.
    """

    # this function is the Gaussian quadrature of a function at three
    # points defined by xi1, xi2 and xi3. Therefore, I will use simple
    # functions for f(xi) to test that the function is working as intended

    def fg(xi):
        return 1

    def gg(xi):
        return 5

    def hg(xi):
        return 0

    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    xi1 = [1/6, 1/6]
    xi2 = [4/6, 1/6]
    xi3 = [1/6, 4/6]

    # calculate the gauss quadrature by hand for the functions
    # this should return 1/6 * (1 + 1 + 1) = 0.5
    gauss_quadf = 1/6 * (fg(xi1) + fg(xi2) + fg(xi3))
    # this should return 1/6 * (5 + 5 + 5) = 2.5
    gauss_quadg = 1/6 * (gg(xi1) + gg(xi2) + gg(xi3))
    # this should return 0
    gauss_quadh = 1/6 * (hg(xi1) + hg(xi2) + hg(xi3))

    # redefine the functions for element_quad as it requires different
    # arguments

    def f(nodes, N, dN):
        return 1

    def g(nodes, N, dN):
        return 5

    def h(nodes, N, dN):
        return 0

    # calculate the quadrature using the element_quad function and compare
    quadf = big_g.element_quad(f, nodes)
    quadg = big_g.element_quad(g, nodes)
    quadh = big_g.element_quad(h, nodes)

    assert(quadf == gauss_quadf), \
        'Element quadrature has been calculated incorrectly.'
    assert(quadg == gauss_quadg), \
        'Element quadrature has been calculated incorrectly.'
    assert(quadh == gauss_quadh), \
        'Element quadrature has been calculated incorrectly.'

    # define some more functions which will be used with element_quad. The
    # output will be compared to the output from sympy.integrate

    def psi1(nodes, N, dN):
        return nodes[0]

    def psi2(nodes, N, dN):
        return 1-nodes[0]-nodes[1]

    # calculate the volume of the element using integration, rather than Gauss
    # quadrature
    # set the element length to be 1
    length = 1
    spx, spy = sympy.symbols('x, y')
    # integrate above functions explicitly by using sympy.integrate
    x_int = sympy.integrate(sympy.integrate(
            spx, (spx, 0, length - spy)), (spy, 0, length))
    xy_int = sympy.integrate(sympy.integrate(
            1 - spx - spy, (spx, 0, length - spy)), (spy, 0, length))

    # now calculate the same thing now using Gauss quadrature
    x_quad = big_g.element_quad(psi1, nodes)
    xy_quad = big_g.element_quad(psi2, nodes)

    # use np.isclose as comparing floating points from different functions
    assert(np.isclose(float(x_int), x_quad)), \
        'Element volume is being calculated incorrectly.'
    assert(np.isclose(float(xy_int), xy_quad)), \
        'Element volume is being calculated incorrectly.'

    if msg is True:
        print('quadrature functions testing passed.')
