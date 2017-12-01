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

    if msg == True:
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

    if msg == True:
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

    if msg == True:
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

    if msg == True:
        print('jacobian function testing passed.')

def test_det_jacobian(msg=False):


