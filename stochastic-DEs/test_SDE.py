import pytest
import numpy as np
from ornstein_SDE import EM, TM


def test_EM(msg=False):
    """
    Testing the EM function.
    """

    # create some placeholder data
    dW = np.zeros(10)
    dt = 0.1
    X0 = 1
    fn = lambda x: 0
    gn = lambda x: 0

    with pytest.raises(AssertionError):
        # the EM function is more robust than the TM function as it can take in
        # functions for the deterministic and stochastic part of the equation. 
        # Thus test to make sure the function checks the fn and gn arguments are 
        # functions
        EM(dW, dt, X0, 0, gn)
        EM(dW, dt, X0, fn, 0)

        # test that the function returns errors when bad input is given
        EM([1], dt, X0, fn, gn)  # not enough wiener increments
    
    # test that an error is returned when bad input for h is given
    with pytest.raises(ValueError):
        EM(dW, 0, X0, fn, gn)
        EM(dW, -1, X0, fn, gn)

    # the first element of the solution returned should be X0 otherwise the 
    # initial value has been lost.
    X = EM(dW, dt, X0, fn, gn)
    assert(X[0] == X0)

    # feed in a simple case where gn returns zero, i.e. there is no 
    # stochastic noise. This results in the EM method reducing to the Euler
    # method of solving an initial value problem thus we can test the output
    X = EM(dW, 0.5, 1, lambda x: x, lambda x: 0)
    # sequence should go 1, 1.5, 2.25, .., 25.62, 38.44, 57.66.
    assert(X[0] == 1)
    assert(X[1] == 1.5)
    assert(np.isclose(X[-1], 57.6650391))
    assert(np.isclose(X[-2], 38.4433594))
    # X should be of size N+1
    assert(X.size == len(dW)+1)

    # same as above, but now gn is 1 and the dW increments will be 1 also, thus
    # the stochastic part will be adding an extra value
    dW = np.ones(10)
    X = EM(dW, 0.5, 1, lambda x: x, lambda x: 1)
    # sequence should go 1, 2.5, 4.75, ..., 74.88, 113.33, 170.00
    assert(X[0] == 1)
    assert(X[1] == 2.5)
    assert(np.isclose(X[-1], 170.9951172))
    assert(np.isclose(X[-2], 113.3300781))
    # X should be of size N+1
    assert(X.size == len(dW)+1)

    if msg is True:
        print('EM testing passed!')


def test_TM(msg=False):
    """
    Test the TM fuction.
    """

    # create some placeholder data
    dW = np.zeros(10)
    dt = 0.1
    X0 = 1
    lamda = 1
    mu = 1
    theta = 0.5

    # check that the function returns an error when bad input is given for dW
    with pytest.raises(AssertionError):
        TM([1], dt, X0, theta, mu, lamda)

    # check that the function returns an error when bad values of h are used
    with pytest.raises(ValueError):
        TM(dW, 0, X0, theta, mu, lamda)
        TM(dW, -1, X0, theta, mu, lamda)

    # the first element of the solution returned should be X0
    X = TM(dW, dt, X0, theta, mu, lamda)
    assert(X[0] == X0)
    # X should be of size N+1
    assert(X.size == len(dW)+1)

    # test TM for some simple parameters, lamda = 1, mu = 0, theta = 0.5. The
    # sequence should go, 1, 0.6, 0.36, ..., 0.016, 0.01, 0.00604
    X = TM(dW, 0.5, 1, 0.5, 0, 1)
    assert(X[0] == 1)
    assert(X[1] == 0.6)
    assert(np.isclose(X[-1], 0.006046618))
    assert(np.isclose(X[-2], 0.010077696))

    # test TM now for when now mu = 1, dW = 1, the sequence should go,
    # 1, 1.4, 1.64, ..., 1.983, 1.989, 1.99
    dW = np.ones(10)
    X = TM(dW, 0.5, 1, 0.5, 1, 1)
    assert(X[0] == 1)
    assert(X[1] == 1.4)
    assert(np.isclose(X[-1], 1.993953382))
    assert(np.isclose(X[-2], 1.989922304))

    if msg is True:
        print('TM testing passed!')