import numpy as np
from matplotlib import pyplot as plt


def EM(dW, h, X0, f, g):
    """
    Compute the approximate solution X(t) of an SDE using the Euler-Maruyama
    method.

    Parameters
    ----------
    dW: 1 x t array of floats.
        The value of the random noise at each point t; this is not a cum sum
        unlike W.
    h: float.
        The grid spacing for t.
    X0: float.
        The initial value of the solution, given by X(0).
    f: function.
        A function to define the X(t) without random noise.
    g: function.
        A function to define X(t) with random noise.

    Returns
    -------
    X_EM: 1 x t+1 array of floats.
        The approximate value of X(t) at each point t as given by the Euler-
        Maryuama method.
    """

    def EM_step(X, dW):
        """
        The stepping alogorithm for the Euler-Maruyama method. Takes in a value
        X(t) and returns X(t+1).

        Parameters
        ----------
        X: float.
            The value of X(t) at any given point.
        dW: float.
            The random noise dW at the point X(t).

        Returns
        -------
        Xn: float.
            The value of X(t+1).
        """

        fn = f(X)
        gn = g(X)

        Xn = X + fn * h + gn * dW

        return Xn

    N = len(dW)
    X_EM = np.zeros(N+1)
    X_EM[0] = X0

    for i in range(N):
        X_EM[i+1] = EM_step(X_EM[i], dW[i])

    return X_EM


def OU_Expectation(X0, mu, lamda, t):
    """
    Computes the expectation and the square of the expectation value of
    the Ornstein-Uhlenbeck problem.

    Parameters
    ----------
    X0: float.
        The initial value of the solution, given by X(0).
    mu: float.
        The scaling parameter for the random part of the O-U equation.
    lamda: float.
        The scaling parameter for the deterministic part of the O-U
        equation.
    t: 1 x t array of floats.
        The values of t to evaulate the O-U equation at.

    Returns
    -------
    Ex: 1 x t array of floats.
        The expectation value of the O-U equation at each value of t.
    ExSq: 1 x t array of floats.
        The square of the expectation value of the O-U equation at each
        value of t.
    """

    Ex = np.exp(-lamda * t) * X0
    ExSq = np.exp(-2 * lamda * t) * X0 ** 2 + (mu ** 2)/(2 * lamda) * \
        (1 - np.exp(-2 * lamda * t))

    return Ex, ExSq


def solve(t, dt, N_sims, N, M, multi_realisations):
    """
    Solve the Ornstein-Uhlenbeck problem.
    
    Parameters
    ----------
    t: 1 x t array of floats.
        The values of t to evaulate the O-U equation at.
    dt: float.
        The grid spacing for t.
    N_sims: int.
        The number of simulations to run.
    N: int.
        The number of random walk paths.
    M: int.
        The number of realisiations.
    multi_realisations: bool.
        Use True to use take the average of multiple realisations and recieve
        a more accurate result. Using False will take an average of the number
        of simulations to provide a visual of the average path being taken. 
    
    Returns
    -------
    X: 1 x t array of floats.
        The approximate value of X(t) at each point t as given by the Euler-
        Maryuama method.
    Ex: 1 x t array of floats.
        The expectation value of the O-U equation at each value of t.
    ExSq: 1 x t array of floats.
        The square of the expectation value of the O-U equation at each
        value of t.
    """
    
    assert(type(multi_realisations) == bool), \
        'multi_realisations has to be a bool'
    
    plt.figure(figsize=(18, 9))  
    if multi_realisations is False:
        # less accurate, but much faster
        X = np.zeros((N_sims, N+1))
        for n in range(N_sims):
            dW = np.sqrt(dt) * np.random.randn(N)
            X[n, :] = EM(dW, dt, X0, fn, gn)
            
            plt.plot(t, X[n, :], '--', label='Simulation {}'.format(n+1), 
                     linewidth=1)
            
        plt.plot(t, np.mean(X, axis=0), '-', label='Average Value', 
                 linewidth=2)
            
    elif multi_realisations is True:
        # more accurate, but slower    
        X = np.zeros((N_sims, N+1, M))
        for n in range(N_sims):
            dW = np.sqrt(dt) * np.random.randn(N, M)
            for m in range(M):
                X[n, :, m] = EM(dW[:, m], dt, X0, fn, gn)
                
            plt.plot(t, np.mean(X[n, :, :], axis=1), '--',
                     label='Simulation {}'.format(n+1), linewidth=1)       
    
    else:
        print("Incorrect value for multi_realisations variable.")
                
    EX, EXSq = OU_Expectation(X0, mu, lamda, t)
    plt.plot(t, EX, '-', label='Expectation', linewidth=3)    
    plt.xlim(t[0], t[-1])
    plt.xlabel('t')
    plt.ylabel('X(t)')    
    plt.legend(loc='upper right')
    plt.title('multi_realisations = {}'.format(multi_realisations))
    plt.savefig('Ornstein-Uhlenbeck_Solutions_multi_realisations={}.pdf'
                .format(multi_realisations))
    plt.show()

    return X, EX, EXSq


# =============================================================================
# Simulation~~
# =============================================================================

# define functions for Euler-Maruyama method
def fn(x):
    lamda = 1
    return - lamda * x


def gn(x):
    mu = 1
    return mu


# define simulation parameters
X0 = 1
lamda = 1
mu = 1
t_start = 0
t_end = 1
N = 2 ** 10
M = 10 ** 3
N_sims = 5
t, dt = np.linspace(t_start, t_end, N+1, retstep=True)

X, EX, EXSq = solve(t, dt, N_sims, N, M, False)
X, EX, EXSq = solve(t, dt, N_sims, N, M, True)
