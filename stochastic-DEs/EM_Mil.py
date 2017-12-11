# =============================================================================
# Evaluate the SDE,
#           dX = lambda * X(t) * dt + mu * X(t) * dW(t), or in general,
#           dX = fn(X) * dt + gn(X) * dW(t),
# using the Euler-Maruyama and Millstein methods.
# =============================================================================

import timeit
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12, 6)

# define some global variables for the functions here because I'm lazy
X0 = 1
mu = 1
lamda = 2
t_end = 1
N = 2 ** 10
M = 10000
ratios = 2 ** np.arange(7)

# define a seed for reproducible solutions
np.random.seed(100)


def exact_sol(t, W):
    """
    Compute the exact solution of the SDE.

    Parameters
    ----------
    t: 1 x t array of floats.
        The values of t to evaluate the SDE at.
    W: 1 x t array of floats.
        The cum sum of the random noise added to the SDE at for each point of
        t.

    Returns
    -------
    X: 1 x t array of floats.
        The exact solution X(t) at each point t.
    """

    X = X0 * np.exp((lamda - 0.5 * mu ** 2) * t + mu * W)

    return X


def EM(dW, h, f, g):
    """
    Compute the approximate solution to an SDE using the Euler-Maruyama method.

    Parameters
    ----------
    dW: 1 x t array of floats.
        The value of the random noise at each point t; this is not a cum sum
        unlike W.
    h: float.
        The grid spacing for t.
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
        X_EM: float.
            The value of X(t+1).
        """

        fn = f(X)
        gn = g(X)

        X_EM = X + fn * h + gn * dW

        return X_EM

    N = len(dW)
    X_EM = np.zeros(N+1)
    X_EM[0] = X0

    for i in range(N):
        X_EM[i+1] = EM_step(X_EM[i], dW[i])

    return X_EM


def milstein(dW, dt, f, g, gprime):
    """
    Compute the approximate solution to an SDE using the Euler-Maruyama method.

    Parameters
    ----------
    dW: 1 x t array of floats.
        The value of the random noise at each point t; this is not a cum sum
        unlike W.
    h: float.
        The grid spacing for t.
    f: function.
        A function to define the X(t) without random noise.
    g: function.
        A function to define X(t) with random noise.
    gprime: function.
        A function to define the derivative of X(t) for the random part of the
        SDE.

    Returns
    -------
    X_Mi: 1 x t+1 array of floats.
        The approximate value of X(t) at each point t as given by the Milstein
        method.
    """

    def milstein_step(X, dW):
        """
        The stepping alogorithm for the Milstein method. Takes in a value X(t)
        and returns X(t+1).

        Parameters
        ----------
        X: float.
            The value of X(t) at any given point.
        dW: float.
            The random noise dW at the point X(t).

        Returns
        -------
        X_M: float.
            The value of X(t+1).
        """

        fn = f(X)
        gn = g(X)
        gnprime = gprime(X)

        X_M = X + dt * fn + gn * dW + 0.5 * gn * gnprime * (dW ** 2 - dt)

        return X_M

    N = len(dW)
    X_Mi = np.zeros(N+1)
    X_Mi[0] = X0

    for i in range(N):
        X_Mi[i+1] = milstein_step(X_Mi[i], dW[i])

    return X_Mi


# =============================================================================
# Define the functions f(xn) and g(xn)
# =============================================================================

def f(x):
    return lamda * x


def g(x):
    return mu * x


def gprime(x):
    return mu

# =============================================================================
# Compute the solutions, plot and compare execution run time
# =============================================================================

# set up integration grid, dt is the spacing on the grid
t, dt = np.linspace(0, t_end, N+1, retstep=True)
dW = np.sqrt(dt) * np.random.randn(N)  # create random noise at each point t
W = np.zeros(N+1)
W[1:] = np.cumsum(dW)                  # cum sum of the random noise

X_EM = EM(dW, dt, f, g)
X_M = milstein(dW, dt, f, g, gprime)
X_Exact = exact_sol(t, W)

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)
ax1.plot(t, X_Exact, 'k-', label='Exact Solution')
ax1.plot(t, X_EM, 'b--', label='Euler-Maruyama')
ax1.plot(t, X_M, 'r--', label='Milstein')
ax1.set_xlabel('$t$')
ax1.set_ylabel('$X(t)$')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# =============================================================================
# Execution time comparisons
# =============================================================================

# use timeit to time the function for x amount of times, where x is a number
# decided by timeit's autorange function
time_EM = timeit.Timer(
        'EM(dW, dt, f, g)',
        setup='from __main__ import EM, dW, dt, f, g')
time_M = timeit.Timer(
        'milstein(dW, dt, f, g, gprime)',
        setup='from __main__ import milstein, dW, dt, f, g, gprime')

avg_t_EM = time_EM.autorange()[1]
avg_t_M = time_M.autorange()[1]

print('Average runtime for Euler-Maruyama = {:0.5f} seconds.'.format(avg_t_EM))
print('Average runtime for Milstein = {:0.5f} seconds.'.format(avg_t_M))
print('Milstein takes {:2.2f}% longer.\n'.format(avg_t_EM / avg_t_M * 100))


# =============================================================================
# Convergence Tests
# =============================================================================

def exact_conv(t, W):
    """
    Redefining the exact solution to work better with a number of realisations.
    """

    return X0 * np.exp((lamda - 0.5 * mu ** 2) * t[:, np.newaxis] + mu * W)


print('Starting convergence tests.. this may take some time.')
# use M realisations this time, so have to change dW to generate M realisations
dW = np.sqrt(dt) * np.random.randn(N, M)
W = np.zeros((N+1, M))
W[1:, :] = np.cumsum(dW, axis=0)
X_Exact = exact_conv(t, W)

# strong convergence
errors_strong_M = np.zeros((len(ratios), M))
errors_strong_EM = np.zeros((len(ratios), M))

for i, r in enumerate(ratios):
    for m in range(M):
        W_r = W[::r, m]
        dW_r = np.hstack((np.diff(W_r), 0))
        X_M = milstein(dW_r, r * dt, f, g, gprime)
        X_EM = EM(dW_r, r * dt, f, g)
        errors_strong_M[i, m] = np.abs(X_M[-1] - X_Exact[-1, m])
        errors_strong_EM[i, m] = np.abs(X_EM[-1] - X_Exact[-1, m])

dT = dt * ratios
strong_error_M = np.mean(errors_strong_M, axis=1)
strong_error_EM = np.mean(errors_strong_EM, axis=1)
strong_fit_M = np.polyfit(np.log(dT), np.log(strong_error_M), 1)
strong_fit_EM = np.polyfit(np.log(dT), np.log(strong_error_EM), 1)

fig = plt.figure(figsize=(24, 6))
ax1 = fig.add_subplot(121)
ax1.loglog(dT, strong_error_M, 'kx')
ax1.loglog(dT, strong_error_EM, 'kx')
ax1.loglog(dT, np.exp(strong_fit_M[1]) * dT ** strong_fit_M[0],
           'k--', label='Milstein Convergence {:2.3f}'.format(strong_fit_M[0]))
ax1.loglog(dT, np.exp(strong_fit_EM[1]) * dT ** strong_fit_EM[0],
           'k--', label='Euler-Maruyama Convergence {:2.3f}'.format(
                   strong_fit_EM[0]))
ax1.set_xlabel('Brownian Time Step, $dt$')
ax1.set_ylabel('Error, $\mathbb{E}$')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
ax1.set_title('Strong Convergence')

# weak convergence
weak_errors_M = np.zeros(len(ratios))
weak_errors_EM = np.zeros(len(ratios))

for i, r in enumerate(ratios):
    dt_r = r * dt
    t_r = t[::r]
    dW = np.sqrt(dt_r) * np.random.randn(N//r, M)
    W = np.zeros((len(t_r), M))
    W[1:, :] = np.cumsum(dW, axis=0)
    X_Exact = exact_conv(t_r, W)

    X_M = np.zeros_like(W)
    X_EM = np.zeros_like(W)
    for m in range(M):
        X_M[:, m] = milstein(dW[:, m], dt_r, f, g, gprime)
        X_EM[:, m] = EM(dW[:, m], dt_r, f, g)

    weak_errors_M[i] = np.abs(np.mean(X_M[-1, :]) - np.mean(X_Exact[-1, :]))
    weak_errors_EM[i] = np.abs(np.mean(X_EM[-1, :]) - np.mean(X_Exact[-1, :]))

dT = dt * ratios
weak_fit_M = np.polyfit(np.log(dT), np.log(weak_errors_M), 1)
weak_fit_EM = np.polyfit(np.log(dT), np.log(weak_errors_EM), 1)

ax2 = fig.add_subplot(122)
ax2.loglog(dT, weak_errors_M, 'kx')
ax2.loglog(dT, weak_errors_EM, 'kx')
ax2.loglog(dT, np.exp(weak_fit_M[1]) * dT ** weak_fit_M[0],
           'k--', label='Milstein Convergence {:2.3f}'.format(weak_fit_M[0]))
ax2.loglog(dT, np.exp(weak_fit_EM[1]) * dT ** weak_fit_EM[0],
           'k--', label='Euler-Maruyama Convergence {:2.3f}'.format(
                   weak_fit_EM[0]))
ax2.set_xlabel('Brownian Time Step, $dt$')
ax2.set_ylabel('Error, $\mathbb{E}$')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
ax2.set_title('Weak Convergence')

plt.savefig('convergence_tests.pdf')
plt.show()
