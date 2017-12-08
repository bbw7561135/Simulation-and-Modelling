# =============================================================================
# Evaluate the SDE,
#           dX = lambda X(t) dt + mu X(t) dW(t),
# using the Euler-Maruyama and Millstein methods.
# =============================================================================

import timeit
import numpy as np
from matplotlib import pyplot as plt

# define some global variables for the functions here because I'm lazy

X0 = 1
mu = 1
lamda = 2
t_end = 1
N = 500

np.random.seed(100)  # define a seed for reproducible solutions


def exact_sol(t, W):
    """
    Define the exact solution to the SDE.
    """
    return X0 * np.exp((lamda - 0.5 * mu ** 2) * t + mu * W)


def EM(dW, dt):
    """
    The Euler-Maruyama method of evaluating an SDE.
    """

    def EM_step(X, h, dW):
        """
        The stepping alogorithm for the Euler-Maruyama method.
        """
        fn = lamda * X
        gn = mu * X

        return X + fn * h + gn * dW

    N = len(dW)
    X_EM = np.zeros(N+1)
    X_EM[0] = X0

    for i in range(N):
        X_EM[i+1] = EM_step(X_EM[i], dt, dW[i])

    return X_EM


def milstein(dW, dt):
    """
    Milstein's method for evaulating an SDE.
    """

    def milstein_step(X, h, dW):
        """
        The stepping alogorithm for the Euler-Maruyama method.
        """

        fn = lamda * X
        gn = mu * X
        gnprime = mu

        return X + h * fn + gn * dW + 0.5 * gn * gnprime * (dW ** 2 - h)

    N = len(dW)
    X_Mi = np.zeros(N+1)
    X_Mi[0] = X0

    for i in range(N):
        X_Mi[i+1] = milstein_step(X_Mi[i], dt, dW[i])

    return X_Mi


t, h = np.linspace(0, t_end, N+1, retstep=True)
dW = np.sqrt(h) * np.random.randn(N)
W = np.zeros(N+1)
W[1:] = np.cumsum(dW)

X_EM = EM(dW, h)
X_M = milstein(dW, h)
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

time_EM = timeit.Timer('EM(dW, dt)', setup='from __main__ import EM, dW, dt')
time_M = timeit.Timer('milstein(dW, dt)',
                      setup='from __main__ import milstein, dW, dt')
avg_t_EM = time_EM.autorange()[1]
avg_t_M = time_M.autorange()[1]
print('Average runtime for Euler-Maruyama = {:0.5f} seconds.'.format(avg_t_EM))
print('Average runtime for Milstein = {:0.5f} seconds.'.format(avg_t_M))
print('Milstein takes {:2.2f}% longer.'.format(avg_t_EM / avg_t_M * 100))
