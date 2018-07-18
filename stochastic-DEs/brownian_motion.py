import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12, 6)


# =============================================================================
# This computes the exact solution!
# =============================================================================

def integrand(x, t):
    return np.sin(t+x)**2*np.exp(-x**2/(2.0*t))/np.sqrt(2.0*np.pi*t)

t_int = np.linspace(0.005, np.pi, 1000)
int_exact = np.zeros_like(t_int)
for i, t in enumerate(t_int):
    int_exact[i], err = quad(integrand, -np.inf, np.inf, args=(t))


# =============================================================================
# Evaulate using Brownian noise 
# =============================================================================

def W(N, M, dt):
    """

    """

    np.random.seed(100)
    dW = np.sqrt(dt) * np.random.randn(N, M)
    W = np.cumsum(dW, 0) - dW[0, :]
    return W

Ns = [500, 1000, 2000]
Ms = [500, 1000, 2000]

fig = plt.figure()
ax1 = fig.add_subplot(111)

for N in Ns:
    t, dt = np.linspace(0.005, np.pi, N, retstep=True)
    for M in Ms:
        Wx = W(N, M, dt).T
        u = np.sin(t + Wx) ** 2
        u_mean = np.mean(u, axis=0)
      
        ax1.plot(t, u_mean, '--', label='N = {}, M = {}'.format(N, M))

ax1.plot(t_int, int_exact, 'k-', label='Exact Solution')
ax1.set_xlim(0.005, np.pi)
ax1.set_ylim(0, 1)
ax1.legend(loc='upper right')
plt.show()
