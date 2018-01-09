import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12, 6)

np.random.seed(100)  # set the random seed to be 100


# =============================================================================
# This computes the exact solution!
# =============================================================================

def integrand(x, t):
    return np.sin(t+x)**2*np.exp(-x**2/(2.0*t))/np.sqrt(2.0*np.pi*t)

t_int = np.linspace(0.005, np.pi, 1000)
int_exact = np.zeros_like(t_int)
for i, t in enumerate(t_int):
    int_exact[i], err = quad(integrand, -np.inf, np.inf, args=(t,))


# =============================================================================
# Brownian Noise Thing
# =============================================================================

def W(M, N, dt):

    np.random.seed(100)
    dW = np.sqrt(dt) * np.random.randn(N, M)
    W = np.cumsum(dW, 0) - dW[0, :]
    return W

Ns = [500, 1000, 2000]
Ms = [500, 1000, 2000]

fig = plt.figure(figsize=(24, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for N in Ns:
    t, dt = np.linspace(0.005, np.pi, N, retstep=True)
    for M in Ms:
        # Alex's version ----
        np.random.seed(100)
        mydW = np.sqrt(dt) * np.random.randn(M, N)
        myW = np.cumsum(mydW, 1)
        myU = np.sin(t + myW) ** 2
        myU_mean = np.mean(myU, axis=0)

        # my version ----
        Wx = W(M, N, dt).T
        u = np.sin(t + Wx) ** 2
        u_mean = np.mean(u, axis=0)

        # ax1.plot(t, u.T, alpha=0.1)
        ax1.plot(t, u_mean, '--', label='N = {}, M = {}'.format(N, M))
        # ax2.plot(t, myU.T, alpha=0.1)
        ax2.plot(t, myU_mean, '--', label='N = {}, M = {}'.format(N, M))

ax1.plot(t_int, int_exact, 'k-', label='Exact Solution')
ax1.set_xlim(0.005, np.pi)
ax1.set_ylim(0, 1)
ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.102), loc=3,
           ncol=2, mode="expand", borderaxespad=1.5)
ax1.set_title('Ed')


ax2.plot(t_int, int_exact, 'k-', label='Exact Solution')
ax2.set_xlim(0.005, np.pi)
ax2.set_ylim(0, 1)
ax2.legend(bbox_to_anchor=(0, 1.02, 1, 0.102), loc=3,
           ncol=2, mode="expand", borderaxespad=1.5)
ax2.set_title('Alex')

plt.show()
