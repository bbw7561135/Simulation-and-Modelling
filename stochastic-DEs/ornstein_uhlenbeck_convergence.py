import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12, 6)


# =============================================================================
# ODE Functions
# =============================================================================

def EM(dW, h, X0, f, g):
    """
    Compute the approximate solution X(t) of an SDE of the form,

        dX = fdt + gdW,

    using the Euler-Maruyama method.

    Parameters
    ----------
    dW: (1 x t) array of floats.
        The value of the random noise at each point t; this is not a cum sum
        unlike W.
    h: float.
        The grid spacing for t.
    X0: float.
        The initial value of the solution, given by X(0).
    f: function.
        A function to define the deterministic part of the SDE.
    g: function.
        A function to define the stochasic part of the SDE.

    Returns
    -------
    X_EM: (1 x t+1) array of floats.
        The approximate value of X(t) at each point t as given by the Euler-
        Maryuama method.
    """

    if h <= 0:  # h has to be positive
        raise ValueError('The grid spacing, h, can\'t be negative or zero.')

    # at least 2 increments are needed to evaulate the SDE
    assert(dW.shape[0] > 1), 'Not enough Wiener increments to evaulate.'

    # check values are finite
    assert(np.isfinite(h)), 'h needs to be a finite number.'
    assert(np.isfinite(X0)), 'X0 needs to be a finite number.'

    # check that the function arguments are callable
    assert(callable(f)), 'The argument for f is not a callable function.'
    assert(callable(f)), 'The argument for g is not a callable function.'

    def EM_step(X, dW):
        """
        The stepping alogorithm for the Euler-Maruyama method. Takes in a value
        X(t) and returns X(t+1).

        Parameters
        ----------
        X: float.
            The value of X(t) at a given point, t.
        dW: float.
            The random noise dW at the point t.

        Returns
        -------
        Xn: float.
            The value of X(t+h).
        """

        fn = f(X)
        gn = g(X)

        Xn = X + fn * h + gn * dW

        # check that Xn is finite, otherwise the solver has blown up
        assert(np.isfinite(Xn))

        return Xn

    N = dW.shape[0]
    X_EM = np.zeros(N+1)
    X_EM[0] = X0

    for i in range(N):
        X_EM[i+1] = EM_step(X_EM[i], dW[i])

    # check that the initial value has not been overwritten
    assert(X_EM[0] == X0), 'Initial value has been lost.'

    return X_EM


def TM(dW, h, X0, theta, mu, lamda):
    """
    Compute the approximate solution X(t) of the Ornstein-Uhlenbeck equation,

        dX = - lambda * X * dt + mu * dW,

    using the Theta method.

    Parameters
    ----------
    dW: (1 x t) array of floats.
        The value of the random noise at each point t; this is not a cum sum
        unlike W.
    h: float.
        The grid spacing for t.
    X0: float.
        The initial value of the solution, given by X(0).
    theta: float.
        The value of the theta parameter. When theta = 0, the theta method
        reduces to the Euler-Maruyama method.
    lamda: float.
        The value of the lambda parameter, i.e. the value for the deterministic
        part of the OU equation.
    mu: function.
        The value of the mu parameter, i.e. the value for the stochastic
        part of the OU equation.

    Returns
    -------
    X_Theta: (1 x t+1) array of floats.
        The approximate value of X(t) at each point t as given by the theta
        method.
    """

    if h <= 0:  # h has to be positive
        raise ValueError('the grid spacing, h, can\'t be negative or zero')

    # at least 2 increments are needed to evaulate the SDE
    assert(dW.shape[0] > 1), 'Not enough Wiener increments to evaulate.'

    # the parameter values have to be finite, otherwise things explode
    assert(np.isfinite(h)), 'h needs to be a finite number.'
    assert(np.isfinite(X0)), 'X0 needs to be a finite number.'
    assert(np.isfinite(theta)), 'theta needs to be a finite number.'
    assert(np.isfinite(mu)), 'mu needs to be a finite number.'
    assert(np.isfinite(lamda)), 'lamda needs to be a finite number.'

    def theta_step(X, dW):
        """
        The stepping alogorithm for the Euler-Maruyama method. Takes in a value
        X(t) and returns X(t+1).

        Parameters
        ----------
        X: float.
            The value of X(t) at a given point, t.
        dW: float.
            The random noise dW at the point t.

        Returns
        -------
        Xn: float.
            The value of X(t+h).
        """

        Xn = (X - lamda * h * (1 - theta) * X + mu * dW) / \
            (1 + lamda * h * theta)

        # check that Xn is finite, otherwise the solver has blown up
        assert(np.isfinite(Xn))

        return Xn

    N = dW.shape[0]
    X_Theta = np.zeros(N+1)
    X_Theta[0] = X0

    for i in range(N):
        X_Theta[i+1] = theta_step(X_Theta[i], dW[i])

    # check that the initial value has not been overwritten
    assert(X_Theta[0] == X0), 'Initial value has been lost.'

    return X_Theta


def OU_Expectation(X0, mu, lamda, t):
    """
    Computes the expectation and the square of the expectation value of
    the Ornstein-Uhlenbeck problem.

    Parameters
    ----------
    X0: float.
        The initial value of the solution, given by X(0).
    mu: float.
        The scaling parameter for the stochastic part of the O-U equation.
    lamda: float.
        The scaling parameter for the deterministic part of the O-U
        equation.
    t: (1 x t) array of floats.
        The values of t to evaulate the O-U expected values.

    Returns
    -------
    Ex: (1 x t) array of floats.
        The expectation value of the O-U equation at each value of t.
    ExSq: (1 x t) array of floats.
        The expectation value of the square value of the O-U equation at each
        value of t.
    """

    assert(t.shape[0] > 1), 'More than one grid space is needed to calculate \
        the expected values.'
    # make sure the values we are being given are finite
    assert(np.isfinite(X0))
    assert(np.isfinite(mu))
    assert(np.isfinite(lamda))

    Ex = np.exp(-lamda * t) * X0
    ExSq = np.exp(-2 * lamda * t) * X0 ** 2 + (mu ** 2)/(2 * lamda) * \
        (1 - np.exp(-2 * lamda * t))

    return Ex, ExSq


# =============================================================================
# Functions/constants for solvers
# =============================================================================

def fn(x):
    """
    A function to define the deterministic part of the O-U equation.

    Parameters
    ----------
    x: float.
        A grid location x.

    Returns
    -------
    The value of the deterministic coefficent at the point x.
    """

    lamda = 1
    return - lamda * x


def gn(x):
    """
    A function to define the stochastic part of the O-U equation.

    Parameters
    ----------
    x: float.
        A grid location x.

    Returns
    -------
    The value of the deterministic coefficent at the point x.
    """

    mu = 1
    return mu


X0 = 1          # the starting value for the O-U equation
lamda = 1       # the parameter for the deterministic part of the O-U equation
mu = 1          # the parameter for the stochastic part of the O-U equation

# =============================================================================
# Weak Convergence - E-M
# =============================================================================

# the weak convergence of the method is being tested using time steps,
# dt = 2 ** -(4, 5, .. 11). Thus 7 resolutions will be used with the number of
# paths being (16) up to (1024) paths.

t_start = 0                      # starting value to solve
t_end = 1                        # end value to solve to
N = 2 ** 10                      # refernce number of Wiener increments
M = 10 ** 5                      # reference number of realisations
n_resolutions = 7                # the number of resolutions to use
resolutions = 2 ** np.arange(n_resolutions)
t, dt = np.linspace(t_start, t_end, N+1, retstep=True)  # reference t, dt vals

# as the expectation value is deterministic, it only needs to be calculated
# once as the final value will stay the same for each resolution
EX, EXSq = OU_Expectation(X0, mu, lamda, t)  # calc expected values

dT = np.zeros(len(resolutions))
weak_errors = np.zeros(len(resolutions))
for i, r in enumerate(resolutions):
    dT[i] = dt_r = r * dt  # calculate grid spacing for N points on grid
    t_r = t[::r]           # use ref grid to get grid points location
    dW = np.sqrt(dt_r) * np.random.randn(N//r, M)

    # calculate solution for each realisation
    X_EM = np.zeros((len(t_r), M))
    for m in range(M):
        X_EM[:, m] = EM(dW[:, m], dt_r, X0, fn, gn)

    # compute average path over all the realisations and compute the
    # errors in X ** 2, as the values will always be positive
    weak_errors[i] = np.abs(np.mean(np.square(X_EM[-1, :])) - EXSq[-1])

# fit linear line to data to find convergence
weak_fit = np.polyfit(np.log(dT), np.log(weak_errors), 1)

plt.loglog(dT, weak_errors, 'kx')
plt.loglog(dT, np.exp(weak_fit[1]) * dT ** weak_fit[0],
           'k--', label='E-M Convergence {:2.3f}'.format(weak_fit[0]))
plt.xlabel('dt')
plt.ylabel('Error')
plt.legend()
plt.title('Euler-Maruyama Convergence')
plt.savefig('EM_converge.pdf')
plt.show()

# =============================================================================
# Histogram - E-M
# =============================================================================

t_start = 0
t_end = 10
N = 1280                    # using this will return dt = 2 ** -7
M = 10 ** 5
t_end = 10
t, dt = np.linspace(t_start, t_end, N+1, retstep=True)
dW = np.sqrt(dt) * np.random.randn(N, M)

X_EM = np.zeros((N+1, M))
for m in range(M):
    X_EM[:, m] = EM(dW[:, m], dt, X0, fn, gn)

# plot X(10) for each realisation on a histogram. The values of X(10) are
# expected to follow a normal distribution as E-M samples the Wiener increments
# randomly from a normal distribution
plt.hist(X_EM[-1, :], bins=50, edgecolor='black', linewidth=1.0, normed=True)
plt.xlabel('X(10)')
plt.ylabel('Normed Frequency')
plt.title('Euler-Maruyama X(10) Histogram')
plt.show()

# =============================================================================
# E-M limit as T -> infinty
# =============================================================================

# compute the variance and mean of the computed solution and construct a normal
# distribution using these values. Compare this distribution to the normal
# distribution computed using the mean and variance of the expected solution
# of the O-U equation.

t = np.linspace(-4, 4, 100)  # linspace for the norm dists

# construct the norm dist for the reference norm
ref_variance = mu ** 2/(2 * lamda - lamda ** 2 * dt)
ref_sigma = np.sqrt(ref_variance)
ref_mean = 0
ref_norm = norm.pdf(t, loc=ref_mean, scale=ref_sigma)

# construct the norm dist using the computed solution
EM_var = np.mean(np.mean(X_EM ** 2, axis=1))
EM_sigma = np.sqrt(EM_var)
EM_mean = np.mean(np.mean(X_EM, axis=1))
EM_norm = norm.pdf(t, loc=EM_mean, scale=EM_sigma)

plt.plot(t, ref_norm, '--',
         label=r'$N(0, \frac{\mu^2}{2\lambda - \lambda^2 \delta t})$')
plt.plot(t, EM_norm, '-', label=r'$X^h(t)$')
plt.xlabel('t')
plt.ylabel(r'$N(\mu, \sigma^2)$')
plt.legend()
plt.show()

# =============================================================================
# Weak Convergence - TM
# =============================================================================

t_start = 0
t_end = 1
N = 2 ** 10
M = 10 ** 5
n_resolutions = 7
resolutions = 2 ** np.arange(n_resolutions)
t, dt = np.linspace(t_start, t_end, N+1, retstep=True)
EX, EXSq = OU_Expectation(X0, mu, lamda, t)

theta = 0.5  # the value of the theta parameter, for the theta method

dT = np.zeros(len(n_resolutions))
weak_errors = np.zeros(len(n_resolutions))
for i, r in enumerate(resolutions):
    dT[i] = dt_r = r * dt
    t_r = t[::r]
    dW = np.sqrt(dt_r) * np.random.randn(N//r, M)

    # calculate solution for each realisation
    X_THETA = np.zeros((len(t_r), M))
    for m in range(M):
        X_THETA[:, m] = TM(dW[:, m], dt_r, X0, theta, mu, lamda)

    weak_errors[i] = np.abs(np.mean(np.square(X_THETA[-1, :])) - EXSq[-1])

weak_fit = np.polyfit(np.log(dT), np.log(weak_errors), 1)
plt.loglog(dT, weak_errors, 'kx')
plt.loglog(dT, np.exp(weak_fit[1]) * dT ** weak_fit[0],
           'k--', label='Theta Convergence {:2.3f}'.format(weak_fit[0]))
plt.xlabel('dt')
plt.ylabel('Error')
plt.legend()
plt.title('Theta Convergence')
plt.savefig('Theta_cnovergence.pdf')
plt.show()

# =============================================================================
# Histogram - TM
# =============================================================================

t_start = 0
t_end = 10
N = 1280
M = 10 ** 5
t_end = 10
t, dt = np.linspace(t_start, t_end, N+1, retstep=True)
dW = np.sqrt(dt) * np.random.randn(N, M)

X_THETA = np.zeros((N+1, M))
for m in range(M):
    X_THETA[:, m] = TM(dW[:, m], dt, X0, theta, mu, lamda)

plt.hist(X_THETA[-1, :], bins=50, edgecolor='black', linewidth=1.0,
         normed=True)
plt.xlabel('X(10)')
plt.ylabel('Normed Frequency')
plt.title('Theta X(10) Histogram')
plt.show()

# =============================================================================
# TM limit as T -> infinty
# =============================================================================

t = np.linspace(-4, 4, 100)  # linspace for norm dist

# calculate the norm dist using the computed solution
TM_var = np.mean(np.mean(X_THETA ** 2, axis=1))
TM_sigma = np.sqrt(TM_var)
TM_mean = np.mean(np.mean(X_THETA, axis=1))
TM_norm = norm.pdf(t, loc=TM_mean, scale=TM_sigma)

plt.plot(t, ref_norm, '--',
         label=r'$N(0, \frac{\mu^2}{2\lambda - \lambda^2 \delta t})$')
plt.plot(t, EM_norm, '-', label=r'$X^h(t)$')
plt.xlabel('t')
plt.ylabel('Frequency')
plt.legend()
plt.show()
