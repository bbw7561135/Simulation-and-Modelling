import numpy as np

def f(x):
    """
    The function to Monte Carlo integrate.

    Parameters
    ----------
    x: float.
        The coordinate location of a random point.

    Returns
    -------
    x * x: float.
        The coordinate location squared.
    """
    
    return x * x


def mc_integrate_multid(f, R, num_dimen, N=100000):
    """
    Monte Carlo integration to compute the volume of an N-dimensional 
    hypersphere.

    Parameters
    ----------
    f: func.
        The function to Monte Carlo integrate.
    R: float.
        The radius of the N-dimensional hypershpere.
    num_dimen: int.
        The number of dimensions of the hypersphere.
    N: int.
        The number of Monte Carlo sampling points.

    Returns
    ------- 
    N_Vol: float.
        The volume of the hypershere given the number of dimensions and its
        radius R.
    """

    # the domain is going to be a list of lists
    # volume is the volume of the box, i.e. the volume of the domain

    volume = (2 * R) ** num_dimen

    xs = np.zeros((num_dimen, N))
    for i in range(num_dimen):
        xs[i] = R * np.random.rand(N)

    k = 0
    for j in range(N):
        sum = 0
        for i in range(num_dimen):
            comp = xs[i][j]
            sum += f(comp)
        if sum <= R ** 2:
            k += 1

    N_Vol = volume * k / N

    return N_Vol

print(mc_integrate_multid(f, 1, 3))
