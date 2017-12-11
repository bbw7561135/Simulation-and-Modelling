import numpy as np

def eq_circle(x, y):
    """
    The equation to calculate the radius of a circle.

    Parameters
    ----------
    x, y: float.
        The x and y coordinate of a random sampling point.

    Returns
    -------
    The radial distance of the sampling point from the centre of the circle.
    """
    
    return np.sqrt(x ** 2 + y ** 2)


def mc_int_circle(f, R, N=10000):
    """
    Monte Carlo integration to find the area of a circle. 

    Parameters
    -----------
    f: func.
        The function to Monte Carlo integrate.
    R: float.
        The radius of the circle.
    N: int.
        The number of Monte Carlo sampling points.

    Returns
    -------
    area_circ: float.
        The area of the circle given its radius R.
    """

    # create random samlping points
    x = R * np.random.rand(N)
    y = R * np.random.rand(N)

    # calculate the area of the square containing the circle to integrate
    area_of_sq = (2 * R) ** 2

    # count the number of random samlping points which are inside the radius
    # of the cricle
    k = np.sum(f(x, y) <= R)

    area_circ = area_of_sq * k / N

    return area_circ


def eq_sphere(x, y, z):
    """
    The equation to calculate the radius of a sphere.

    Parameters
    ----------
    x, y, z: float.
        The coordinates of the random sampling point.

    Returns
    --------
    The radial distance of the sampling point from the centre of the sphere.
    """

    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def mc_int_sphere(f, R, N=10000):
    """
    Monte Carlo integration to find the volume of a sphere.

    Parameters
    -----------
    f: func.
        The function to Monte Carlo integrate.
    R: float.
        The radius of the sphere.
    N: int.
        The number of Monte Carlo sampling points.

    Returns
    -------
    vol_sphere: float.
        The area of the circle given its radius R.
    """

    # create the random sampling points
    x = R * np.random.rand(N)
    y = R * np.random.rand(N)
    z = R * np.random.rand(N)

    # compute the volume of the box containing the sphere
    volume_of_cube = (2 * R) ** 3

    # count the number of random sampling points inside the sphere
    k = np.sum(f(x, y, z) <= R)

    vol_sphere = volume_of_cube * k / N
    
    return vol_sphere 


print(mc_int_circle(eq_circle, 1, N=int(1e6)))
print(mc_int_sphere(eq_sphere, 1, N=int(1e6)))
