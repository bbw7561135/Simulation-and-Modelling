# =============================================================================
# An example of cellular automota: Conway's Game of Life.
#
# Looks at how many neighbours surrounding a cell are alive (1) or dead (0):
# Any live cell with fewer than two live neighbours dies ("under-population")
# Any live cell with two or three neighbours lives ("survival")
# Any live cell with four or more neighbours dies ("over-population")
# Any dead cell with exactly three neigbours lives ("reproduction")
#
# Edward PJohn Parkinson
# e.j.parkinson@soton.ac.uk
# =============================================================================

import numpy as np
from matplotlib import pyplot, animation


def find_neighbours(x):
    """
    Determine the number of alive neighbours surrounding a cell. 

    Parameters
    ----------
    x: n x n array of 0 and 1. 
        The grid of cells. An alive cell has a value of 1 and a dead cell has
        a value of 0.

    Returns
    -------
    nb: n x n array of ints.
        Returns an array the same size as x where each cell has a count of the
        number of alive neighbours. 
    """

    nb = np.zeros_like(x)
    x_neighs = np.zeros(8)

    for i in range(1, len(x)-1):
        for j in range(1, len(x)-1):
            x_neighs[0] = x[i - 1, j - 1]
            x_neighs[1] = x[i - 1, j]
            x_neighs[2] = x[i - 1, j + 1]
            x_neighs[3] = x[i, j - 1]
            x_neighs[4] = x[i, j + 1]
            x_neighs[5] = x[i + 1, j - 1]
            x_neighs[6] = x[i + 1, j]
            x_neighs[7] = x[i + 1, j + 1]

            nb[i, j] = np.sum(x_neighs)

    return nb


def conway_iteration(grid):
    """
    Compute a single step for Conway's Game of Life.

    Parameters
    ----------
    grid: n x n array of 0 and 1.
        The grid of cells which are either dead or alive.

    Returns
    -------
    grid: n x n array of 0 and 1.
        The grid of cells which are either dead or alive.
    """

    # find the number of neighbours which are alive
    neighbours = find_neighbours(grid)

    # where it's 1 on the grid, the 1 survives
    survive = np.logical_and(grid == 1, np.logical_or(neighbours == 2,
                                                      neighbours == 3))
    # where it's 1 on the grid, the 1 is reborn
    reborn = np.logical_and(grid == 0, neighbours == 3)

    # set all the cells to 0
    grid[:, :] = 0
    grid[np.logical_or(survive, reborn)] = 1

    # apply the periodic boundary conditions using ghost cells
    grid[:, 0] = grid[:, -2]
    grid[:, -1] = grid[:, 1]
    grid[0, :] = grid[-2, :]
    grid[-1, :] = grid[1, :]

    return grid


def life_animation(grid, frames=10):
    """
    Returns an animation of Conway's Game of Life.

    Parameters
    ----------
    grid: n x n array of 0 and 1.
        The grid of cells which are either dead or alive.
    frames: int.
        The number of frames to animate for, i.e. the number of iterations.

    Returns
    -------
    An animation of Conway's Game of Life.
    """

    fig = pyplot.figure()
    im = pyplot.imshow(grid[1:-1, 1:-1],
                       cmap=pyplot.get_cmap('gray'),
                       interpolation='nearest')
    im.set_clim(-0.005, 1)

    def init():
        """
        The initialisation function for Matplotlib's animations function. This
        function sets up first frame.
        """

        im.set_array(grid[1:-1, 1:-1])
        return im,

    def animate(i):
        """
        The function used to update the frames for the animation.

        Returns
        -------
        im: func.
            The grid at an iteration.
        """

        conway_iteration(grid)
        im.set_array(grid[1:-1, 1:-1])
        return im

    return animation.FuncAnimation(fig, animate, init_func=init,
                                   interval=100, frames=frames,
                                   blit=True)


# =============================================================================
# Different types of grids to try out..
# =============================================================================


grid_loaf = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0]])

glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])
glider_grid = np.zeros((25, 25))
glider_grid[6:9, 6:9] = glider

# =============================================================================
# Animation
# =============================================================================

anim = life_animation(glider_grid, frames=500)
