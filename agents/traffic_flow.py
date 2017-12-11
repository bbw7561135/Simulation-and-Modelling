# =============================================================================
# Simulation of traffic flow using agents
# Uses ghost cells to as periodic boundary conditions
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12, 6)


def evolve_traffic(grid):
    """
    Evolves a single lane of traffic by looking at each car and seeing if the
    section of road before it is empty or not. If it is empty, the car moves 
    forwards one space. If the road is occupied, the car stays where it is. 
    
    The road is a circular road in that periodic boundary conditions are 
    enforced at the edge of the grid used to simulate the road.

    Parameters
    ----------
    grid: 1 x n array of 0 and 1.
        The road which the cars are moving on.

    Returns
    -------
    grid: 1 x n array of 0 and 1.
        The road after traffic has been evolved for one time step.
    average_speed: float.
        The average speed of all the cars during the time step.
    """
    N = len(grid)
    N_cars = np.sum(grid[1:-1])
    car_moved_to = np.zeros_like(grid)
    car_stays_at = np.zeros_like(grid)

    # Implement the rules for a car to move
    for i in range(1, N-1):
        if grid[i] == 1:
            # if there is a car in grid[i], apply the movement rules
            if grid[i + 1] == 0:
                car_moved_to[i + 1] = 1
            else:
                car_stays_at[i] = 1

    # Update the grid
    grid = np.logical_or(car_moved_to, car_stays_at)

    # look at the ghost cells
    # if the final grid has a car, place it in the first cell
    if grid[-1] == 1:
        grid[1] = grid[-1]

    grid[0] = grid[-2]
    grid[-1] = grid[1]

    # check that the number of cars hasn't changed
    assert(np.sum(grid[1:-1]) == N_cars), "Number of cars has changed."
    average_speed = np.sum(car_moved_to) / N_cars

    return grid, average_speed

# =============================================================================
# Generate the grid:
# Use periodic boundary conditions using ghost cells
# =============================================================================

N = 100
desired_densities = np.linspace(0, 1, 50, endpoint=False)
actual_densities = np.zeros_like(desired_densities)
limiting_v = np.zeros_like(desired_densities)

for i, desired_density in enumerate(desired_densities):
    initial_grid = np.zeros(N + 2)  # N + 2 for ghost cells
    # generate the road
    initial_grid[1:-1] = np.array(np.random.rand(N) < desired_density,
                                  dtype=int)
    if np.sum(initial_grid[1:-1]) == 0:
        initial_grid[1] = 1  # need at least 1 car in the road
    # populate the ghost cells
    initial_grid[0] = initial_grid[-2]
    initial_grid[-1] = initial_grid[1]

    actual_densities[i] = np.sum(initial_grid[1:-1]) / N

    n_steps = 100
    velocity = np.zeros(n_steps)
    grid = initial_grid.copy()

    for step in range(n_steps):
        grid, velocity[step] = evolve_traffic(grid)

    limiting_v[i] = velocity[-1]

# =============================================================================
# Plot velocities as function of traffic density
# =============================================================================

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(actual_densities, limiting_v, 'kx')
ax1.set_xlabel('Traffic density')
ax1.set_ylabel('Limiting velocity')

plt.show()
