# =============================================================================
# Simulation of traffic flow using agents
# =============================================================================

import numpy as np
from matplotlib import pyplt as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12,6)


def evolve_traffic(grid):
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
    # update the ghost cells
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

N = 1000
desired_density = 0.75
initial_grid = np.zeros(N + 2)  # N + 2 for ghost cells
initial_grid[1:-1] = np.array(np.random.rand(N) < desired_density, dtype=int)
# populate the ghost cells
initial_grid[0] = initial_grid[-2]
initial_grid[-1] = initial_grid[1]

actual_density = np.sum(initial_grid[1:-1]) / N
print('Desired density: {}. \nActual density: {}.'.format(desired_density,
                                                          actual_density))

# =============================================================================
# Evolve the grid
# =============================================================================

n_steps = 100
velocity_series = np.zeros(n_steps + 1)
velocity_series[0] = 0
grid = initial_grid

for step in range(n_steps):
    print(step)
    grid, average_speed = evolve_traffic(grid)
    velocity_series[step + 1] = average_speed.copy()
