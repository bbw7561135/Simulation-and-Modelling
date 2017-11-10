# =============================================================================
# Molecular Dynamics code using Newton's laws.
# =============================================================================


import timeit
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import rcParams
from numba import jit
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12, 6)


# =========================================================================== #
#                                 Functions                                   #
# =========================================================================== #


@jit
def LJ_potential(r_sq, r_c):
    """
    Calculates the Lennard-Jones potential given a seperation r_sq
    """

    dphi = np.where(r_sq < (r_c ** 2), 24 * (2 * (1/(r_sq ** 7)) -
                                             (1/(r_sq ** 4))), 0 * r_sq)

    return dphi


@jit
def reset_particles(x, BoxSize):
    """
    Impose periodic boundaries.
    """

    x[x < 0] += BoxSize
    x[x > BoxSize] -= BoxSize

    return x


@jit
def acceleration_calc(positions, mass, BoxSize, r_c):
    """
    Calculate the position of a system of particles given their x, y, z
    coordinates.
    """

    n_particles = len(positions)
    a = np.zeros_like(positions)

    for i in range(n_particles):
        for j in range(i+1, n_particles):
            r = positions[i, :] - positions[j, :]
            # If any r > BoxSize/2, then there will be a closer particle.
            # This is due to the periodic boundary conditions.
            r[np.abs(r) > BoxSize/2] -= np.sign(r[np.abs(r) > BoxSize/2]) \
                * BoxSize
            r_sq = np.dot(r, r)
            dphi = LJ_potential(r_sq, r_c)
            a[i, :] += dphi * r / mass[i]
            a[j, :] -= dphi * r / mass[j]

    return a


@jit
def time_step(vel, accel, positions, masses, BoxSize, delta_t):
    """
    Compute a time step for the system. Returns the new velocities, positions
    and accelerations.
    """

    positions_new = positions + delta_t * vel + 0.5 * delta_t ** 2 * accel
    positions_new = reset_particles(positions_new, BoxSize)
    vel_star = vel + 0.5 * delta_t * accel
    accel_new = acceleration_calc(positions_new, masses, BoxSize, r_c)
    vel_new = vel_star + 0.5 * delta_t * accel_new

    return positions_new, accel_new, vel_new


@jit
def calc_temp(velocities, positions, masses):
    """
    Calculate the temperature for a given configuration of particles.
    """

    N = len(velocities)
    E_kinetic = np.zeros(N)

    for i in range(N):
        E_kinetic[i] = 0.5 * masses[i] * np.dot(velocities[i, :],
                                                velocities[i, :])

    temp = 2/(3 * N) * np.sum(E_kinetic)

    return temp


# =============================================================================
# Simulation Parameters and Initial States
# =============================================================================


start = timeit.default_timer()

BoxSize = 6.1984
r_c = BoxSize/2
particles = np.loadtxt('input.dat', skiprows=1)
velocities = np.zeros_like(particles)
masses = np.ones(particles.shape[0])
accelerations = acceleration_calc(particles, masses, BoxSize, r_c)
delta_t = 0.005
n_steps = 400

t = np.linspace(0, (n_steps + 1/2) * delta_t, n_steps + 1)
positions_array = np.zeros((n_steps+1, particles.shape[0], particles.shape[1]))
positions_array[0, :, :] = particles.copy()
temperature_array = np.zeros(n_steps+1)
temperature_array[0] = calc_temp(velocities, particles, masses)


# =============================================================================
# Time Evolution
# =============================================================================


for i in range(n_steps):
    particles, accelerations, velocities = time_step(velocities, accelerations,
                                                     particles, masses,
                                                     BoxSize, delta_t)
    temperature = calc_temp(velocities, particles, masses)
    positions_array[i+1, :, :] = particles.copy()
    temperature_array[i+1] = temperature


# =============================================================================
# Plot Positions and Temperature
# =============================================================================


# position of one particle as a function of time
fig = plt.figure()
plt.plot(t, positions_array[:, 99, 0] - positions_array[0, 99, 0], label='x')
plt.plot(t, positions_array[:, 99, 1] - positions_array[0, 99, 1], label='y')
plt.plot(t, positions_array[:, 99, 2] - positions_array[0, 99, 2], label='z')
plt.ylabel('Position in box'), plt.xlabel('Time, t')
plt.legend()
plt.savefig('MD_positions_t.pdf')
plt.show()

# 3d plot of trajectories of two particles
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in [129, 148]:
    ax.plot3D(positions_array[:, i, 0] - positions_array[0, i, 0],
              positions_array[:, i, 1] - positions_array[0, i, 1],
              positions_array[:, i, 2] - positions_array[0, i, 2], 'o')
plt.savefig('MD_trajectories_1.pdf')
plt.show()

# 3d plot of trajectories of all particles
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(positions_array.shape[1]):
    ax.plot3D(positions_array[:, i, 0],
              positions_array[:, i, 1],
              positions_array[:, i, 2], 'o')
plt.savefig('MD_trajectories_2.pdf')
plt.show()

# plot the temperature as a function of time
fig = plt.figure()
plt.plot(t, temperature_array)
plt.xlabel('Time, t'), plt.ylabel('Temperature, T')
plt.savefig('MD_temperature_t.pdf')
plt.show()

stop = timeit.default_timer()


# =============================================================================
# Printing stuff
# =============================================================================


print('Run time {:6.2f} seconds'.format(stop - start))
