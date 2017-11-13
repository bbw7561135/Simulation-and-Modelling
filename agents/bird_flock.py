# =============================================================================
# Simulation of a flock of birds using Cellular Automato techniques
# =============================================================================

from __future__ import division

from copy import deepcopy
import numpy as np
import scipy as scp
from scipy.optimize import minimize_scalar
from matplotlib import pyplot, animation
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


class Agent(object):

    def __init__(self, location, velocity, C=1, A=5, S=0.1):
        self._coords = np.array(location, dtype=np.float64)
        self._vel = np.array(velocity, dtype=np.float64)
        self._cost = (lambda theta, theta_z, theta_v, theta_zmin, delta_zmin:
                      - (C * np.cos(theta - theta_z) +
                         A * np.cos(theta - theta_v) -
                         S * np.cos(theta-theta_zmin)/delta_zmin ** 2))

    def step(self, dt):
        self._coords += dt * self._vel

    def steer(self, neighbours):
        N = len(neighbours)
        if N:  # if there are neighbours..
            # record the location and velocities of the nighbours
            flock_locations = np.zeros((N, 2))
            flock_velocities = np.zeros((N, 2))
            for i, boid in enumerate(neighbours):
                flock_locations[i, :] = boid._coords
                flock_velocities[i, :] = boid._vel

            # calculate average location and velocitiy of flock
            avg_coords = np.mean(flock_locations, axis=0)
            avg_vel = np.mean(flock_velocities, axis=0)

            # calculate angle parameters ======================================
            # direction to average location
            dz = avg_coords - self._coords  # distance from flock location
            theta_z = np.arctan2(dz[1], dz[0])
            # direction of average velocity
            theta_v = np.arctan2(avg_vel[1], avg_vel[0])

            # direction and distance to closest neighbour =====================
            # disance from other boids
            deltaz = flock_locations - self._coords
            # find closest neighbour
            z_min = deltaz[np.argmin(np.linalg.norm(deltaz, axis=1)), :]
            # calculate the angles now
            theta_dzmin = np.arctan2(z_min[1], z_min[0])  # angle to neighbour
            delta_zmin = np.linalg.norm(z_min)  # sep from nearest neighbour

            # calculate new angle for boid ====================================
            # use scipy.optmize.minimize to minimize the cost function to find
            # the optimum value of theta for the boid
            theta_min = minimize_scalar(self._cost, bracket=(-(3/2) * np.pi,
                                                              (3/2) * np.pi),
                                        args=(theta_z, theta_v, theta_dzmin,
                                              delta_zmin))

            # set theta to the maximised value found
            theta = theta_min.x
            vel_mag = np.linalg.norm(self._vel)
            # calculate new velocity vector
            self._vel[0] = vel_mag * np.cos(theta)
            self._vel[1] = vel_mag * np.sin(theta)

a1 = Agent([-1.0, 0.0], [0.0, 1.0])
a2 = Agent([1.0, 0.0], [0.0, 1.0])
a1.steer([a2])
print("One neighbour, separation 2")
print(a1._coords, a1._vel, np.linalg.norm(a1._vel))
