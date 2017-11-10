# =============================================================================
# Simulation of a flock of birds using Cellular Automato techniques
# =============================================================================

from __future__ import division

from copy import deepcopy
import numpy as np
import scipy as scp
from scipy.optimize import minimize
from matplotlib import pyplot, animation
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


class Agent(object):

    def __init__(self, location, velocity, C=1, A=5, S=0.1):
        self._coords = np.array(location, dtype=np.float64)
        self._vel = np.array(velocity, dtype=np.float64)
        self._cost = (lambda theta, theta_z, theta_v, theta_zmin, delta_zmin:
                      (C * np.cos(theta - theta_z) +
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
                flock_locations[i] = boid._coords
                flock_velocities[i] = boid._vel

            # calculate average location and velocities
            avg_coords = np.mean(flock_locations, axis=0)
            deltaz = avg_coords - self._coords
            avg_vel = np.mean(flock_velocities, axis=0)

            # calculate z_min, the distance to the closest neighbour
            r = np.zeros(N)
            for i, boid in enumerate(neighbours):
                sep = self._coords - flock_locations[i]
                r[i] = np.dot(sep, sep)

            z_min =
            # calculate angle parameters
            theta_z = np.arctan2(avg_coords[1], avg_coords[0])
            theta_v = np.arctan2(avg_vel[1], avg_vel[0])
            theta_dzmin = np.arctan2(z_min[1], z_min[0])
            delta_zmin = np.lingalf.norm(z_min)

            # use scipy.optmize.minimize to minimize the cost function to find
            # the optimum value of theta for the boid
            theta = minimize(self._cost, theta,
                             (theta_z, theta_v, theta_dzmin, delta_zmin),
                             bounds=(-(3/2) * np. pi, (3/2) * np.pi))
