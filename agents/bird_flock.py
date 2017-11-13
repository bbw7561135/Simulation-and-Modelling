# =============================================================================
# Simulation of a flock of birds using Cellular Automato techniques
# =============================================================================

from __future__ import division

from copy import deepcopy
import numpy as np
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12, 6)


class Agent(object):
    def __init__(self, location, velocity, C=1, A=5, S=0.1):
        """
        Defines an object to contain the coordinates and velocity for an agent
        boid. Also defines the utilty function to minimise to find the new
        heading for a boid.
        """
        self._coords = np.array(location, dtype=np.float64)
        self._vel = np.array(velocity, dtype=np.float64)
        self._cost = (lambda theta, theta_z, theta_v, theta_zmin, delta_zmin:
                      - (C * np.cos(theta - theta_z) +
                         A * np.cos(theta - theta_v) -
                         S * np.cos(theta-theta_zmin)/delta_zmin ** 2))

    def step(self, dt):
        """
        Compute a timestep for a boid to update it's position.
        """
        self._coords += dt * self._vel

    def steer(self, neighbours):
        """
        Compute a boid's new velocity due to the influence of it's neighbours.
        """
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


class Flock(object):
    def __init__(self, locations, velocities, rl=1):
        """
        Define an object for holding the locations, velocities and the agents
        which make up the flock.
        """
        self._flockmate_locations = np.array(locations, dtype=np.float64)
        self._flockmate_velocities = np.array(velocities, dtype=np.float64)
        self._rl = rl  # rl is the looking radius
        self._agents = []
        for location, velocity, in zip(locations, velocities):
            self._agents.append(Agent(location, velocity))

    def step(self, dt):
        """
        Compute a timestep where each agent will have it's velocity and
        location updated.
        """
        # create list of current agents at time step
        current_agents = []
        for agent in self._agents:
            current_agents.append(deepcopy(agent))

        # change the heading of all the agents
        for i, boid in enumerate(self._agents):
            # create list of neighbours for agent from current_agents
            # check to see that the agent in the current_agents list is not
            # the agent being look at
            # check that the agent is in the viewing radius of the flock
            boid_neighbours = []
            for j, another_boid in enumerate(current_agents):
                if i != j:  # when boid != another_boid
                    sep = boid._coords - another_boid._coords
                    dist = np.linalg.norm(sep)
                    # if the boid is in the looking radis of the other
                    # boid, then it will have its heading changed so add
                    # to the list of neighbours
                    if dist < self._rl:
                        boid_neighbours.append(another_boid)

            # change the boid's heading due to the neighbours and compute
            # a timestep
            boid.steer(boid_neighbours)
            boid.step

    def locations(self):
        for i, boid in enumerate(self._agents):
            self._flockmate_locations[i, :] = boid._coords

        return self._flockmate_locations

    def velocities(self):
        for i, boid in enumerate(self._agents):
            self._flockmate_velocities[i, :] = boid._vel

        return self._flockmate_velocities

    def average_location(self):
        return np.mean(self._flockmate_locations, axis=0)

    def average_velocity(self):
        return np.mean(self._flockmate_velocities, axis=0)

    def average_width(self):
        sep = self._flockmate_locations - self.average_location()

        return np.mean(np.linalg.norm(sep, axis=1))
