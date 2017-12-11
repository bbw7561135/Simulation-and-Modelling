# =============================================================================
# Simulation of a flock of birds
# =============================================================================

from __future__ import division

from copy import deepcopy
import numpy as np
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (24, 12)


class Boid(object):
    """
    Constructs a boid which contains the boid's location and velocity.
    Functions in the class evolve's the boid's location and steer the boid
    due to the influence of it's neighbours.

    Class functions
    ---------------
    _step: Evolves the boid's location by a timestep dt using the Euler
        stepping method.
    s+teer: Steers the heading of a boid by taking into account the influence
        of it's neighbours on a utilty function.

    """

    def __init__(self, location, velocity, C=1, A=5, S=0.25):
        """
        Creates an object boid which contains the coordinates and velocities
        for a boid. The Boid object also contains the utility function which
        is used to change the heading of a boid and is control by three
        parameters: C, A and S.

        Parameters
        ----------
        location: 1x2 array of floats.
            The 2D location of the boid.
        velocity: 1 x 2 array of floats.
            The 2D velocity of the boid.
        C: float.
            The cohesion parameter of the utilty function used to steer
            a boid. The cohesion controls how much a boid will steer towards
            the average location of a flock.
        A: float.
            The alignment parameter of the utility function used to steer a
            boid. The alignment controls how much a boid will steer towards
            the average heading of the flock.
        S: float.
            The separation parameter. The separation parameter controls how
            much a boid will steer to avoid crowding in the flock.
        """

        # the coordinates of the boid
        self._coords = np.array(location, dtype=np.float64)
        # the velocity of the boids
        self._vel = np.array(velocity, dtype=np.float64)
        # the utility function to calculate the heading of the boids
        self._cost = (lambda theta, theta_z, theta_v, theta_zmin, delta_zmin:
                      - (C * np.cos(theta - theta_z) +
                         A * np.cos(theta - theta_v) -
                         S * np.cos(theta-theta_zmin)/delta_zmin ** 2))

    def _step(self, dt):
        """
        Compute a timestep for a boid to update it's position using the Euler
        stepping method.

        Parameters
        ----------
        dt: float.
            The size of the timestep.
        """

        if dt <= 0:
            raise ValueError('The timestep must be greater than zero.')

        self._coords += dt * self._vel

    def _steer_boid(self, neighbours):
        """
        Compute a boid's new velocity due to the influence of the boid's
        neighbours in its immediate area.

        Parameters
        ----------
        neighbours: 1xN array of floats.
            N is the number of neighbours, where the neighbours in the array
            should be Boid objects.
        """

        N = len(neighbours)
        if N:  # if there are neighbours
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
            theta_min = minimize_scalar(self._cost, bounds=(-(3/2) * np.pi,
                                                             (3/2) * np.pi),
                                        args=(theta_z, theta_v, theta_dzmin,
                                              delta_zmin), method='bounded')

            # set theta to the maximised value found
            theta = theta_min.x
            vel_mag = np.linalg.norm(self._vel)
            # calculate new velocity vector
            self._vel[0] = vel_mag * np.cos(theta)
            self._vel[1] = vel_mag * np.sin(theta)


class Flock(object):
    """
    Constructs a flock of boids and provides functions to return data about the
    flock as well as a function to evolve the flock.

    Class functions
    ---------------
    _step: Computes a timestep for a flock by looking at each boid and
        finding it's neighbours.
    locations: Creates an array containing the locations of the all of the
        boids in a flock.
    _flock_average_location: Computes the centre of a flock by looking at the
        location of all the boids in a flock.
    _flock_width: Computes the average width of the entire flock.
    """

    def __init__(self, locations, velocities, C=1, A=5, S=0.25, rl=1):
        """
        Create a Flock object containing the location and velocities of the
        boids making up a flock.

        Parameters
        ----------
        locations: N x 2 array of floats.
            An array containing all of the (x, y) locations of each boid. N is
            the number of boids in the flock.
        velocities: N x 2 array of floats.
            An array containing all of the (Vx, Vy) velocities of each boid.
            N is the number of boids in the flock.
        C: float.
            The Cohesion parameter which controls how much a boid will steer
            towards the center of mass of a flock.
        A: float.
            The alignment parameter of the utility function used to steer a
            boid. The alignment controls how much a boid will steer towards the
            average heading of the flock.
        S: float.
            The separation parameter. The separation parameter controls how
            much a boid will steer to avoid crowding in the flock.
        rl: float.
            The looking radius of a boid, i.e. how far it can see and be
            affected by it's neighbouring boids.
        """

        if rl <= 0:
            raise ValueError('Boids are (hopfully) not blind, nor do they \
                             have a negative looking radius.')

        self._flockmate_locations = np.array(locations, dtype=np.float64)
        self._flockmate_velocities = np.array(velocities, dtype=np.float64)
        self._rl = rl  # rl is the looking radius, i.e. how far a boid can see
        self._boids = []  # list contaning all the boids in the flock
        for location, velocity, in zip(locations, velocities):
            self._boids.append(Boid(location, velocity, C, A, S))

        assert(len(self._boids) > 0), 'No boids in the flock :-('

    def _step(self, dt):
        """
        Compute a timestep where each agent will have it's velocity and
        location updated due to the influence of it's neighbours.

        Parameters
        ----------
        dt: float.
            The length of a time step to update a boid's position.
        """

        if dt <= 0:
            raise ValueError('The timestep must be greater than zero.')

        # create list of current agents at time step
        current_boids = []
        for agent in self._boids:
            current_boids.append(deepcopy(agent))

        # change the heading of the boid i in list _boids, i.e. all the boids
        for i, boid in enumerate(self._boids):
            # create list of neighbouring boids
            boid_neighbours = []
            for j, another_boid in enumerate(current_boids):
                if j != i:  # when another_boid != boid
                    sep = boid._coords - another_boid._coords
                    dist = np.linalg.norm(sep)
                    # if the boid is in the looking radis of the other
                    # boid, then it will be influenced by this boid so add it
                    # to the list of the boid's neighbours
                    if dist < self._rl:
                        boid_neighbours.append(another_boid)

            assert(len(boid_neighbours) <= len(current_boids)), \
                'A boid has more neighbours than there are boids in total.'

            # change the boid's heading due to the neighbours and compute
            # a timestep
            boid._steer_boid(boid_neighbours)
            boid._step(dt)

            # now update the positions of the boid in the locations array
            for i, boid in enumerate(self._boids):
                self._flockmate_locations[i, :] = boid._coords
                self._flockmate_velocities[i, :] = boid._vel

    def _flock_average_location(self):
        """
        Compute the centre of the flock by calculating the average location
        of all the boids in a flock. Requires there to be a list of all
        the boid locations in the flock.

        Returns
        -------
        A 1 x 2 array of floats cointaing the coordinates for the centre of the
        flock, using the average location of the boids.
        """

        return np.mean(self._flockmate_locations, axis=0)

    def _flock_width(self):
        """
        Compute the width of the flock by taking the average of the distance
        each boid is from the (average) centre of the flock.

        Returns
        -------
        A float value for the width of the flock.
        """

        sep = np.abs(self._flockmate_locations -
                     self._flock_average_location())

        return np.mean(np.linalg.norm(sep, axis=1))

# =============================================================================
# Simulation Parameters and Time Steps
# =============================================================================

n_boids = 50
boid_locations = 5 * np.random.rand(n_boids, 2)
boid_velocities = np.ones_like(boid_locations) + 0.01 * \
    np.random.rand(n_boids, 2)

# define the flock
flock = Flock(boid_locations, boid_velocities)

# define simulation parameters
n_steps = 500
dt = 0.01

# create arrays to store things
locations = np.zeros((n_steps, n_boids, 2))
average_width = np.zeros(n_steps)
time_steps = np.zeros(n_steps)

# create arrays to store things
locations = np.zeros((n_steps, n_boids, 2))
width = np.zeros(n_steps)
time_steps = np.zeros(n_steps)

# set the first entries
locations[0, :, :] = flock._flockmate_locations
width[0] = flock._flock_width()

# evolve the flock
for step in range(1, n_steps):
    flock._step(dt)  # call the Flock stepping function
    locations[step, :, :] = flock._flockmate_locations
    width[step] = flock._flock_width()
    time_steps[step] = time_steps[step - 1] + dt

# =============================================================================
# Plot the flock
# =============================================================================

fig = plt.figure()

ax1 = fig.add_subplot(121)
ax1.plot(locations[0, :, 0], locations[0, :, 1], 'k*', 
         label='Initial position')
ax1.plot(locations[:, :, 0], locations[:, :, 1], label='Boids')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Flock locations')

ax2 = fig.add_subplot(122)
ax2.plot(time_steps, average_width)
ax2.set_xlabel('Time, t')
ax2.set_ylabel('Average Width')
ax2.set_title('Average width of the flock')

plt.savefig('boid_simulation.pdf')
plt.show()
