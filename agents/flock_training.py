# =============================================================================
# Simulation of a flock of birds
# =============================================================================

import numpy as np
from copy import deepcopy
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (24, 12)


class Boid(object):
    """
    Description of the class.
    """

    def __init__(self, location, velocity, C, A=5, S=0.25):
        """
        Creates an object boid which contains the coordinates and velocities
        for a boid. The Boid object also contains the utility function which
        is used to change the heading of a boid and is control by three
        parameters: C, A and S.

        Parameters
        ----------
        location: 1x2 array. The 2D location of the boid.
        velocity: 1x2. The 2D velocity of the boid.
        C: float. The cohesion parameter of the utilty function used to steer
            a boid. The cohesion controls how much a boid will steer towards
            the average location of a flock.
        A: float. The alignment parameter of the utility function used to
            steer a boid. The alignment controls how much a boid will steer
            towards the average heading of the flock.
        S: float. The separation parameter. The separation parameter controls
            how much a boid will steer to avoid crowding in the flock.
        """

        self._coords = np.array(location, dtype=np.float64)
        self._vel = np.array(velocity, dtype=np.float64)
        self._cost = (lambda theta, theta_z, theta_v, theta_zmin, delta_zmin:
                      - (C * np.cos(theta - theta_z) +
                         A * np.cos(theta - theta_v) -
                         S * np.cos(theta-theta_zmin)/delta_zmin ** 2))

    def step(self, dt):
        """
        Compute a timestep for a boid to update it's position using the Euler
        stepping method.

        Parameters
        ----------
        dt: float. The size of the timestep.
        """

        self._coords += dt * self._vel

    def steer(self, neighbours):
        """
        Compute a boid's new velocity due to the influence of the boid's
        neighbours in its immediate area.

        Parameters
        ----------
        neighbours: 1xN array. N is the number of neighbours, where the
            neighbours in the array should be Boid objects.
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
    Description of the class.
    """

    def __init__(self, locations, velocities, C=1, rl=1):
        """
        Create a Flock object containing the location and velocities of the
        boids making up a flock.

        Parameters
        ----------
        locations: N x 2 array. An array containing all of the (x, y) locations
            of each boid. N is the number of boids in the flock.
        velocities: N x 2 array. An array containing all of the (Vx, Vy)
            velocities of each boid. N is the number of boids in the flock.
        C: float. The Cohesion parameter which controls how much a boid will
            steer towards the center of mass of a flock.
        rl: float. The looking radius of a boid, i.e. how far it can see and
            be affected by it's neighbouring boids.
        """

        self._flockmate_locations = np.array(locations, dtype=np.float64)
        self._flockmate_velocities = np.array(velocities, dtype=np.float64)
        self._rl = rl  # rl is the looking radius
        self._boids = []
        for location, velocity, in zip(locations, velocities):
            self._boids.append(Boid(location, velocity, C))

    def step(self, dt):
        """
        Compute a timestep where each agent will have it's velocity and
        location updated due to the influence of it's neighbours.

        Parameters
        ----------
        dt: float. The length of a time step to update a boid's position.
        """

        # create list of current agents at time step
        current_boids = []
        for agent in self._boids:
            current_boids.append(deepcopy(agent))

        # change the heading of the boid i in list _boids, i.e. all the boids
        for i, boid in enumerate(self._boids):
            # create list of neighbouring boids
            boid_neighbours = []
            for j, another_boid in enumerate(current_boids):
                if i != j:  # when boid != another_boid
                    sep = boid._coords - another_boid._coords
                    dist = np.linalg.norm(sep)
                    # if the boid is in the looking radis of the other
                    # boid, then it will be influenced by this boid so add it
                    # to the list of the boid's neighbours
                    if dist < self._rl:
                        boid_neighbours.append(another_boid)

            # change the boid's heading due to the neighbours and compute
            # a timestep
            boid.steer(boid_neighbours)
            boid.step(dt)

    def locations(self):
        """
        Create an array of all the locations of the boids in the flock. This
        function requires there to be a list of boids defined in the flock.

        Returns
        -------
        An array containing all of the locations for each boid in the flock.
        """

        for i, boid in enumerate(self._boids):
            self._flockmate_locations[i, :] = boid._coords

        return self._flockmate_locations

    def average_location(self):
        """
        Compute the centre of the flock by calculating the average location
        of all the boids in a flock. Requires there to be a list of all
        the boid locations in the flock.

        Returns
        -------
        A 1 x 2 array cointaing the coordinates for the centre of the flock,
        using the average location of the boids.
        """

        return np.mean(self._flockmate_locations, axis=0)

    def average_width(self):
        """
        Compute the width of the flock by taking the average of the distance
        each boid is from the (average) centre of the flock.

        Returns
        -------
        A float value for the width of the flock.
        """

        sep = self._flockmate_locations - self.average_location()

        return np.mean(np.linalg.norm(sep, axis=1))


def train_flock(C, locations, velocities):
    """
    Trains a small flock to find the best cohesion parameter, i.e. the best
    value of C to minimise the variance in the width of the flock.

    Parameters
    ----------
    C: float. The cohesion parameter to optimise.
    locations: N x 2 array. The coordinate locations of the boids which are
        to be in the flock.
    velocities: N x 2 array. The velocity components of the boids which are
        to be in the flock.

    Returns
    -------
    variance: float. The variance of the width of the flock (the width as a
        function of time).
    """
    # create the flock
    flock = Flock(locations, velocities, C)

    # define simulation parameters
    n_boids = len(locations)
    n_steps = 50
    dt = 0.1

    # create arrays to store things
    locations = np.zeros((n_steps, n_boids, 2))
    average_width = np.zeros(n_steps)

    # set the first entries
    locations[0, :, :] = flock.locations()
    average_width[0] = flock.average_width()

    # evolve the flock
    for step in range(1, n_steps):
        flock.step(dt)
        locations[step, :, :] = flock.locations()
        average_width[step] = flock.average_width()

    variance = np.var(average_width)

    return variance


def evolve_flock(locations, velocities, n_steps, dt, C=1):
    """
    Create a flock of boids and evolve it in time using the Euler stepping
    method to update it's position.

    Parameters
    ----------
    locations: N x 2 array. The coordinate locations of the boids which are
        to be in the flock. N corresponds to the number of boids in the flock.
    velocities: N x 2 array. The velocity components of the boids which are
        to be in the flock. N corresponds to the number of boids in the flock.
    n_steps: integer. The number of steps to take to evolve the flock.
    dt: float. The length of the timestep for each step.
    C: float. The cohesion parameter for the flock. This controls how much a
        boid will want to steer towards the centre of the flock.

    Returns
    -------
    locations: n_steps x N x 2 array. The coordinate locations of the boids
        which have been evolved. N corresponds to the number of boids in the
        flock.
    average_width: n_steps x 1 array. The average width of the flock at each
        time step.
    time_steps: n_steps x 1 array. The corresponding value of time at the time
        steps.
    """

    # create the flock
    flock = Flock(locations, velocities, C)

    # define simulation parameters
    n_boids = len(locations)

    # create arrays to store things
    locations = np.zeros((n_steps, n_boids, 2))
    width = np.zeros(n_steps)
    time_steps = np.zeros(n_steps)

    # set the first entries
    locations[0, :, :] = flock.locations()
    width[0] = flock.average_width()

    # evolve the flock
    for step in range(1, n_steps):
        flock.step(dt)
        locations[step, :, :] = flock.locations()
        width[step] = flock.average_width()
        time_steps[step] = time_steps[step - 1] + dt

    return locations, width, time_steps


def plot_flock(x, width, t, extra_filename=''):
    """
    Plot the locations of the boids in a flock and the width of the flock as
    a function of time.

    Parameters
    ----------
    x: n_steps x N x 2 array. The locations of each boid in the flock at each
        time step, n_steps.
    width: n_steps x 1 array. The width of the flock at each time step, n_step.
    t: n_steps x 1 array. The value of time at each time step, n_step.
    extra_filename: string. Useful when plotting different things, but want to
        save the figure with a different name.

    Returns
    -------
    Two plots showing the initial positions of the boids, the postions of the
    boids as they evolved in time and the width of the flock as the flock
    evovled in time. The plots are output to the console as well as saved to
    the working directory.
    """

    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.plot(x[0, :, 0], x[0, :, 1], 'k*', label='Initial positions')
    ax1.plot(x[:, :, 0], x[:, :, 1])
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_title('Flock locations')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(t, width)
    ax2.set_xlabel('Time, $t$')
    ax2.set_ylabel('Average Width')
    ax2.set_title('Width of the flock')

    plt.savefig('flock_locations_width{}.pdf'.format(extra_filename))
    plt.show()


# =============================================================================
# Simulation Parameters & Train the Boids
# =============================================================================

# create the 4 training boids
boid_locations = np.array([[0.75, 0.75], [0.25, 0.25], [0.75, 0.25],
                           [0.25, 0.75]])
boid_velocities = np.ones_like(boid_locations)

# use minimize_scalar to minimize the output (the variance of the average
# distance) of the training fuction
min_c = minimize_scalar(train_flock, bounds=(0.1, 10),
                        args=(boid_locations, boid_velocities),
                        method='bounded')

# =============================================================================
# Apply the training to 50 random boids
# =============================================================================

n_boids = 50  # create 50 random boid locations and velocities
boid_locations = 5 * np.random.rand(n_boids, 2)
boid_velocities = np.ones_like(boid_locations) + 0.01 * \
    np.random.rand(n_boids, 2)

# call the function to create and evolve a flock
flock_locations, flock_width, t = evolve_flock(boid_locations, boid_velocities,
                                               200, 0.05, min_c.x)

# plot the flock and print the value of C used
plot_flock(flock_locations, flock_width, t, '_trained')
print('Optimum cohension value for minimum variance: {}.'.format(min_c.x))
