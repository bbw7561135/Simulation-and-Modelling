# =============================================================================
# Test functions for flock_training
# =============================================================================

import pytest
import numpy as np
from flock_training import Boid, Flock


def test_stepping_function():
    """
    Tests the output from the Boid stepping function.
    """

    # create a test boid with some velocity
    test_boid = Boid([1, 1], [1, 0.5], 1)

    # test to make sure the stepping function is returning a ValueError when
    # it is given a negative or zero timestep.
    # the test will pass if a ValueError is raised
    with pytest.raises(ValueError):
        test_boid._step(-5)
        test_boid._step(0)

    # test that the stepping function will return the expected value given
    # the locations, velocities and a timestep.
    # expecting the locations to be incremented by dt * v as the new updated
    # locations are x + dt * v which will be x + 0.1 with dt = 0.1 and v = 1.
    # testing with different velocities in each component to make sure both
    # components are updated independentally
    test_boid._step(0.1)

    assert(test_boid._coords[0] == 1.1), \
        'Time step function for boid isn\'t working'
    assert(test_boid._coords[1] == 1.05), \
        'Time step function for boid isn\'t working'

    # check that a boid's position isn't changed if it's stationary
    stationary_boid = Boid([1, 1], [0, 0], 1)
    stationary_boid._step(1)
    assert(np.all(stationary_boid._coords == [1, 1])), \
        'The boid is moving when it has no velocity.'


def test_flock_positions():
    """
    Tests to see if the flock of boids is being set up correctly, i.e. that
    the location and velocities of the boids are in the correct order.
    """

    # check to see if the Flock is being set up correctly
    boid_locations = [[1, 1.6], [1.5, 11]]
    boid_velocities = [[1.111111, 333], [0, 23]]
    test_flock = Flock(boid_locations, boid_velocities)

    # expecting the first list to be the first row in the flock location
    # array and expecting the elements to not be muddled up.
    # check that the location of the boids are being appended to the flockmate
    # locations list correctly
    assert(np.all(test_flock._flockmate_locations[0, :] ==
                  boid_locations[0])), \
        'Flockmates being added to location array incorrectly'
    assert(np.all(test_flock._flockmate_locations[1, :] ==
                  boid_locations[1])), \
        'Flockmates being added to location array incorrectly'
    # check that the velocities of the boids are being added to the
    # flockmate velocities array correctly
    assert(np.all(test_flock._flockmate_velocities[0, :] ==
                  boid_velocities[0])), \
        'Flockmates being added to location array incorrectly'
    assert(np.all(test_flock._flockmate_velocities[1, :] ==
                  boid_velocities[1])), \
        'Flockmates being added to location array incorrectly'


def test_simple_neighbours():
    """
    Runs tests on simple flocks to test if the ouput for these flocks is as
    expected.
    """

    # create a flock where two boids can't see each other, so they should not
    # influence each other's paths and thus just travel in a straight line

    distant_boids = Flock([[0, 0], [2, 2]], [[1, 1], [1, 1]])
    distant_boids._step(0.1)
    assert(np.all(distant_boids._flockmate_locations[0, :] == 0.1))
    assert(np.all(distant_boids._flockmate_locations[1, :] == 2.1))

    # create a flock where two boids are very close, so they should stop moving
    # or move apart rapidly?
