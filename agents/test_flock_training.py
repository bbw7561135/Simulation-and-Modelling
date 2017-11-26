# =============================================================================
# Test functions for flock_training
# =============================================================================

import pytest
import numpy as np
from copy import deepcopy
from flock_training import Boid, Flock


def test_boid_stepping_function():
    """
    Tests that the function _step in the Boid class is working correctly.
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


def test_flock_positions_and_velocities():
    """
    Tests to see if the flock of boids is being set up correctly, i.e. that
    the location and velocities of the boids are in the correct order. Tests
    the _flockmate_locations and _flockmate_velocities functions in the Flock
    class.
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


def test_flock_step():
    """
    Test the flock stepping function _step in the Flock class.
    """

    # create flock where two boids will see each other
    boid_locations = [[1, 1.6], [1.5, 1]]
    boid_velocities = [[1.3, 1.2], [1.8, 0.4]]
    test_flock = Flock(boid_locations, boid_velocities)

    # the two boids in the above flock should see each other and so should
    # evolve. Thus, calling the Flock _step function on them should see that
    # their positions and velocities change
    int_locs = deepcopy(test_flock._flockmate_locations)
    int_vels = deepcopy(test_flock._flockmate_velocities)
    test_flock._step(0.01)
    assert(np.any(test_flock._flockmate_locations != int_locs)), \
        'The flock should have changed, but it hasn\'t.'
    assert(np.any(test_flock._flockmate_velocities != int_vels)), \
        'The flock should have changed, but it hasn\'t.'

    # test flock where two boids can't see one another
    boid_locations = [[1, 1.6], [3.4, 3.7]]
    boid_velocities = [[3.3, 3.2], [1, 1]]
    test_flock = Flock(boid_locations, boid_velocities)

    # the two boids can't see each other as they are outside the looking radius
    # thus their direction should remain unaffected and they will travel in a
    # straight line
    new_location = deepcopy(test_flock._flockmate_locations) + 0.1 * \
        deepcopy(test_flock._flockmate_velocities)
    int_vels = deepcopy(test_flock._flockmate_velocities)
    test_flock._step(0.1)

    assert(np.all(test_flock._flockmate_locations == new_location)), \
        'The boids should move in a straight line, but haven\t.'
    assert(np.all(test_flock._flockmate_velocities == int_vels)), \
        'The boid\'s velocities have changed even though they shouldn\'t.'


def test_small_flocks():
    """
    Runs tests on small flocks to test if the ouput for these flocks is as
    expected. These tests will test the _step function in the Flock class,
    where the flocks are generated and the neighbours are found. The function
    _steer_boid for Boid is also included in the Flock _step function and thus
    this test function will also test that the _steer function is functioning
    as expected.
    """

    # create a flock where two boids can't see each other, so they should not
    # influence each other's paths and thus just travel in a straight line
    distant_boids = Flock([[0, 0], [2, 2]], [[1, 1], [1, 1]])
    distant_boids._step(0.1)
    assert(np.all(distant_boids._flockmate_locations[0, :] == 0.1))
    assert(np.all(distant_boids._flockmate_locations[1, :] == 2.1))

    # create a flock where two boids are very close. This means that the first
    # boid should steer away from boid 2, i.e. it's velocity should be less
    # than (or) zero as it has turned around from its original velocity.
    # Boid 2's velocity should remain in the same direction, has boid 1 has
    # already turned around
    close_boids = Flock([[0, 0], [0.001, 0.0]], [[1, 0], [1, 0]])
    close_boids._step(0.1)
    assert(np.all(close_boids._flockmate_locations[0, :] <= 0))

    # create a flock where one boid is moving very fast. This boid should move
    # fast enough in the timestep that it is outside of the viewing radius of
    # the boid before and after and so the slow boid will not be influenced by
    # the fast boid as it doesn't have enough time (or timestep resolution) to
    # react to the fast boid
    fast_boid = Flock([[0, 0], [1.5, 0.0]], [[1, 0], [100, 0]])
    int_locs = fast_boid._flockmate_locations
    int_vels = fast_boid._flockmate_velocities
    new_locs = int_locs + 0.1 * int_vels
    fast_boid._step(0.1)
    assert(np.all(fast_boid._flockmate_locations == new_locs)), \
        'Slow boid should have been unaffected by fast boid.'
    assert(np.all(fast_boid._flockmate_velocities == int_vels)), \
        'The slow and fast boid velocities should be unaffected.'

    # create a flock of stationary boids and evolve for one timestep. The boids
    # shouldn't move even if the boid has neighbours.
    stationary_flock = Flock([[1, 1], [1.5, 1.5]], [[0, 0], [0, 0]])
    int_locs = stationary_flock._flockmate_locations
    stationary_flock._step(0.1)
    assert(np.all(stationary_flock._flockmate_locations == int_locs)), \
        'The boids have moved even when they have no velocity.'


def test_flock_average_location():
    """
    Tests the _flock_average_location function in the Flock class.
    """

    # put two boids on top of each other. Not realistic at all, but expecting
    # the function to return the position which both boids occupy. In practice
    # the boids shouldn't end up in this situation, as they should steer away
    # from each other, but it's a simple test either way.
    pauli_violated = Flock([[1, 1], [1, 1]], [[1, 1], [1, 1]])
    assert(np.all(pauli_violated._flock_average_location() == 1))

    # test the mean function for a bigger and more realistic flock. The value
    # of the class function should match the values of x_avg and y_avg which
    # are calculated below
    flock_locations = np.array([[1, 1], [1.3, 1.5], [2, 1.3], [0.4, 0.9],
                               [5, 2]])
    flock_vels = np.ones(5)
    test_flock = Flock(flock_locations, flock_vels)
    x_avg = np.sum(flock_locations[:, 0]) / 5
    y_avg = np.sum(flock_locations[:, 1]) / 5
    assert(np.all(test_flock._flock_average_location() ==
                  np.hstack((x_avg, y_avg)))), \
        'Average of the flock locations is being calculated incorrectly.'


def test_flock_width():
    """
    Tests the _flock_width function in the Flock class.
    """

    # create a flock
    flock_locations = np.array([[1, 1], [1.3, 1.5], [2, 1.3], [0.4, 0.9],
                               [5, 2]])
    flock_vels = np.ones(5)
    test_flock = Flock(flock_locations, flock_vels)
    # calculate the average location of the flock
    average_location = (np.sum(flock_locations[:, 0]) / 5,
                        np.sum(flock_locations[:, 1]) / 5)
    # calculate a the separation of the flockmates from the average location
    # of all the boids in a flock and then calculate the euclidian distance
    # from the average location of the boids
    test_flock_sep = np.abs(test_flock._flockmate_locations -
                            average_location)
    test_flock_r = np.sqrt(np.sum(test_flock_sep ** 2, axis=1))
    # the average width is then the average of these distances
    flock_width = np.mean(test_flock_r)
    # expecting a value of around 1.3 for flock_width. This is due to the
    # average location of the flock being (1.94, 1.34). Thus when the locations
    # are subtracted and then the mean of those values are found, the value of
    # around 1.3 appears

    # now test the hand calculated values against the function
    assert(np.all(flock_width == test_flock._flock_width()))
