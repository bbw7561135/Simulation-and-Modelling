# Agents

## bird_flock

Simulates bird's flocking together. The code achieves this by minimising the negative of a utility function to compute a new heading and velocity for a bird due to the flockmates it can see.

## conway_game_of_life

Creates an animation of Conway's Game of Life. Currently there are only two masks to use in the grid. More will eventually be implemented when I have the time to do it.

## flock_training

Based on the bird_flock code. This code uses a blackbox minimiser to find the optimum value of the cohesion parameter *C* for a flock of birds described by a utility function. The optimum value of C is found by training a flock of 4 boids. The variance in the average width of the flock is then minimised by finding the optimum value of C. This value of C is then used to simulate a flock of 50 randomly placed boids.

## test_flock_training

Pytest style unit tests for the functions in flock_training.

## traffic_flow

WIP: Simulates the flow of traffic on a multiple-lane road.
