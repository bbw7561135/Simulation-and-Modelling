# agents
## bird_flock
Simulates bird's flocking together. The code achieves this by minimising the
negative of a utility function to compute a new heading and velocity for a
bird due to the flockmates it can see.
## conway_game_of_life
Creates an animation of Conway's Game of Life. Currently there are only two
masks to use in the grid. More will eventually be implemented when I have
the time to do it.
## flock_training
Based on the bird_flock code. This code uses a blackbox minimiser to find the
optimum value of the cohesion parameter *C* for a flock of birds described by
a utility function. The optimum value of C is found by training a flock of 4
boids. The variance in the average width of the flock is then minimised by
finding the optimum value of C. This value of C is then used to simulate a flock
of 50 randomly placed boids.
## test_flock_training
Pytest style unit tests for the functions in flock_training.
## traffic_flow
WIP: Simulates the flow of traffic on a multiple-lane road.

# assignment-notebooks
## Coursework 01 MC
The submitted Jupyter notebook for the Monte Carlo coursework.
## Coursework 01 MD
The submitted Jupter notebook for the Molecular Dynamics coursework.

# finite-elements
## 1d-finite-element-solver
Computes the temperature of a 1D system using finite element methods.
## 2d-finite-element-solver
Computes the temperature of a 2D system using finite element methods given the
node locations, IEN and ID arrays for a mesh and heat source function.
## big_g
An application of 2d-finite-element-solver to solve for the temperature of a
g shaped grid. The boundary conditions of the grid set the right side of the
grid to a fixed temperature of T = 0. At all other boundaries, the normal
derivative of the temperature vanishes.

# molecular-dynamics
## co_molecule
Simulates the interactions of the atoms in a CO molecule using the Morse and the
Kratzer potential. The code needs to be generalised further and then have a
harmonic potential implemented correctly.
## molecular_dynamics_basics
Simulation of particles interacting through the Lennard-Jones
potential. The particles are contained inside a box with periodic boundary
conditions and are evolved in time using a Verlet integrator.
## n_water_molecules
A generalisation of the water_molecule code for multiple water molecules. The
molecules are contained in a box with periodic boundary conditions where they
experience an internal and external force. The internal force between the atoms
in a molecule is similar to the strong nuclear force potential. The external
force is between all atoms, except atoms of the same molecule, and is a
combination of a Coulomb and Lennard-Jones interaction between the particles.
The molecules are evolved in time using a Verlet integrator.
## water_molecule
Simulation of a water molecule using a potential between the three atoms in the
molecule. The potential is similar to the strong nuclear force potential and is
used to generate a force function using Sympy to symbolically manipulate and
differentiate the potential function.

# monte-carlo
## ising_model
Simulation of a system of spins using the Ising model. The system is a 2D square
lattice evolved in time by using the Metropolis-Hasting algorithm to randomly
sample the system using Monte Carlo techniques. The code attempts to find an
equilibrium state of the system at varying grid sizes and temperatures before
it calculates the final magnetisation of the system.
## lennard_jones_fluid
WIP: Simulates a gas using the Lennard-Jones potential. Calculates the pressure
of the gas after being evolved in time in the NVT ensemble.
## mc_int
An n-dimensional generalisation of mc_int_circle_sphere to calculate the volume
of an n-dimensional hypersphere using Monte Carlo integration.
## mc_int_circle_sphere
Calculates the area and volume of a sphere using Monte Carlo integration.
## metropolis-hastings
An application of the Metropolis-Hasting algorithm to calculate the ground
state energy of particles in a box.

# stochastic-DEs
## brownian-motion
Evaluates a function with a Brownian noise term over N different paths and M
different realisations, where N, M = [500, 1000, 2000].
