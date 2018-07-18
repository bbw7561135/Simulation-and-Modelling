# molecular-dynamics

## co_molecule

Simulates the interactions of the atoms in a CO molecule using the Morse and the Kratzer potential. The code needs to be generalised further and then have a harmonic potential implemented correctly.

## molecular_dynamics_basics

Simulation of particles interacting through the Lennard-Jones potential. The particles are contained inside a box with periodic boundary conditions and are evolved in time using a Verlet integrator.

## n_water_molecules

A generalisation of the water_molecule code for multiple water molecules. The molecules are contained in a box with periodic boundary conditions where they experience an internal and external force. The internal force between the atoms in a molecule is similar to the strong nuclear force potential. The external force is between all atoms, except atoms of the same molecule, and is a combination of a Coulomb and Lennard-Jones interaction between the particles.
The molecules are evolved in time using a Verlet integrator.

## water_molecule

Simulation of a water molecule using a potential between the three atoms in the molecule. The potential is similar to the strong nuclear force potential and is used to generate a force function using Sympy to symbolically manipulate and differentiate the potential function.
