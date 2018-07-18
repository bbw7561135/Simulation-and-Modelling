# finite-elements

## 1d_finite_element_solver

Computes the temperature of a 1D system using finite element methods.

## 2d_finite_element_solver

Computes the temperature of a 2D system using finite element methods given the node locations, IEN and ID arrays for a mesh and heat source function.

## big_g

An application of 2d-finite-element-solver to solve for the temperature of a g shaped grid. The boundary conditions of the grid set the right side of the grid to a fixed temperature of T = 0. At all other boundaries, the normal derivative of the temperature vanishes.

## test_big_g
Pytest style tests for big_g.
