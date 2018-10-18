# stochastic-DEs

## brownian_motion

Evaluates a function with a Brownian noise term over N different paths and M different realisations, where N, M = [500, 1000, 2000].

## EM_Mil

Evaluates an SDE using the Eular-Maruyama and Milstein methods. The code compares both solutions against the exact solution on the same plot and also compares the code execution time.

## ornstein_uhlenbeck_convergence

The Ornstein-Uhlenbeck equation is solved using the Euler-Maruyama method and the theta method. The weak convergence of each method is tested. The methods are tested by plotting a histogram of the results at one point for many realisations. The late time solution is compared to the expected late solution. The E-M method is expected to have a weak convergence of 1 (this  result can be improved by increasing the amount of realisations) and the Theta method is expected to have no convergence. Both the histogram and late time result solutions show that both methods are sampling correctly and  computing the expected solution.

## OU

Evaluates the Ornstein-Uhlenbeck equation using the Euler-Maruyama method for single and multiple realisiations.

## test_SDE

Pytest style unit tests for ornstein_uhlenbeck_convergence.
