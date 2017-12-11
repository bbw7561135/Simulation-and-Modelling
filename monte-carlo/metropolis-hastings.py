# =========================================================================== #
# MCMC Metropolis Hastings algorithm to calculate the ground state energy     #
# of a system of n-particles in a box.                                        #
# =========================================================================== #

import matplotlib.pyplot as plt
import numpy as np


def E(n):
    """
    Compute the energy of a system given the quantum numbers n.

    Parameters
    ----------
    n: n_particles x 3 array of ints.
        The value of the quantum numbers describing the system.

    Returns
    -------
    E: float.
        The total energy of the system.
    """

    h = 2 * np.pi
    E = h ** 2/(8 * np.pi) * np.sum(n ** 2)
    
    return E


def metropolis_hastings(states, n_steps, n_particles=100, beta=0.1):
    """
    Applies the Metropolis-Hastings algorithm to evolve a box towards its 
    ground state energy.

    Parameters
    ----------
    states: n_particles x 3 array of ints.
        The quantum numbers for each particle contained in the box.
    n_steps: int:
        The number of steps to evolve the box of particles for.
    n_particles: int.
        The number of particles contained in the box.
    beta: float.
        The value of the inverse temperature of the box.

    Returns
    -------
    energy_output: n_steps x 2 array of floats.
        Column 0 contains the energy of the sytem at each time step and column 
        2 contains the value of time at that timestep.
    """

    spin_change = [-1, 1]
    energy_output = np.zeros((n_steps, 2))

    n = 0
    for i in range(n_steps):

        n += 1

        # generate the random change variables
        random_particle = np.random.randint(n_particles)
        random_direction = np.random.randint(3)
        random_change = spin_change[np.random.randint(2)]

        # calculate the energy of the initial state
        E_i = E(states)
        # make copy of initial states to compare
        changed_states = states.copy()

        # change a random state
        changed_states[random_particle][random_direction] += random_change

        if changed_states[random_particle][random_direction] < 1:
            changed_states[random_particle][random_direction] = \
                states[random_particle][random_direction]
            # n can't be less than 1, so go back to original state

        E_j = E(changed_states)  # calculate the energy of the new states

        if E_j < E_i:
            # accept E_j value
            energy = E_j
            states = changed_states
        else:
            # calculate the probabilty for the transistion to happen
            prob_accept = np.exp(-1 * beta * (E_j - E_i))
            # pull random number and compare to prob_accept
            rand_num_accept = np.random.rand(1)
            if rand_num_accept <= prob_accept:
                # accept E_j
                energy = E_j
                states = changed_states
            else:
                # reject E_j
                energy = E_i

        energy_output[i][0], energy_output[i][1] = energy, n

    return energy_output

n_particles = 100
n_steps = 50000

states = np.random.randint(low=1, high=20, size=(n_particles, 3))
x = metropolis_hastings(states, n_steps, n_particles=n_particles)

plt.figure(figsize=(15, 8))
plt.plot(x[:, 1], x[:, 0], 'k-', linewidth=0.1)
plt.xlabel('Number of steps, n_steps')
plt.ylabel('Energy of system, E(n)')
plt.yscale('log')
plt.savefig('mcmc_100n.pdf')
plt.show()

plt.hist(x[30000:, 0], bins=50, edgecolor='black', linewidth=1.0, normed=True)
plt.ylabel('Frequency')
plt.xlabel('Energy, E')
plt.savefig('mcmc_100n_hist.pdf')
plt.show()
