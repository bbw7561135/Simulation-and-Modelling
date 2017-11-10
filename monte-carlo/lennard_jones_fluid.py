# =========================================================================== #
#  Markov Chain Monte Carlo using the Metropolis Hastings algorithm           #
# to simulate a Lennard-Jones fluid in the NVT ensemble.                      #
#                                                                             #
# Written by Edward Parkinson                                                 #
# Email: e.j.parkinson@soton.ac.uk                                            #
# =========================================================================== #

import matplotlib.pyplot as plt
import numpy as np


# =========================================================================== #
#                           Simulation Parameters                             #
# =========================================================================== #


n_steps = 10000
n_particles = 100
a = 1
density = a/10
temperature = 2
beta = 1/temperature
pressure_term = density/beta
epsilon = 1
sigma = 1
L = (n_particles/density) ** (1/3)
volume = L ** 3
halfL = L/2
r_c = halfL ** 2


# =========================================================================== #
#                               Functions                                     #
# =========================================================================== #


def LJ_potential(r_sq, r_c):
    """
    Returns the value of the Lennard-Jones potential
    for values of r. Returns 0 for cut off value r > r_c (minimum image
    convention). Epsilon and sigma are scaling parameters.
    Uses (1/r) ** 6 and (1/r) ** 3 as r is squared already.
    """

    U = np.where(r_sq <= r_c, 4 * epsilon * ((sigma/r_sq) ** (6) -
                                             (sigma/r_sq) ** (3)), 0 * r_sq)

    return U


def LJ_force(r_sq, r_c):
    """
    Calculate the force between two particles. Returns F if r_sq < r_c and
    will return 0 otherwise.
    """

    F = np.where(r_sq < r_c, 24 * (2 * (1/r_sq ** 7) - (1/r_sq ** 4)) *
                 np.sqrt(r_sq), 0 * r_sq)

    return F


def LJ_potential_test():
    """
    Function to test that the Lennard-Jones potential is being calculated
    correctly. Returns nothing if no errors are encountered.
    """

    # test that for r < 1, U >= 0
    x = np.linspace(0.8, 1.0)
    assert(np.all(LJ_potential(x, 1)) >= 0)

    # test that for r > 1, U <= 0
    x = np.linspace(1, 2, 100)
    assert(np.all(LJ_potential(x, 2.1) <= 0))

    # test that at r = 1, U(r) = 0
    assert(np.allclose(LJ_potential(1, 1), 0))

    # test that the cutoff is working correctly. If r > r_c, U will return
    # 0. Thus, expecting 0 from this
    assert(np.allclose(LJ_potential(0.75, 0.5), 0))

    # test that it is negative for a point, r = 1.4
    assert(LJ_potential(1.4, 1.6) < 0)

    # test that it is postive for a point, r = 0.4
    assert(LJ_potential(0.4, 1.2) > 0)


def plot_potential():
    """
    Plots the Lennard-Jones potential with no scaling factors to show
    the shape of the potential. Run by using plot_potential(LJ_potential).
    """
    x = np.linspace(0.5, 2, 200)
    y = LJ_potential(x, 2)

    plt.figure(figsize=((15, 8)))
    plt.xlim((0.8, 2)), plt.ylim((-2, 2))
    plt.xlabel('Radius, r'), plt.ylabel('U(r)')
    plt.grid(True, which='both')
    plt.plot(x, y, 'r-')
    plt.show()


def separation(target, none_target):
    """
    Calculate the value the separation, i.e r ** 2 = dx ** 2 + dy ** 2 +
    dz ** 2,  between two particles.
    """

    # size = 3 used for total system energy calculation
    if np.size(none_target) == 3:
        difference = none_target - target
        differece_sq = np.square(difference)
        r_squared = np.sum(differece_sq)
    # used for particle - system of particle
    else:
        difference = none_target - target  # line below removes the zero entry
        difference_no_zero = difference[np.all(difference != 0, axis=1)]
        differece_sq = np.square(difference_no_zero)
        r_squared = np.sum(differece_sq, axis=1)

#    r_squared[np.abs(r_squared) > r_c] -= np.sign(r_squared[np.abs(r_squared) > r_c]) * L

    return r_squared


def total_system_energy_force(system):
    """
    Calculate the total energy of the system by looking at the iteractions
    each particle has with the other particles.
    """

    energy = 0
#    force = 0
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            r_sq = separation(system[i], system[j])
            energy += LJ_potential(r_sq, r_c)
#            force += np.dot(LJ_force(r_sq, r_c), np.sqrt(r_sq))

    return energy#, force


def pair_energy_force(particle1, particle2):
    """
    Calculate the energy between two particles.
    """

    r_sq = separation(particle1, particle2)
    energy = np.sum(LJ_potential(r_sq, r_c))
#    force = np.sum(LJ_force(r_sq, r_c))

    return energy#, force


def periodic_boundary(particle):
    """
    Apply periodic boundary conditions to a particle
    """

    periodic_particle = np.where(particle > L, particle - L, particle)
    periodic_particle = np.where(particle < 0, particle + L, particle)

    return periodic_particle


def probabilty_check(E_1, E_2):
    """
    Calculates the Boltzmann probabilty for a transition from E1 to E2
    """

    probabilty_check = np.exp(-beta * (E_2 - E_1))
    MH_check = np.random.uniform(0, 1)

    if MH_check < probabilty_check:
        return 1


def energy_tail_correction():
    """
    Compute the energy tail correction when r > r_c.
    """

    correction = ((8 * np.pi * density)/3) * ((1/3) * (1/r_c ** 4.5) -
                                              (1/r_c ** 1.5))

    return correction


def pressure_tail_correction():
    """
    Compute the pressure tail corretion when r > r_c.
    """

    correction = ((16 * np.pi * density ** 2)/3) * ((2/3) * (1/r_c ** 4.5) -
                                                    (1/r_c ** 1.5))

    return correction


def plot_energy(energy_array):
    """
    Plots the energy of the system. The function will only graph only
    values from the graph_from variable onwards in the array.
    """

    plot_from = 5000

    plt.figure(figsize=(15, 8))
    plt.plot(energy_array[plot_from:, 0], energy_array[plot_from:, 1], 'r-')
#    plt.semilogy(energy_array[plot_from:, 1], 'r-')
    plt.xlabel('Number of steps, n_steps'), plt.ylabel('Energy, U')
    plt.savefig('Lennard-Johnson_Energy.pdf')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.hist(energy_array[plot_from:, 1], bins=50, edgecolor='black',
             linewidth=1.0, normed=True)
    plt.xlabel('Energy, U')
    plt.ylabel('Frequency')
    plt.savefig('Lennard-Johnson_Energy_histogram.pdf')
    plt.show()


# =========================================================================== #
#                            Monte Carlo Sampling                             #
# =========================================================================== #


# test that the function LJ_potential() is still working correctly
LJ_potential_test()

# initialise the box of particles
particles = L * np.random.rand(n_particles, 3)
system_energy, system_force = total_system_energy_force(particles)
system_pressure = pressure_term + system_force/volume
energy_correction = energy_tail_correction()
pressure_correction = pressure_tail_correction()

output_array = np.zeros((n_steps+1, 3))
output_array[0][0] = 0
output_array[0][1] = system_energy + energy_correction
output_array[0][2] = system_pressure + pressure_correction

n = 0
accept = 0
for step in range(n_steps):

    n += 1

    # random variables for the target particle to perturb
    target_index = np.random.randint(n_particles)
    rand_direction = np.random.randint(3)
    rand_pert = np.random.uniform(-1, 1)

    target_particle = np.copy(particles[target_index])
    initial_energy, initial_force = pair_energy_force(
            target_particle, particles)
    target_particle[rand_direction] += rand_pert
    target_particle = periodic_boundary(target_particle)
    perturbed_energy, perturbed_force = pair_energy_force(
            target_particle, particles)

    if perturbed_energy < initial_energy:
        particles[target_index] = target_particle
        system_energy += (perturbed_energy - initial_energy)
        system_force += (perturbed_force - initial_force)
    else:
        transition_probabilty = probabilty_check(initial_energy,
                                                 perturbed_energy)
        if transition_probabilty == 1:
            particles[target_index] = target_particle
            system_energy += (perturbed_energy - initial_energy)
            system_force += (perturbed_force - initial_force)
            accept += 1

    system_pressure = pressure_term + system_force/volume

    energy_correction = energy_tail_correction()
    pressure_correction = pressure_tail_correction()

    output_array[step+1][0] = n
    output_array[step+1][1] = system_energy + energy_correction
    output_array[step+1][2] = system_pressure + pressure_correction

plot_energy(output_array)
print('Acceptance rate: {}%'.format((accept/n_steps) * 100))
print('Final pressure: {}'.format(output_array[n_steps][2]))
