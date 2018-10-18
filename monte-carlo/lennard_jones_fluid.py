import datetime
import numpy as np
from numba import jit
from matplotlib import pyplot as plt

# Define Global constants
n_particles = 50
n_steps = 50000
epsilon = 1.0
sigma = 1.0
delta = 0.04
density = 1.0
L = (n_particles / density) ** (1/3)
rc = L / 2
dL = delta * L
temp = 2.0
beta = 1 / temp
out_freq = n_steps / 10


def compute_lj_potential(rsq, rcsq):
    """
    Return the Lennard Jones potential for a given r squared value and r cutoff value
    """
    LJ_U = np.where(rsq < rcsq, 4 * epsilon * ((sigma ** 2 / rsq) ** 6 - (sigma ** 2 / rsq) ** 3), 0)

    return LJ_U


def compute_lj_force(rsq, rcsq):
    """
    Return the Lennard Jones force for a given r square value and r cutoff value
    """

    LJ_F = np.where(rsq < rcsq, 24 * (2 / rsq ** 7 - 1 / rsq ** 4), 0)

    return LJ_F

def dx_particles(particles, pIdx):
    """
    Compute the separation of all particles for particle p
    """

    dx = particles - particles[pIdx, :]
    # As we are using periodic boundary conditions, if any particles |dx| > L/2, there is a closer mirror particle
    # np.sign returns an indication if a number if negative or not, i.e. if dx < 0, then np.sign(dx) == -1
    dx[np.abs(dx) > L/2] -= np.sign(dx[np.abs(dx) > L/2]) * L

    return dx


def move_rand_particle(particles, lj_pot):
    """
    Given the potential between particles, move the particles
    """

    # Make a copy of arrays because Python
    particles_new = particles.copy()
    lj_pot_new = lj_pot.copy()

    # Pick a random particle and move it slightly
    pIdx = np.random.randint(n_particles)
    particles_new[pIdx, :] += dL * np.random.rand(3)
    dx = dx_particles(particles_new, pIdx)

    dU = 0.0
    for i in range(pIdx):
        rsq = np.dot(dx[i, :], dx[i, :])
        lj_pot_new[i, pIdx] = compute_lj_potential(rsq, rc ** 2)
        dU = lj_pot_new[i, pIdx] - lj_pot[i, pIdx]
    for i in range(pIdx + 1, n_particles):
        rsq = np.dot(dx[i, :], dx[i, :])
        lj_pot_new[i, pIdx] = compute_lj_potential(rsq, rc ** 2)
        dU = lj_pot_new[i, pIdx] - lj_pot[i, pIdx]

    return particles_new, lj_pot_new, dU


def compute_pressure(particles):
    """
    Compute the pressure of the current particle configuration
    """

    pressure_tail = 16 / 3 * np.pi * density ** 2 * (2 - rc ** (-9) / 3 - rc ** (-3))
    pressure = pressure_tail + density / beta
    for i in range(n_particles):
        dx = dx_particles(particles, i)
        for j in range(i + 1, n_particles):
            rsq = np.dot(dx[j, :], dx[j, :])
            pressure += compute_lj_force(rsq, rc ** 2) / L ** 3

    return pressure


def initial_energy(particles, lj_pot, energy):
    """
    Calculate the initial potential energy of the system
    """

    for i in range(n_particles):
        dx = dx_particles(particles, i)
        for j in range(i):
            rsq = np.dot(dx[j, :], dx[j, :])
            lj_pot[j, i] = compute_lj_potential(rsq, rc ** 2)
        for j in range(i + 1, n_particles):
            rsq = np.dot(dx[j, :], dx[j, :])
            lj_pot[j, i] = compute_lj_potential(rsq, rc ** 2)

    energy[0] = np.sum(lj_pot)

    return particles, lj_pot, energy


@jit  # Let's jit this shit
def mcmc(particles, lj_pot, energy):
    """
    Monte Carlo Markov Chain iterations using the Metropolis Hastings algorithm
    """

    for step in range(1, n_steps + 1):
        particles_new, lj_pot_new, dU = move_rand_particle(particles, lj_pot)
        if dU < 0.0 or np.random.rand() < np.exp(-beta * dU):
            particles = particles_new
            lj_pot = lj_pot_new
            energy[step] = energy[step - 1] + dU
        else:
            energy[step] = energy[step - 1]

        if step % out_freq == 0:
            print("{} steps out of {} completed ({}%)".format(step, n_steps, step / n_steps * 100))

    return particles, lj_pot, energy


def main():
    """
    The main steering function of the script
    """

    print("\nBeginning simulation: current date and time {}\n".format(datetime.datetime.now()))

    # Initialise the particles, potential and energy array
    particles = np.random.rand(n_particles, 3) * L
    lj_pot = np.zeros((n_particles, n_particles))
    energy = np.zeros(n_steps + 1)

    # Calculate the initial energies and then do the MCMC iterations and *hopefully* converge
    particles, lj_pot, energy = initial_energy(particles, lj_pot, energy)
    particles, lj_pot, energy = mcmc(particles, lj_pot, energy)
    pressure = compute_pressure(particles)

    return particles, lj_pot, energy, pressure


if __name__ == "__main__":
    n_steps = 50000
    n_particles = 100

    paticles, lj_pot, energy, pressure = main()

    # Output some "useful" data
    energy_min = np.min(energy)
    energy_max = np.max(energy)

    print("\n----------\n")
    print("Minimum energy value:      {:1.4e}".format(energy_min))
    print("Maximum energy value:      {:1.4e}".format(energy_max))
    print("Energy of final system:    {:1.4e}".format(energy[-1]))
    print("Pressure exerted on box:   {:1.4e}".format(pressure))
    print("\n----------\n")

    # Plot the energy against time step
    fig, ax = plt.subplots()
    steps = np.arange(0, n_steps)
    energy_plot = energy - energy_min
    ax.semilogy(steps, energy_plot[1:])
    ax.set_xlabel("Step")
    ax.set_ylabel(r"Normalised Energy, $E - E_{min}$")
    ax.set_xlim(0, n_steps)
    plt.show()
