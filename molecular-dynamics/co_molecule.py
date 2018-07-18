# =============================================================================
# Simulation of a CO molecule
# =============================================================================

import numpy as np
import sympy
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12, 6)

sympy.init_printing()

# =============================================================================
# Simulation parameters
# =============================================================================

k = 2743  # Harmonic force constant
r_eq = 1.1283  # Equilibrium distance
D0 = 258.9  # Dissociation energy
alpha = 2.302  # Morse parameter

delta_t = 0.001
n_steps = 1000

# =============================================================================
# Functions
# =============================================================================


def acceleration_calc(positions, mass, potential):
    """
    Calculate the acceleration due to a force.
    """

    n_particles = len(positions)
    accel = np.zeros_like(positions)

    for i in range(n_particles):
        for j in range(i+1, n_particles):
            r = positions[i, :] - positions[j, :]
            r_sq = np.dot(r, r)
            dphi = potential(D0, alpha, np.sqrt(r_sq), r_eq)
            accel[i, :] += dphi * r / mass[i]
            accel[j, :] -= dphi * r / mass[j]

    return accel


def time_step(vel, accel, positions, masses, potential):
    """
    Compute a time step for the system. Returns the new velocities, positions
    and accelerations.
    """

    positions_new = positions + delta_t * vel + 0.5 * delta_t ** 2 * accel
    vel_star = vel + 0.5 * delta_t * accel
    accel_new = acceleration_calc(positions_new, masses, potential)
    vel_new = vel_star + 0.5 * delta_t * accel_new

    return positions_new, accel_new, vel_new


# =============================================================================
# Morse Potential
# =============================================================================

E_Morse, sp_r, sp_r_eq, sp_D0, sp_alpha = sympy.symbols(
        'E_M, r, r_eq, D_0, alpha')
E_Morse = sp_D0 * (1 - sympy.exp(-sp_alpha * (sp_r - sp_r_eq))) ** 2
F_Morse = -sympy.diff(E_Morse, sp_r).simplify()
f_morse_func = sympy.utilities.lambdify((sp_D0, sp_alpha, sp_r, sp_r_eq),
                                        F_Morse, modules='numpy')

particles = np.array([[0, 0, 0], [r_eq+10e-5, 0, 0]])
masses = np.array([1, 1.332029])
velocity = np.zeros_like(particles)
acceleration = acceleration_calc(particles, masses, f_morse_func)

t = np.linspace(0, (n_steps + 1/2) * delta_t, n_steps + 1)
positions_array_morse = np.zeros((n_steps+1, particles.shape[0],
                                  particles.shape[1]))
positions_array_morse[0, :, :] = particles.copy()

for i in range(n_steps):
    particles, acceleration, velocity = time_step(velocity, acceleration,
                                                  particles, masses,
                                                  f_morse_func)
    positions_array_morse[i+1, :, :] = particles.copy()

fig = plt.figure()
plt.plot(t, positions_array_morse[:, 0, 0] - positions_array_morse[0, 0, 0],
         label='Morse x disp for C')
plt.plot(t, positions_array_morse[:, 1, 0] - positions_array_morse[0, 1, 0],
         label='Morse x disp for O')
plt.xlabel('Time, t'), plt.ylabel('Displacement, r')
plt.legend()
plt.savefig('co_molecule_morse.pdf')
plt.show()

# =============================================================================
# Kratzer Potential
# =============================================================================

E_Kratzer, sp_r, sp_r_eq, sp_D0 = sympy.symbols('E_K, r, r_eq, D_0')
E_Kratzer = sp_D0 * ((sp_r - sp_r_eq)/sp_r) ** 2
F_Kratzer = -sympy.diff(E_Kratzer, sp_r).simplify()
f_kratzer_func = sympy.utilities.lambdify((sp_D0, sp_alpha, sp_r, sp_r_eq),
                                          F_Kratzer, modules='numpy')

particles = np.array([[0, 0, 0], [r_eq+10e-5, 0, 0]])
masses = np.array([1, 1.332029])
velocity = np.zeros_like(particles)
acceleration = acceleration_calc(particles, masses, f_kratzer_func)

t = np.linspace(0, (n_steps + 1/2) * delta_t, n_steps + 1)
positions_array_kratzer = np.zeros((n_steps+1, particles.shape[0],
                                   particles.shape[1]))
positions_array_kratzer[0, :, :] = particles.copy()

for i in range(n_steps):
    particles, acceleration, velocity = time_step(velocity, acceleration,
                                                  particles, masses,
                                                  f_kratzer_func)
    positions_array_kratzer[i+1, :, :] = particles.copy()
fig = plt.figure()
plt.plot(t, positions_array_kratzer[:, 0, 0] - positions_array_kratzer[0, 0,
         0], '--', label='Kratzer x disp for C')
plt.plot(t, positions_array_kratzer[:, 1, 0] - positions_array_kratzer[0, 1,
         0], '--', label='Kratzer x disp for O')
plt.xlabel('Time, t'), plt.ylabel('Displacement, r')
plt.legend()
plt.savefig('co_molecule_kratzer.pdf')
plt.show()

# =============================================================================
# Harmonic Potential
# =============================================================================

#E_Harmonic, sp_r, sp_k, sp_r_eq = sympy.symbols('E_K, r, k, r_eq')
#E_Harmonic = 0.5 * sp_k * (sp_r - sp_r_eq) ** 2
#F_Harmonic = -sympy.diff(E_Harmonic, sp_r).simplify()
#f_harmonic_func = sympy.utilities.lambdify((sp_D0, sp_alpha, sp_r, sp_r_eq),
#                                           F_Harmonic, modules='numpy')
#
#particles = np.array([[0, 0, 0], [r_eq+10e-5, 0, 0]])
#masses = np.array([1, 1.332029])
#velocity = np.zeros_like(particles)
#acceleration = acceleration_calc(particles, masses, f_harmonic_func)
#
#t = np.linspace(0, (n_steps + 1/2) * delta_t, n_steps + 1)
#positions_array_harmonic = np.zeros((n_steps+1, particles.shape[0],
#                                    particles.shape[1]))
#positions_array_harmonic[0, :, :] = particles.copy()
#
#for i in range(n_steps):
#    particles, acceleration, velocity = time_step(velocity, acceleration,
#                                                  particles, masses,
#                                                  f_harmonic_func)
#    positions_array_harmonic[i+1, :, :] = particles.copy()
#
#fig = plt.figure()
#plt.plot(t, positions_array_kratzer[:, 0, 0] - positions_array_kratzer[0, 0,
#         0], '--', label='Harmonic x disp for C')
#plt.plot(t, positions_array_kratzer[:, 1, 0] - positions_array_kratzer[0, 1,
#         0], '--', label='Harmonic x disp for O')
#plt.xlabel('Time, t'), plt.ylabel('Displacement, r')
#plt.legend()
#plt.savefig('co_molecule_harmonic.pdf')
#plt.show()

# =============================================================================
# Plot potentials together
# =============================================================================

fig = plt.figure()
plt.plot(t, positions_array_morse[:, 0, 0] - positions_array_morse[0, 0, 0],
         label='Morse x disp for C')
plt.plot(t, positions_array_morse[:, 1, 0] - positions_array_morse[0, 1, 0],
         label='Morse x disp for O')
plt.plot(t, positions_array_kratzer[:, 0, 0] - positions_array_kratzer[0, 0,
         0], '--', label='Kratzer x disp for C')
plt.plot(t, positions_array_kratzer[:, 1, 0] - positions_array_kratzer[0, 1,
         0], '--', label='Kratzer x disp for O')
plt.xlabel('Time, t'), plt.ylabel('Displacement, r')
plt.legend()
plt.savefig('combined_potentials.pdf')
plt.show()
