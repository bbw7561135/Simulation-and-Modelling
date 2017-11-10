# =============================================================================
# Water molecule
# =============================================================================

import timeit
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import scipy as scp
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (9, 25)


# =============================================================================
# Functions
# =============================================================================


def molecule_positions(x):
    return V_func(0, 0, 0, x[0], x[1], 0, x[2], x[3], 0)


def fO(rX, rY, rZ):
    return np.array(F_O(rX[0], rY[0], rZ[0], rX[1], rY[1], rZ[1], rX[2],
                        rY[2], rZ[2]))


def fH1(rX, rY, rZ):
    return np.array(F_H1(rX[0], rY[0], rZ[0], rX[1], rY[1], rZ[1], rX[2],
                         rY[2], rZ[2]))


def fH2(rX, rY, rZ):
    return np.array(F_H2(rX[0], rY[0], rZ[0], rX[1], rY[1], rZ[1], rX[2],
                         rY[2], rZ[2]))


def acceleration(x, mass):

    a = np.zeros_like(x)

    # calculate acceleration for oxygen
    a[0, :] = fO(x[:, 0], x[:, 1], x[:, 2]) / mass[0]
    # calculate acceleration for hydrogen atoms
    a[1, :] = fH1(x[:, 0], x[:, 1], x[:, 2]) / mass[1]
    a[2, :] = fH2(x[:, 0], x[:, 1], x[:, 2]) / mass[1]

    return a


def verlet_int(x, v, a, mass, dt):

    xnew = x + dt * v + 0.5 * dt ** 2 * a
    vstar = v + 0.5 * dt * a
    anew = acceleration(xnew, mass)
    vnew = vstar + 0.5 * dt * anew

    return xnew, vnew, anew


# =============================================================================
# Construct the potential and force functions using SymPy
# =============================================================================

start = timeit.default_timer()

sp_x_H1, sp_y_H1, sp_z_H1 = sp.symbols('x_{H_1}, y_{H_1}, z_{H_1}')
sp_x_H2, sp_y_H2, sp_z_H2 = sp.symbols('x_{H_2}, y_{H_2}, z_{H_2}')
sp_x_O, sp_y_O, sp_z_O = sp.symbols('x_{O}, y_{O}, z_{O}')

sp_r_OH1 = sp.sqrt((sp_x_O - sp_x_H1) ** 2 + (sp_y_O - sp_y_H1) ** 2 +
                   (sp_z_O - sp_z_H1) ** 2)
sp_r_OH2 = sp.sqrt((sp_x_O - sp_x_H2) ** 2 + (sp_y_O - sp_y_H2) ** 2 +
                   (sp_z_O - sp_z_H2) ** 2)
sp_r_HH = sp.sqrt((sp_x_H1 - sp_x_H2) ** 2 + (sp_y_H1 - sp_y_H2) ** 2 +
                  (sp_z_H1 - sp_z_H2) ** 2)

# equilibrium distances
sp_r_OHeq = 1
sp_r_HHeq = 1.633

# define the separation vectors
sp_dr_HH = sp_r_HH - sp_r_HHeq
sp_dr_OH1 = sp_r_OH1 - sp_r_OHeq
sp_dr_OH2 = sp_r_OH2 - sp_r_OHeq

# define some constants
sp_D0 = 101.9188
sp_alpha = 2.567
sp_k_theta = 328.645606
sp_k_rtheta = -211.4672
sp_k_rr = 111.70765

# construct the sympy expressions for the potential terms
V1 = sp_D0 * (1 - sp.exp(sp_alpha * sp_dr_OH1)) ** 2 + \
                sp_D0 * (1 - sp.exp(sp_alpha * sp_dr_OH2)) ** 2
V2 = (1/2) * sp_k_theta * sp_dr_HH ** 2
V3 = sp_k_rtheta * sp_dr_HH * (sp_dr_OH1 + sp_dr_OH2)
V4 = sp_k_rr * sp_dr_OH1 * sp_dr_OH2

V = V1 + V2 + V3 + V4
V_func = sp.utilities.lambdify((sp_x_O, sp_y_O, sp_z_O, sp_x_H1, sp_y_H1,
                                sp_z_H1, sp_x_H2, sp_y_H2, sp_z_H2), V,
                               modules='numpy')

# differentiate the potential function wrt different parameters
sp_F_O = (-V.diff(sp_x_O), -V.diff(sp_y_O), -V.diff(sp_z_O))
sp_F_H1 = (-V.diff(sp_x_H1), -V.diff(sp_y_H1), -V.diff(sp_z_H1))
sp_F_H2 = (-V.diff(sp_x_H2), -V.diff(sp_y_H2), -V.diff(sp_z_H2))

# construct functions for the differentials
F_O = sp.utilities.lambdify((sp_x_O, sp_y_O, sp_z_O, sp_x_H1, sp_y_H1, sp_z_H1,
                             sp_x_H2, sp_y_H2, sp_z_H2), sp_F_O,
                            modules='numpy')
F_H1 = sp.utilities.lambdify((sp_x_O, sp_y_O, sp_z_O, sp_x_H1, sp_y_H1,
                              sp_z_H1, sp_x_H2, sp_y_H2, sp_z_H2), sp_F_H1,
                             modules='numpy')
F_H2 = sp.utilities.lambdify((sp_x_O, sp_y_O, sp_z_O, sp_x_H1, sp_y_H1,
                              sp_z_H1, sp_x_H2, sp_y_H2, sp_z_H2), sp_F_H2,
                             modules='numpy')

# =============================================================================
# Set up molecules and run the code
# =============================================================================

# find the minimum of the potential function
minimised = scp.optimize.minimize(molecule_positions, [0.8, 0.6, -0.8,
                                                       0.6], tol=1e-4)

molecules = np.array([[0, 0, 0], [minimised.x[0], minimised.x[1], 0],
                      [minimised.x[2], minimised.x[3], 0]])
v = np.zeros_like(molecules)
a = np.zeros_like(molecules)
masses = [15.999, 1.008]
n_steps = 1000
dt = 0.001
t = np.zeros((n_steps+1, 1))

positions = np.zeros((n_steps+1, molecules.shape[0], molecules.shape[1]))
positions[0, :, :] = molecules.copy()

# interate over the steps
for step in range(n_steps):
    t[step+1] = t[step] + dt
    molecules, v, a = verlet_int(molecules, v, a, masses, dt)
    positions[step+1, :, :] = molecules.copy()

# =============================================================================
# Plotting
# =============================================================================

fig = plt.figure()

ax1 = fig.add_subplot(311)
ax1.plot(t, positions[:, 0, 0] - positions[0, 0, 0], label='x')
ax1.plot(t, positions[:, 0, 1] - positions[0, 0, 1], label='y')
ax1.plot(t, positions[:, 0, 2] - positions[0, 0, 2], label='z')
ax1.set_xlim(0, 1)
ax1.set_xlabel('Time, t')
ax1.set_ylabel('Change in location for Oxygen atom')
ax1.legend(fancybox=True, framealpha=0.2)

ax2 = fig.add_subplot(312)
ax2.plot(t, positions[:, 1, 0] - positions[0, 1, 0], label='x')
ax2.plot(t, positions[:, 1, 1] - positions[0, 1, 1], label='y')
ax2.plot(t, positions[:, 1, 2] - positions[0, 1, 2], label='z')
ax2.set_xlim(0, 1)
ax2.set_xlabel('Time, t')
ax2.set_ylabel('Change in location for a Hydrogen atom 1')
ax2.legend(fancybox=True, framealpha=0.2)

ax3 = fig.add_subplot(313)
ax3.plot(t, positions[:, 2, 0] - positions[0, 2, 0], label='x')
ax3.plot(t, positions[:, 2, 1] - positions[0, 2, 1], label='y')
ax3.plot(t, positions[:, 2, 2] - positions[0, 2, 2], label='z')
ax3.set_xlim(0, 1)
ax3.set_xlabel('Time, t')
ax3.set_ylabel('Change in location for a Hydrogen atom 2')
ax3.legend(fancybox=True, framealpha=0.2)

fig.tight_layout()
plt.savefig('positions_of_atoms.pdf')

stop = timeit.default_timer()

# =============================================================================
# Printing
# =============================================================================

print("Run time: {} seconds.".format(stop - start))
print(minimised.x)