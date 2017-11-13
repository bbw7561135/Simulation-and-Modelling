# =============================================================================
# 1 dimensional finite element solver for computing the temperature of a sytem
# given a force vector F(x).
# =============================================================================

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12, 6)

# =============================================================================
#
# =============================================================================

# define the interval, number of elements and the notes
interval = [0, 1]
N_Elements = 4
x, h = np.linspace(interval[0], interval[1], N_Elements, retstep=True)

# set up location matrix
LM = np.zeros((2, N_Elements))
for e in range(N_Elements):
    LM[0, e] = e      # element node
    LM[1, e] = e + 1  # element right hand node

LM[1, -1] = -1  # right hand node of the final element is not considered due
                # to the boundary conditions

# create arrays for the global stiffness matrix and force vector
K_global = np.zeros((N_Elements, N_Elements))
F_global = np.zeros(N_Elements)

for a in range(N_Elements):
    for b in range(N_Elements):
        K_global[a, b] += ((-1) ** (a + b)) / (interval[1] - interval[0])
