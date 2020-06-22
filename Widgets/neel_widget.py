#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 2020

@author: Brett
"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider
from scipy.optimize import fsolve
from scipy.constants import physical_constants as cst


### Set up figure
###______________________________________________________________

# Generate figure
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(15,11.25))
plt.subplots_adjust(bottom=0.32)
plt.style.use('dark_background')
#plt.style.use('lab')

# Label axes
ax1.set_xlabel(r'$M_{b}$ (kA m$^{-1}$)', fontsize=16)
ax1.set_ylabel(r'$M_{a}$ (kA m$^{-1}$)', fontsize=16)

ax2.set_xlabel('Temperature (K)', fontsize=16)
ax2.set_ylabel(r'Magnetization (kA m$^{-1}$)', fontsize=16)


### Constants
###______________________________________________________________

numpoints = 500           # Number of points used in equation solver

# Physical constants
mu0 = cst['permittivity of free space'][0]       # Permeability of free space
me = cst['electron mass'][0]     # Electron mass [kg]
h = 6.62607015e-34        # Planck constant [J-s]
hbar = h/(2*np.pi)        # Reduced Planck constant
e = 1.602176634e-19       # Elementary charge [C]
muB = 9.2740100783e-24    # Bohr Magneton
g = 2.00231930436256      # G-factor
kB = 1.380649e-23         # Boltzmann constant [J/K]

# Coupling Constants 
#lambda_aa = 735.84
#lambda_ab = 1100.3
#lambda_ba = 1100.3
#lambda_bb = 344.59

#Sublattice parameters
Na = 8.441e27             # Moments per unit volume in sublattice A
Nb = 12.66e27             # Moments per unit volume in sublattice B
Ja = 5/2                  # Total AM quantum number for sublattice A
Jb = 5/2                  # Total AM quantum number for sublattice B
ga = g                    # Lande g-factor for sublattice A  
gb = g                    # Lande g-factor for sublattice B

mua_max = ga*muB*Ja       # Maximum moment on sublattice A
Ma_max = Na*ga*muB*Ja     # Maximum moment on sublattice B
mub_max = gb* muB*Jb      # Maximum magnetization of sublattice A
Mb_max = Nb*gb*muB*Jb     # Maximum magnetization of sublattice B