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
mu0 = cst['vacuum mag. permeability'][0]      # Permeability of free space
me = cst['electron mass'][0]                  # Electron mass [kg]
h = cst['planck constant'][0]                 # Planck constant [J-s]
hbar = cst['reduced Planck constant'][0]      # Reduced Planck constant
e = cst['atomic unit of charge'][0]           # Elementary charge [C]
muB = cst['Bohr magneton'][0]                 # Bohr Magneton
g = -cst['electron g factor'][0]              # G-factor
kB = cst['Boltzmann constant'][0]             # Boltzmann constant [J/K]

#Sublattice parameters
Na = 8.441e27             # Moments per unit volume in sublattice A
Nb = 12.66e27             # Moments per unit volume in sublattice B
Ja = 5/2                  # Total AM quantum number for sublattice A
Jb = 5/2                  # Total AM quantum number for sublattice B
ga = g                    # Lande g-factor for sublattice A  
gb = g                    # Lande g-factor for sublattice B

mua_max = ga*muB*Ja       # Maximum moment on sublattice A
Ma_max = Na*ga*muB*Ja     # Maximum magnetization of sublattice A
mub_max = gb* muB*Jb      # Maximum moment on sublattice B
Mb_max = Nb*gb*muB*Jb     # Maximum magnetization of sublattice B


### Function definitions
###______________________________________________________________

def coth(x):
    return 1/np.tanh(x)


def brillouin(y, J):
    eps = 1e-3 # should be small
    y = np.array(y); B = np.empty(y.shape)
    m = np.abs(y)>=eps # mask for selecting elements 
                       # y[m] is data where |y|>=eps;
                       # y[~m] is data where |y|<eps;
    
    B[m] = (2*J+1)/(2*J)*coth((2*J+1)*y[m]/(2*J)) - coth(y[m]/(2*J))/(2*J)
    
    # First order approximation for small |y|<eps
    # Approximation avoids divergence at origin
    B[~m] = ((2*J+1)**2/J**2/12-1/J**2/12)*y[~m]
    
    return B


def get_intersect(z1, z2):
    diff = np.sign(z2-z1) # array with distinct boundary
                          # this boundary between -1's and 1's
                          # is the intersection curve of the two 
                          # surfaces z1 and z2
                          
    c = plt.contour(diff) 
    data = c.allsegs[0][0] # intersection contour
                           # Return (x,y) of intersect curve
    return ( data[:, 0], data[:, 1] )


def mag_eq_a(Ma, Mb, lambda_aa, lambda_ab, T, H):
    arg = mu0 * mua_max * (H - lambda_aa * Ma - lambda_ab * Mb) / (kB*T)
    return Ma_max * brillouin(arg, Ja)


def mag_eq_b(Ma, Mb, lambda_bb, lambda_ba, T, H):
    arg = mu0 * mua_max * (H - lambda_ba * Ma - lambda_bb * Mb) / (kB*T)
    return Mb_max * brillouin(arg, Jb)


def equations(mags, lam, T, H):
    Ma, Mb = mags
    lambda_aa, lambda_bb, lambda_ab, lambda_ba = lam
    eq1 = mag_eq_a(Ma, Mb, lambda_aa, lambda_ab, T, H) - Ma
    eq2 = mag_eq_b(Ma, Mb, lambda_bb, lambda_ba, T, H) - Mb
    return (eq1, eq2)


def get_mag(T_min, T_max, numpoints, lam, H):
    Tvec = np.linspace(T_min, T_max, numpoints)
    Ma = np.empty(numpoints)
    Mb = np.empty(numpoints)
    guess = [-Ma_max, Mb_max] # Initial guess
    
    for i in range(numpoints):
        ma, mb = fsolve(equations, x0=guess, args=(lam, Tvec[i], H))
        Ma[i] = ma; Mb[i] = mb # Update solution
        guess = [ma, mb]       # Update guess to last solution
        
    return (Tvec, Ma, Mb)