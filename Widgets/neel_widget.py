#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 2020
Last updated Thu Jun 25 2020

@author: Brett MacNeil
"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider
from scipy.optimize import fsolve
from scipy.constants import physical_constants as cst


### Set up figure
###______________________________________________________________

# Generate figure
fig, [ax1, ax2] = plt.subplots(1, 2)
a = plt.axes([0,0,0,0])
plt.subplots_adjust(bottom=0.4)
plt.style.use('dark_background')
#plt.style.use('lab')

# Label axes
ax1.set_xlabel(r'$M_{b}$ (kA m$^{-1}$)', fontsize=16)
ax1.set_ylabel(r'$M_{a}$ (kA m$^{-1}$)', fontsize=16)
ax1.grid(color='grey', linestyle='dotted', linewidth=1)

ax2.set_xlabel('Temperature (K)', fontsize=16)
ax2.set_ylabel(r'Magnetization (kA m$^{-1}$)', fontsize=16)
ax2.grid(color='grey', linestyle='dotted', linewidth=1)


### Constants
###______________________________________________________________

numpoints = 500           # Number of points used in equation solver

# Physical constants
mu0 = cst['vacuum mag. permeability'][0]      # Permeability of free space
me = cst['electron mass'][0]                  # Electron mass [kg]
h = cst['Planck constant'][0]                 # Planck constant [J-s]
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


def get_intersect(z1, z2, x, y):
    diff = np.sign(z2-z1) # array with distinct boundary
                          # this boundary between -1's and 1's
                          # is the intersection curve of the two 
                          # surfaces z1 and z2
                                     
    c = a.contour(x, y, diff)
    a.cla() # Clear dummy axes
    
    data = c.allsegs[0][0] # intersection contour
    x = data[:, 0]
    y = data[:, 1]
                           # Return (x,y) of intersect curve              
    return (x, y)


def mag_eq_a(Ma, Mb, lam_aa, lam_ab, T, H):
    arg = mu0 * mua_max * (H - lam_aa * Ma - lam_ab * Mb) / (kB*T)
    return Ma_max * brillouin(arg, Ja)


def mag_eq_b(Ma, Mb, lam_bb, lam_ba, T, H):
    arg = mu0 * mua_max * (H - lam_ba * Ma - lam_bb * Mb) / (kB*T)
    return Mb_max * brillouin(arg, Jb)


def equations(mags, lam, T, H):
    Ma, Mb = mags
    lam_aa, lam_bb, lam_ab, lam_ba = lam
    eq1 = mag_eq_a(Ma, Mb, lam_aa, lam_ab, T, H) - Ma
    eq2 = mag_eq_b(Ma, Mb, lam_bb, lam_ba, T, H) - Mb
    return (eq1, eq2)


def get_mag(T_min, T_max, numpoints, lam, H):
    Tvec = np.linspace(T_min, T_max, numpoints)
    Ma = np.empty(numpoints)
    Mb = np.empty(numpoints)
    guess = [-Mb_max, Ma_max] # Initial guess
    
    for i in range(numpoints):
        mb, ma = fsolve(equations, x0=guess, args=(lam, Tvec[i], H))
        Ma[i] = ma; Mb[i] = mb # Update solution
        guess = [mb, ma]       # Update guess to last solution
        
    return (Tvec, Ma, Mb)


### Sliders
###______________________________________________________________

# External field 
H_loc = plt.axes([0.125, 0.30, 0.775, 0.03])
H_init = 0.
H_max = 10.   # mu0 H
H_min = -10.  # mu0 H
H_sl = Slider(H_loc, label=r'$\mu_0 H$ (T)', valmin=H_min, valmax=H_max, \
              valinit=H_init)
H_sl.label.set_size(16)

# Coupling constants
lam_aa_loc = plt.axes([0.125, 0.25, 0.775, 0.03])
lam_aa_init = 0.
lam_aa_max = 1000.
lam_aa_min = 0.
lam_aa_sl = Slider(lam_aa_loc, label=r'$\lambda_{aa}$', valmin=lam_aa_min, \
                   valmax=lam_aa_max, valinit=lam_aa_init)
lam_aa_sl.label.set_size(16)

lam_bb_loc = plt.axes([0.125, 0.20, 0.775, 0.03])
lam_bb_init = 0.
lam_bb_max = 1000.
lam_bb_min = 0.
lam_bb_sl = Slider(lam_bb_loc, label=r'$\lambda_{bb}$', valmin=lam_bb_min, \
                   valmax=lam_bb_max, valinit=lam_bb_init)
lam_bb_sl.label.set_size(16)

lam_ab_loc = plt.axes([0.125, 0.15, 0.775, 0.03])
lam_ab_init = 500.
lam_ab_max = 1000.
lam_ab_min = 0.
lam_ab_sl = Slider(lam_ab_loc, label=r'$\lambda_{ab}$', valmin=lam_ab_min, \
                   valmax=lam_ab_max, valinit=lam_ab_init)
lam_ab_sl.label.set_size(16)

lam_ba_loc = plt.axes([0.125, 0.10, 0.775, 0.03])
lam_ba_init = 500.
lam_ba_max = 1000.
lam_ba_min = 0.
lam_ba_sl = Slider(lam_ba_loc, label=r'$\lambda_{ba}$', valmin=lam_ba_min, \
                   valmax=lam_ba_max, valinit=lam_ba_init)
lam_ba_sl.label.set_size(16)

# Temperature
T_loc = plt.axes([0.125, 0.05, 0.775, 0.03])
T_init = 300.
T_max = 600.
T_min = 1.
T_sl = Slider(T_loc, label=r'$T$ (K)', valmin=T_min, valmax=T_max, \
              valinit=T_init)
T_sl.label.set_size(16)


### Plots
###______________________________________________________________

# Self-consistent subplot (Left, axis 1)
Ma_scale = np.linspace(-Ma_max, Ma_max, numpoints)
Mb_scale = np.linspace(-Mb_max, Mb_max, numpoints)

Ma_grid, Mb_grid = np.meshgrid(Ma_scale, Mb_scale)

# Brillouin function surface
Ma_surf = mag_eq_a(Ma_grid, Mb_grid, lam_aa_init, lam_ab_init, T_init, H_init)
Mb_surf = mag_eq_b(Ma_grid, Mb_grid, lam_bb_init, lam_ba_init, T_init, H_init)

# Self-consistent solutions
# Intersect of Brillouin surfaces and Ma or Mb plane
Ma_x, Ma_y = get_intersect(Ma_grid, Ma_surf, Ma_grid, Mb_grid)
Mb_x, Mb_y = get_intersect(Mb_grid, Mb_surf, Ma_grid, Mb_grid)
Ma_x /= 1e3
Mb_x /= 1e3
Ma_y /= 1e3
Mb_y /= 1e3

Ma_plot1, = ax1.plot(Ma_x, Ma_y, color='cyan')
Mb_plot1, = ax1.plot(Mb_x, Mb_y, color='orange')

# Magnetization-temperature subplot (Right, axis 2)
# Magnetization is divided by 1000 to match plotting units of kA/m
lam_init = [lam_aa_init, lam_bb_init, lam_ab_init, lam_ba_init]
Temp_vec, Mag_a, Mag_b = get_mag(T_min, T_max, numpoints, lam_init, H_init)
Mag_a /= 1e3
Mag_b /= 1e3

Ma_plot2, = ax2.plot(Temp_vec, Mag_a, color='cyan')
Mb_plot2, = ax2.plot(Temp_vec, Mag_b, color='orange')
Mtot_plot2, = ax2.plot(Temp_vec, (Mag_a+Mag_b), color='white', linewidth=3)
Mag_min = min( min(Mag_a), min(Mag_b) )
Mag_max = max( max(Mag_a), max(Mag_b) )

Temp_line, = ax2.plot([T_init,T_init], [Mag_min, Mag_max], color='red')

ax1.legend([r'Sublattice a', 'Sublattice b'], loc=1, fontsize=16)
ax2.legend([r'Sublattice a', 'Sublattice b', 'Total'], loc=1, fontsize=16)


### Updates
###______________________________________________________________

def update(val):
    # Pull val from sliders
    H_new = H_sl.val/mu0
    
    lam_aa_new = lam_aa_sl.val
    lam_bb_new = lam_bb_sl.val
    lam_ab_new = lam_ab_sl.val
    lam_ba_new = lam_ba_sl.val
    
    T_new = T_sl.val
    
    # Update axis 1
    Ma_surf_new = mag_eq_a(Ma_grid, Mb_grid, lam_aa_new, lam_ab_new, T_new, \
                           H_new)
    Mb_surf_new = mag_eq_b(Ma_grid, Mb_grid, lam_bb_new, lam_ba_new, T_new, \
                           H_new)
    Ma_x_new, Ma_y_new = get_intersect(Ma_grid, Ma_surf_new, Ma_grid, Mb_grid)
    Mb_x_new, Mb_y_new = get_intersect(Mb_grid, Mb_surf_new, Ma_grid, Mb_grid)
    Ma_x_new /= 1e3
    Mb_x_new /= 1e3
    Ma_y_new /= 1e3
    Mb_y_new /= 1e3
    
    Ma_plot1.set_xdata(Ma_x_new)
    Ma_plot1.set_ydata(Ma_y_new)
    Mb_plot1.set_xdata(Mb_x_new)
    Mb_plot1.set_ydata(Mb_y_new)
    
    # Update axis 2
    lam_new = [lam_aa_new, lam_bb_new, lam_ab_new, lam_ba_new]
    _, Mag_a_new, Mag_b_new = get_mag(T_min, T_max, numpoints, lam_new, H_new)
    Mag_a_new /= 1e3
    Mag_b_new /= 1e3
    Mag_min_new = min( min(Mag_a_new), min(Mag_b_new) )
    Mag_max_new = max( max(Mag_a_new), max(Mag_b_new) )
    
    Ma_plot2.set_ydata(Mag_a_new)
    Mb_plot2.set_ydata(Mag_b_new)
    Mtot_plot2.set_ydata((Mag_a_new+Mag_b_new))
    Temp_line.set_xdata([T_new,T_new])
    Temp_line.set_ydata([Mag_min_new, Mag_max_new])
    ax2.set_ylim([Mag_min_new*1.1, Mag_max_new*1.1])
    
    return None
    

H_sl.on_changed(update)
lam_aa_sl.on_changed(update)
lam_bb_sl.on_changed(update)
lam_ab_sl.on_changed(update)
lam_ba_sl.on_changed(update)
T_sl.on_changed(update)
fig.canvas.draw_idle()
fig.show()