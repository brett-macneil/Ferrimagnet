#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 2020

@author: Brett
"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider
from scipy.optimize import fsolve
#import time


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
H = 0                     # Set H=0 for simpicity later

# Physical constants
mu0 = 4*np.pi*1e-7        # Permeability of free space
me = 9.1093837015e-31     # Electron mass [kg]
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
    diff = np.sign(z2-z1)
    c = plt.contour(diff)
    data = c.allsegs[0][0]
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


def get_mag(T_min, T_max, numpoints, lam, H, kilo=True):
    #t_start = time.time()
    
    Tvec = np.linspace(T_min, T_max, numpoints)
    Ma = np.empty(numpoints)
    Mb = np.empty(numpoints)
    guess = [-Ma_max, Mb_max] # Initial guess
    
    for i in range(numpoints):
        ma, mb = fsolve(equations, x0=guess, args=(lam, Tvec[i], H))
        Ma[i] = ma; Mb[i] = mb
        guess = [ma, mb]
        
    #t_end = time.time()
    #print('Get {:.3f} seconds'.format(t_end-t_start))
    
    if kilo == True:
        Ma /= 1e3; Mb /= 1e3
        
    return (Tvec, Ma, Mb)


### Sliders
###______________________________________________________________

# Coupling constants
lam_aa_loc = plt.axes([0.125, 0.22, 0.775, 0.03])
lam_aa_init = 0.
lam_aa_max = 1000.
lam_aa_min = 0.
lam_aa_sl = Slider(lam_aa_loc, label=r'$\lambda_{aa}$', valmin=lam_aa_min, \
                   valmax=lam_aa_max, valinit=lam_aa_init)
lam_aa_sl.label.set_size(16)

lam_bb_loc = plt.axes([0.125, 0.17, 0.775, 0.03])
lam_bb_init = 0.
lam_bb_max = 1000.
lam_bb_min = 0.
lam_bb_sl = Slider(lam_bb_loc, label=r'$\lambda_{bb}$', valmin=lam_bb_min, \
                   valmax=lam_bb_max, valinit=lam_bb_init)
lam_bb_sl.label.set_size(16)

lam_ab_loc = plt.axes([0.125, 0.12, 0.775, 0.03])
lam_ab_init = 500.
lam_ab_max = 1000.
lam_ab_min = 0.
lam_ab_sl = Slider(lam_ab_loc, label=r'$\lambda_{ab}$', valmin=lam_ab_min, \
                   valmax=lam_ab_max, valinit=lam_ab_init)
lam_ab_sl.label.set_size(16)

lam_ba_loc = plt.axes([0.125, 0.07, 0.775, 0.03])
lam_ba_init = 500.
lam_ba_max = 1000.
lam_ba_min = 0.
lam_ba_sl = Slider(lam_ba_loc, label=r'$\lambda_{ba}$', valmin=lam_ba_min, \
                   valmax=lam_ba_max, valinit=lam_ba_init)
lam_ba_sl.label.set_size(16)


# Temperature
T_loc = plt.axes([0.125, 0.02, 0.775, 0.03])
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

Ma_surf = mag_eq_a(Ma_grid, Mb_grid, lam_aa_init, lam_ab_init, T_init, H)
Mb_surf = mag_eq_b(Ma_grid, Mb_grid, lam_bb_init, lam_ba_init, T_init, H)
a_self_x, a_self_y = get_intersect(Ma_grid, Ma_surf)
b_self_x, b_self_y = get_intersect(Mb_grid, Mb_surf)

Ma_plot1, = ax1.plot(a_self_x, a_self_y, color='cyan')
Mb_plot1, = ax1.plot(b_self_x, b_self_y, color='orange')

# Magnetization-temperature subplot (Right, axis 2)
lam_init = [lam_aa_init, lam_bb_init, lam_ab_init, lam_ba_init]
Temp_vec, Mag_a, Mag_b = get_mag(T_min, T_max, numpoints, lam_init, H=0)

Ma_plot2, = ax2.plot(Temp_vec, Mag_a, color='cyan')
Mb_plot2, = ax2.plot(Temp_vec, Mag_b, color='orange')
Mtot_plot2, = ax2.plot(Temp_vec, Mag_a+Mag_b, color='white', ls='dotted')
Mag_min = min( min(Mag_a), min(Mag_b) )
Mag_max = max( max(Mag_a), max(Mag_b) )

Temp_line, = ax2.plot([T_init,T_init], [Mag_min, Mag_max], color='red')

ax1.legend([r'Sublattice a', 'Sublattice b'], loc=1, fontsize=16)
ax2.legend([r'Sublattice a', 'Sublattice b', 'Total'], loc=1, fontsize=16)

### Updates
###______________________________________________________________

def update(val):
    # Pull val from sliders
    lam_aa_new = lam_aa_sl.val
    lam_bb_new = lam_bb_sl.val
    lam_ab_new = lam_ab_sl.val
    lam_ba_new = lam_ba_sl.val
    
    T_new = T_sl.val
    
    # Update axis 1
    Ma_surf_new = mag_eq_a(Ma_grid, Mb_grid, lam_aa_new, lam_ab_new, T_new, H)
    Mb_surf_new = mag_eq_b(Ma_grid, Mb_grid, lam_bb_new, lam_ba_new, T_new, H)
    a_self_x_new, a_self_y_new = get_intersect(Ma_grid, Ma_surf_new)
    b_self_x_new, b_self_y_new = get_intersect(Mb_grid, Mb_surf_new)
    
    Ma_plot1.set_xdata(a_self_x_new)
    Ma_plot1.set_ydata(a_self_y_new)
    Mb_plot1.set_xdata(b_self_x_new)
    Mb_plot1.set_ydata(b_self_y_new)
    
    # Update axis 2
    lam_new = [lam_aa_new, lam_bb_new, lam_ab_new, lam_ba_new]
    _, Mag_a_new, Mag_b_new = get_mag(T_min, T_max, numpoints, lam_new, H=0)
    Mag_min_new = min( min(Mag_a_new), min(Mag_b_new) )
    Mag_max_new = max( max(Mag_a_new), max(Mag_b_new) )
    Ma_plot2.set_ydata(Mag_a_new)
    Mb_plot2.set_ydata(Mag_b_new)
    Mtot_plot2.set_ydata(Mag_a_new+Mag_b_new)
    Temp_line.set_xdata([T_new,T_new])
    Temp_line.set_ydata([Mag_min_new, Mag_max_new])
    ax2.set_ylim([Mag_min_new*1.1, Mag_max_new*1.1])
    
    return None
    

lam_aa_sl.on_changed(update)
lam_bb_sl.on_changed(update)
lam_ab_sl.on_changed(update)
lam_ba_sl.on_changed(update)
T_sl.on_changed(update)
fig.canvas.draw_idle()
fig.show()