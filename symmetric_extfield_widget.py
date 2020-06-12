#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 2020

@author: Brett
"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider, Button
from scipy.optimize import fsolve
#import time


### Set up figure
###______________________________________________________________

# Generate figure
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(15,11.25))
plt.subplots_adjust(bottom=0.25)
plt.style.use('dark_background')
plt.style.use('lab')

# Label axes
ax1.set_xlabel(r'$M_{b}$ (kA m$^{-1}$)', fontsize=16)
ax1.set_ylabel(r'$M_{a}$ (kA m$^{-1}$)', fontsize=16)

ax2.set_xlabel('Temperature (K)', fontsize=16)
ax2.set_ylabel(r'Magnetization (kA m$^{-1}$)', fontsize=16)


### Constants
###______________________________________________________________

numpoints = 75           # Number of points used in equation solver

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
    eps = 0.1
    y = np.array(y); B = np.empty(y.shape)
    m = np.abs(y)>=eps # mask for selecting elements 
                       # y[m] is data where |y|>=eps;
                       # y[~m] is data where |y|<eps;
    
    B[m] = (2*J+1)/(2*J)*coth((2*J+1)*y[m]/(2*J)) - coth(y[m]/(2*J))/(2*J)
    
    # First order approximation for small |y|<eps
    # Approximation avoids divergence at origin
    B[~m] = ((2*J+1)**2/J**2/12-1/J**2/12)*y[~m]
    
    return B


def mag_eq_a(Mb, lambda_ab, T, H):
    return Ma_max * brillouin(mu0*mua_max*(H - lambda_ab*Mb) / (kB*T), Ja)


def mag_eq_b(Ma, lambda_ab, T, H):
    return Mb_max * brillouin(mu0*mub_max*(H - lambda_ab*Ma) / (kB*T), Jb)


def equations(mags, lambda_ab, T, H):
    Ma, Mb = mags
    eq1 = mag_eq_a(Mb, lambda_ab, T, H) - Ma
    eq2 = mag_eq_b(Ma, lambda_ab, T, H) - Mb
    return (eq1, eq2)


def get_mag(T_min, T_max, numpoints, lambda_ab, H, kilo=True, bfield=True):
    #t_start = time.time()
    
    if bfield == True: 
        H /= mu0
    
    Tvec = np.linspace(T_min, T_max, numpoints)
    Ma = np.empty(numpoints)
    Mb = np.empty(numpoints)
    guess = [-Ma_max, Mb_max] # Initial guess
    
    for i in range(numpoints):
        ma, mb = fsolve(equations, x0=guess, args=(lambda_ab, Tvec[i], H))
        Ma[i] = ma; Mb[i] = mb
        guess = [ma, mb]
        
    #t_end = time.time()
    #print('Get {:.3f} seconds'.format(t_end-t_start))
    
    if kilo == True:
        Ma /= 1e3; Mb /= 1e3
        
    return (Tvec, Ma, Mb)


### Sliders and buttons
###______________________________________________________________

# Coupling constant
lam_loc = plt.axes([0.125, 0.10, 0.775, 0.03])
lam_ab_init = 500.
lam_ab_max = 800.
lam_ab_min = 200.
lam_ab_sl = Slider(lam_loc, label=r'$\lambda_{ab}$', valmin=lam_ab_min, \
                   valmax=lam_ab_max, valinit=lam_ab_init)
lam_ab_sl.label.set_size(16)

# Temperature
T_loc = plt.axes([0.125, 0.05, 0.775, 0.03])
T_init = 300.
T_max = 600.
T_min = 1.
T_sl = Slider(T_loc, label=r'$T$ (K)', valmin=T_min, valmax=T_max, \
              valinit=T_init)
T_sl.label.set_size(16)

# External field
H_loc = plt.axes([0.125, 0.15, 0.775, 0.03])
H_init = 0.
H_max = 10.
H_min = -10.
H_sl = Slider(H_loc, label=r'$\mu_{0}H$ (T)', valmin=H_min, valmax=H_max, \
              valinit=H_init)
H_sl.label.set_size(16)

# Reset button
rst_loc = plt.axes([0.125, 0.9, 0.075, 0.05])
rst_button = Button(rst_loc, 'Reset', color='C4', hovercolor='C3')
rst_button.label.set_size(16)
    
### Plots
###______________________________________________________________

# Self-consistent subplot (Left, axis 1)
Ma_scale = np.linspace(-Ma_max, Ma_max, 100)
Mb_scale = np.linspace(-Mb_max, Mb_max, 100)

Ma_curve = mag_eq_a(Ma_scale, lam_ab_init, T_init, H_init)
Mb_curve = mag_eq_b(Mb_scale, lam_ab_init, T_init, H_init)

Ma_plot1, = ax1.plot(Mb_scale/1e3, Ma_curve/1e3, color='cyan')
Mb_plot1, = ax1.plot(Mb_curve/1e3, Ma_scale/1e3, color='orange')

# Magnetization-temperature subplot (Right, axis 2)
Temp_vec, Mag_a, Mag_b = get_mag(T_min, T_max, numpoints, lam_ab_init, H_init)

Ma_plot2, = ax2.plot(Temp_vec, Mag_a, color='cyan')
Mb_plot2, = ax2.plot(Temp_vec, Mag_b, color='orange')
Mtot_plot2, = ax2.plot(Temp_vec, Mag_a+Mag_b, color='white', zorder=10)
Mag_min = min( min(Mag_a), min(Mag_b) )
Mag_max = max( max(Mag_a), max(Mag_b) )

Temp_line, = ax2.plot([T_init,T_init], [Mag_min, Mag_max], color='red')

ax1.legend([r'Sublattice a', 'Sublattice b'], loc=1, fontsize=16)
ax2.legend([r'Sublattice a', 'Sublattice b', 'Total'], loc=1, fontsize=16)


### Updates
###______________________________________________________________

def update(val):
    # Pull val from sliders
    lam_ab_new = lam_ab_sl.val
    T_new = T_sl.val
    H_new = H_sl.val
    
    # Update axis 1
    Ma_curve_new = mag_eq_a(Ma_scale, lam_ab_new, T_new, H_new)
    Mb_curve_new = mag_eq_b(Mb_scale, lam_ab_new, T_new, H_new)
    Ma_plot1.set_data(Mb_scale/1e3, Ma_curve_new/1e3)
    Mb_plot1.set_data(Mb_curve_new/1e3, Ma_scale/1e3)
    
    # Update axis 2
    _, Mag_a_new, Mag_b_new = get_mag(T_min, T_max, numpoints, lam_ab_new, \
                                      H_new)
    Mag_min_new = min( min(Mag_a_new), min(Mag_b_new) )
    Mag_max_new = max( max(Mag_a_new), max(Mag_b_new) )
    Ma_plot2.set_ydata(Mag_a_new)
    Mb_plot2.set_ydata(Mag_b_new)
    Mtot_plot2.set_ydata(Mag_a_new+Mag_b_new)
    Temp_line.set_xdata([T_new,T_new])
    Temp_line.set_ydata([Mag_min_new, Mag_max_new])
    

def reset(event):
    lam_ab_sl.reset()
    T_sl.reset()
    

H_sl.on_changed(update)
lam_ab_sl.on_changed(update)
T_sl.on_changed(update)
rst_button.on_clicked(reset)

fig.canvas.draw_idle()
fig.show()