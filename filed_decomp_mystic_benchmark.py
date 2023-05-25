# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 06:45:47 2023

@author: Linus
"""

import numpy as np
import matplotlib.pyplot as plt
import mystic as my
from itertools import repeat
import field_decomposition as fd

a = 10*1e-3
b = 5*1e-3
epsR = 1
freq = 60*1e9
N = 50
x = np.linspace(0, a, N)
y = np.linspace(0, b, N)
X, Y = np.meshgrid(x, y)

modes = fd.get_propagating_modes(a, b, freq, epsR, 5)

nr_modes = len(modes)

print('# of propagating modes: '+str(nr_modes))

E, H = fd.get_field(a, b, freq, epsR, X, Y, modes, False)

# Correct normalization:
#E = np.array([Ex, Ey, Ez])
E_norm = E/np.linalg.norm(E)
I_norm = fd.cal_intensity(E_norm)

# Create test field for benchmark:
Etot_t, Htot_t, c_t = fd.get_random_field(E,H,nr_modes)
Etot_t = Etot_t/np.linalg.norm(Etot_t)
#print(np.linalg.norm(Etot_t))

# Save files:
# filename = 'a_2mm_b_2mm_freq_90GHz'
# fd.create_files(Etot_t, X, Y, filename)

#%%

def cost(c):
    # insert code below
    c = np.reshape(c, c0_m.shape)  # To get c back in matrix-form
    c_complex = np.vectorize(complex)(c[:, 0], c[:, 1]) # Create array of complex numbers

    E_fit = np.array([np.tensordot(c_complex, E[0], axes=1), 
                      np.tensordot(c_complex, E[1], axes=1), 
                      np.tensordot(c_complex, E[2], axes=1)])  # Field we try to fit
    
    norm = np.linalg.norm(E_fit)
    if norm > 0: 
        E_fit = E_fit/norm
    return np.linalg.norm(Etot_t - E_fit)

def penalty_normalized(c,target):
    c = np.array(c)
    return np.linalg.norm(c)**2-target

@my.penalty.quadratic_equality(condition=penalty_normalized, kwds={'target':1})
def penalty(c):
    return 0

@my.constraints.normalized()
def constraint(c): 
    c = np.array(c)
    sq = c**2
    return sq

# bounds on optimized parameters
bnds=tuple(repeat((-1,1),2*nr_modes)) # NOTE: bounds must be double the len of modes bcs the unravel

#initial guess
c0_m = fd.get_random_initial_guess(nr_modes)
print(c0_m)
c0 = c0_m.ravel()

#%% Run: 
if __name__ == '__main__':
    mon = my.monitors.VerboseMonitor(10)
    result = my.solvers.diffev2(cost, 
                                x0 = c0,
                                bounds=bnds, 
                                #constraints = constraint, 
                                penalty = penalty,
                                npop=40, 
                                ftol=1e-8, 
                                gtol=200, 
                                disp=True, 
                                full_output=False,
                                maxiter=5000,
                                cross=0.8, 
                                scale=0.9,
                                itermon=mon)
  
    print(result)

#%% Plot results:
c_opt = np.array(result)
c_opt = np.reshape(c_opt, c0_m.shape)
c_opt = np.vectorize(complex)(c_opt[:, 0], c_opt[:, 1]) # Create array of complex numbers    

Ex = np.tensordot(c_opt, Ex, axes=1)
Ey = np.tensordot(c_opt, Ey, axes=1)
Ez = np.tensordot(c_opt, Ez, axes=1)
E_opt= np.array([Ex, Ey, Ez])    
mm = 1e3

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.contourf(X*mm, Y*mm, fd.cal_intensity(Etot_t), levels=30)
ax1.set_title('$|E^s(x,y)|^2$, Simulation')
ax1.set_aspect('equal', 'box')
ax1.set_xlabel('x (mm)')
ax1.set_ylabel('y (mm)')

ax2.contourf(X*mm, Y*mm, fd.cal_intensity(E_opt), levels=30)
ax2.set_title('$|\sum_{n=1}^n c_{n}E_n(x,y)|^2$, Optimized')
ax2.set_aspect('equal', 'box')
ax2.set_xlabel('x (mm)')
ax2.set_ylabel('y (mm)')
# plt.title(f'TE{mode[0]}{mode[1]}')
fig.tight_layout()
plt.show()

fd.plot_mode_content(modes, c_opt)