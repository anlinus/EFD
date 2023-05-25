# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 13:15:23 2023

@author: Linus
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import repeat
import field_decomposition as fd
import matplotlib.gridspec as gridspec

plt.style.use('seaborn-colorblind')
plt.rcParams['figure.dpi'] = 300
font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

a = 2*1e-3
b = 1.4*1e-3
epsR = 6.8
freq = 100*1e9
N = 100
x = np.linspace(0, a, N)
y = np.linspace(0, b, N)
X, Y = np.meshgrid(x, y)

modes = fd.get_propagating_modes(a, b, freq, epsR, 3)
fd.list_modes(modes)
modes_list = list(modes.items())
modes_list = modes_list[0:len(modes_list)-2]
modes = dict(modes_list)
print('---------------------------------\n')
fd.list_modes(modes)
nr_modes = len(modes)
print('# of propagating modes: '+str(nr_modes))

E,H = fd.get_field(a, b, freq, epsR, X, Y, modes, False)

# Create test field for benchmark:
Etot_t, Htot_t, c_t = fd.get_random_field(E,H, nr_modes)
Etot_t = Etot_t/np.linalg.norm(Etot_t)
Itot_t = fd.cal_intensity(Etot_t)
print('Norm of E-field: '+str(np.linalg.norm(Etot_t)))
print('Norm of power: '+str(np.sum(Itot_t)))
fd.plot_field_components(Etot_t, X, Y, 'Test')

# %% Minimize:


def cost(c):
    # insert code below
    c = np.reshape(c, c0_m.shape)  # To get c back in matrix-form
    
    # Create array of complex numbers: [a1+jb1,...,an+jbn]
    c_complex = np.vectorize(complex)(c[:, 0], c[:, 1])

    E_fit = np.array([np.tensordot(c_complex, E[0], axes=1),
                      np.tensordot(c_complex, E[1], axes=1)])
                      #np.tensordot(c_complex, E[2], axes=1)])  # Field we try to fit
    norm = np.linalg.norm(E_fit)
    
    if norm > 0: # Can't divide by zero
        E_fit = E_fit/norm
        
    return np.linalg.norm(Etot_t - E_fit)

def constraint(c):
    return np.linalg.norm(c)**2 - 1

con1 = {'type': 'eq', 'fun': constraint}
cons = [con1]

# bounds on optimized parameters
bnds = tuple(repeat((-1, 1), 2*nr_modes)) # NOTE: bounds must be double the len of modes bcs the np.ravel

# initial guess
c0_m = fd.get_random_initial_guess(nr_modes)
c0 = c0_m.ravel()

result= minimize(cost, c0, method='SLSQP', constraints = cons, bounds=bnds, tol=1e-6, options={'maxiter':1000}) # constraints=cons
#result= minimize(cost, c0, method='SLSQP', tol=1e-9) # constraints=cons
print(result)

#%%
c_opt = result.x
c_opt = np.reshape(c_opt, c0_m.shape)
c_opt = np.vectorize(complex)(c_opt[: , 0], c_opt[: , 1])

Ex = np.tensordot(c_opt, E[0], axes=1)
Ey = np.tensordot(c_opt, E[1], axes=1)
E_opt= np.array([Ex, Ey])
E_opt = E_opt/np.linalg.norm(E_opt)
I_opt = fd.cal_intensity(E_opt)
diff = Itot_t-I_opt
# %% Plot results:
    
# Compute the correlation between the two arrays
corr = np.corrcoef(Itot_t, I_opt)[0, 1]

# Print the correlation
print("Correlation between Is and I_opt:", corr) 

vmax = np.max(Itot_t)
vmin = np.min(Itot_t)

mm = 1e3
    
# TM modes

fig = plt.figure()

gs = gridspec.GridSpec(3, 2,figure = fig)

# Mode content plots:
height = [np.abs(c)**2 for c in c_opt]
keys = [key for key in modes.keys()]

ax1 = plt.subplot(gs[:,0])
ax1.barh(keys,height)
ax1.set_xlabel('$|c_n|^2$',font)
ax1.grid(True,linestyle='--',alpha=0.4)
ax1.set_yticklabels(ax1.get_yticklabels(),rotation=0,size=12,fontfamily="Times New Roman")
ax1.set_xlim((0,1.1))
ax1.set_title(f'Mode Content, {int(freq*1e-9)} GHz',font)

lvls = 30
size = 3

# Intensity plots: 

ax2 = plt.subplot(gs[0,1])
cf1 = ax2.pcolormesh(X*mm,Y*mm,Itot_t, vmin = vmin, vmax = vmax)
ax2.set_title('$I_{t}$',font)
ax2.set_aspect('equal')
#ax2.set_anchor('W')
ax2.set_xlabel('x (mm)',font)
ax2.set_ylabel('y (mm)',font)
cbar1 = fig.colorbar(cf1, ax=ax2)

ax3 = plt.subplot(gs[1,1])
cf2 = ax3.pcolormesh(X*mm, Y*mm, I_opt, vmin = vmin, vmax = vmax) 
ax3.set_title(f'$I_r$, C = {np.around(corr,5)}',font)
ax3.set_aspect('equal')
#ax3.set_anchor('W')
ax3.set_xlabel('x (mm)',font)
ax3.set_ylabel('y (mm)',font)
cbar2 = fig.colorbar(cf2, ax=ax3)

ax4 = plt.subplot(gs[2,1])
cf3 = ax4.pcolormesh(X*mm,Y*mm,diff, vmin = vmin, vmax = vmax)
ax4.set_title('$I_{s}-I_{r}$',font)
ax4.set_aspect('equal')
#ax4.set_anchor('W')
ax4.set_xlabel('x (mm)',font)
ax4.set_ylabel('y (mm)',font)
cbar3 = fig.colorbar(cf2, ax=ax4)

gs.tight_layout(fig)
#fig.colorbar(cf1, ax = (ax1,ax2,ax3))
name = f'mode_decomp_benchmark_{int(freq*1e-9)}GHz'
loc = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Plots\Thesis Plots'
plt.savefig(loc+ r'\\' + name+'.svg',format='svg')
plt.savefig(loc+ r'\\' + name+'.png',format='png')
plt.show()

#fd.plot_mode_content(modes, c_opt)

#fd.plot_field_components_vec(Es, X, Y, x, y, 'Simulated')
#fd.plot_field_components_vec(E_opt, X,Y,x,y, 'Reconstructed')