# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:19:33 2023

@author: anlinus
"""

import numpy as np
import matplotlib.pyplot as plt
import mystic as my
from itertools import repeat
import field_decomposition as fd
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata

plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

a = 2*1e-3
b = 1.4*1e-3
epsR = 6.8
freq = 105*1e9
remove = 9
modes = fd.get_propagating_modes(a, b, freq, epsR, 8)
nr_modes = len(modes)
#fd.list_modes(modes)
#modes_list = list(modes.items())
#modes_list = modes_list[0:len(modes_list)-remove]
#modes = dict(modes_list)
print('-----------------------------\n')
fd.list_modes(modes)
print('# of propagating modes: '+str(nr_modes))



# Filter array shallow HW: 
#file_re_ex = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h1_HW1_Ex_real.txt'
#file_im_ex = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h1_HW1_Ex_imag.txt'

#file_re_ey = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h1_HW1_Ey_real.txt'
#file_im_ey = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h1_HW1_Ey_imag.txt'

#file_re_ez = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h1_HW1_Ez_real.txt'
#file_im_ez = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h1_HW1_Ez_imag.txt'

#file_re_hx = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h1_HW1_Hx_real.txt'
#file_im_hx = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h1_HW1_Hx_imag.txt'

#file_re_hy = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h1_HW1_Hy_real.txt'
#file_im_hy = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h1_HW1_Hy_imag.txt'

#file_re_hz = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h1_HW1_Hz_real.txt'
#file_im_hz = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h1_HW1_Hz_imag.txt'

#file_power_flow = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h1_HW1_power_flow.txt'


# Filter array shallow HW: 
file_re_ex = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_HW1_Ex_real.txt'
file_im_ex = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_HW1_Ex_imag.txt'

file_re_ey = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_HW1_Ey_real.txt'
file_im_ey = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_HW1_Ey_imag.txt'

file_re_ez = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_HW1_Ez_real.txt'
file_im_ez = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_HW1_Ez_imag.txt'

file_re_hx = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_HW1_Hx_real.txt'
file_im_hx = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_HW1_Hx_imag.txt'

file_re_hy = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_HW1_Hy_real.txt'
file_im_hy = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_HW1_Hy_imag.txt'

file_re_hz = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_HW1_Hz_real.txt'
file_im_hz = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_HW1_Hz_imag.txt'

file_power_flow = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_HW1_power_flow.txt'

# Load data: 
data_re_ex = fd.load_comsol_data(file_re_ex)
data_im_ex = fd.load_comsol_data(file_im_ex)

data_re_ey = fd.load_comsol_data(file_re_ey)
data_im_ey = fd.load_comsol_data(file_im_ey)

data_re_ez = fd.load_comsol_data(file_re_ez)
data_im_ez = fd.load_comsol_data(file_im_ez)
ROTATE = False
# Load field data: 
Ex_re_data = fd.load_comsol_fields(data_re_ex,ROTATE)
Ex_im_data = fd.load_comsol_fields(data_im_ex,ROTATE)

Ey_re_data = fd.load_comsol_fields(data_re_ey,ROTATE)
Ey_im_data = fd.load_comsol_fields(data_im_ey,ROTATE)

Ez_re_data = fd.load_comsol_fields(data_re_ez,ROTATE)
Ez_im_data = fd.load_comsol_fields(data_im_ez,ROTATE)

# Exctract Comsol solution points: 
skip = 100
x = Ex_re_data[1]#[::skip]
y = Ex_re_data[2]#[::skip]

# Compose field: 
Ex_s = Ex_re_data[0]+1j*Ex_im_data[0]
Ey_s = Ey_re_data[0]+1j*Ey_im_data[0]
Ez_s = Ez_re_data[0]+1j*Ez_im_data[0]

#Ex_s = Ex_s/np.linalg.norm(Ex_s)
#Ey_s = Ex_s/np.linalg.norm(Ey_s)
#Ez_s = Ex_s/np.linalg.norm(Ez_s)

#Es = np.array([Ex_s[::skip],Ey_s[::skip],Ez_s[::skip]])
Es = np.array([Ex_s,Ey_s,Ez_s])

# H-field: 
    
# Load data: 
data_re_hx = fd.load_comsol_data(file_re_hx)
data_im_hx = fd.load_comsol_data(file_im_hx)

data_re_hy = fd.load_comsol_data(file_re_hy)
data_im_hy = fd.load_comsol_data(file_im_hy)

data_re_hz = fd.load_comsol_data(file_re_hz)
data_im_hz = fd.load_comsol_data(file_im_hz)
ROTATE = False
# Load field data: 
Hx_re_data = fd.load_comsol_fields(data_re_hx,ROTATE)
Hx_im_data = fd.load_comsol_fields(data_im_hx,ROTATE)

Hy_re_data = fd.load_comsol_fields(data_re_hy,ROTATE)
Hy_im_data = fd.load_comsol_fields(data_im_hy,ROTATE)

Hz_re_data = fd.load_comsol_fields(data_re_hz,ROTATE)
Hz_im_data = fd.load_comsol_fields(data_im_hz,ROTATE)

# Exctract Comsol solution points: 
skip = 100
x = Hx_re_data[1]#[::skip]
y = Hx_re_data[2]#[::skip]

# Compose field: 
Hx_s = Hx_re_data[0]+1j*Hx_im_data[0]
Hy_s = Hy_re_data[0]+1j*Hy_im_data[0]
Hz_s = Hz_re_data[0]+1j*Hz_im_data[0]

#Ex_s = Ex_s/np.linalg.norm(Ex_s)
#Ey_s = Ex_s/np.linalg.norm(Ey_s)
#Ez_s = Ex_s/np.linalg.norm(Ez_s)

#Es = np.array([Ex_s[::skip],Ey_s[::skip],Ez_s[::skip]])
Hs = np.array([Hx_s,Hy_s,Hz_s])

# Normalize: 

k = np.sqrt(np.sum(fd.cal_avg_poynting_vector(Es, Hs)))
Is = fd.cal_avg_poynting_vector(Es/k, Hs/k) #fd.cal_avg_poynting_vector(Es/k, Hs/k)
Es = Es/k
Hs = Hs/k

print('Sum of intensity: '+str(np.sum(Is)))

# Plot simulated field: 
#mm = 1e3
plt.scatter(x,y,c=Is)
plt.colorbar()
plt.axis('equal')
# Calculate eigenmodes: 
E,H = fd.get_field(a, b, freq, epsR, x, y, modes, False)

#%% Minimize: 
    
def cost(c):
    # insert code below
    c = np.reshape(c, c0_m.shape)  # To get c back in matrix-form
    c_complex = np.vectorize(complex)(c[:, 0], c[:, 1]) # Create array of complex numbers

    E_fit = np.array([np.tensordot(c_complex, E[0], axes=1), 
                      np.tensordot(c_complex, E[1], axes=1), 
                      np.tensordot(c_complex, E[2], axes=1)])  # Field we try to fit
    
    #k_fit = np.linalg.norm(E_fit[0:2])
    #E_fit = E_fit/k_fit
    return np.linalg.norm(Es-E_fit)**2


#@my.constraints.normalized()
#def constraint(c): 
#    c = np.array(c)
#    sq = np.abs(c)**2
#    return sq

def penalty_norm(c,target):
    c = np.array(c)
    return np.linalg.norm(c)**2-target

@my.penalty.quadratic_equality(condition=penalty_norm, kwds={'target':1})
def penalty(c):
    return 0

# bounds on optimized parameters
bnds=tuple(repeat((-1,1),2*nr_modes)) # NOTE: bounds must be double the len of modes bcs the unravel

#%% Solve:
#initial guess
c0_m = fd.get_random_initial_guess(nr_modes)
print(c0_m)
c0 = c0_m.ravel()    
if __name__ == '__main__':
    mon = my.monitors.VerboseMonitor(10)
    result = my.solvers.diffev2(cost, 
                                x0 = c0,
                                #bounds=bnds, 
                                #constraints = constraint, 
                                #penalty = penalty,
                                npop=300, 
                                ftol=1e-8, 
                                gtol=200, 
                                disp=True, 
                                full_output=False,
                                maxiter=8000,
                                cross=0.9, 
                                scale=0.8,
                                itermon=mon)
  
    print(result)

c_opt = np.array(result)
c_opt = np.reshape(c_opt, c0_m.shape)
c_opt = np.vectorize(complex)(c_opt[:, 0], c_opt[:, 1]) # Create array of complex numbers 

#%% Plot fields:

E_opt = np.array([np.tensordot(c_opt, E[0], axes=1), 
                  np.tensordot(c_opt, E[1], axes=1), 
                  np.tensordot(c_opt, E[2], axes=1)])  # Field we try to fit

H_opt = np.array([np.tensordot(c_opt, H[0], axes=1), 
                  np.tensordot(c_opt, H[1], axes=1), 
                  np.tensordot(c_opt, H[2], axes=1)]) 

#E_opt = E_opt 
#H_opt = H_opt 

#norm_opt = np.sqrt(np.sum(fd.cal_avg_poynting_vector(E_opt, H_opt)))
#print('Normalization factor from power: '+str(norm))
#print('Normalization factor from field: '+str(norm_field))
#print('Normalization factor from opt. field: '+str(norm_opt))

I_opt = fd.cal_avg_poynting_vector(E_opt, H_opt)
#I_opt = fd.cal_avg_poynting_vector(E_opt, H_opt)

print('Integral of simulated intensity: '+ str(np.sum(Is)))
print('Integral of reconstructed intensity: '+ str(np.sum(I_opt)))
#print('Integral of power flow from COMSOL: '+ str(np.sum(power)))
print('Integral of complex coeff. : '+str(np.sum(np.abs(c_opt)**2))) 
#%%
#Ex = np.tensordot(c_opt, E[0],axes=1)
#Ey = np.tensordot(c_opt, E[1],axes=1)
#Ez = np.tensordot(c_opt, E[2],axes=1)
#E_opt = np.array([Ex,Ey,Ez])
#I_opt = fd.cal_intensity(E_opt)#/np.linalg.norm(E_opt[0:2]))

# Calculate the difference in intensities: 
diff = np.abs(Is-I_opt)

# Calculate the correlation using the paper formula: 
dIs = Is-np.mean(Is)
dI_opt = I_opt-np.mean(I_opt)

nom = np.sum(dIs*dI_opt)
denom = np.sqrt(np.sum(dIs**2)*np.sum(dI_opt**2))
C = np.abs(nom/denom)
print('C = '+str(C))

# Compute the correlation between the two arrays
corr = round(np.corrcoef(Is, I_opt)[0, 1],8)

# Print the correlation
print("Correlation between Is and I_opt:", corr) 
#%% Interpolate results: 

N = 100 # Number of points for interpolation
xp = np.linspace(0,a,N)
yp = np.linspace(0,b,N)
X, Y = np.meshgrid(xp,yp)
Is = griddata((x,y),Is,(X,Y),fill_value = 0,method = 'cubic')
I_opt = griddata((x,y),I_opt,(X,Y),fill_value = 0,method = 'cubic')
diff = griddata((x,y),diff,(X,Y),fill_value = 0,method = 'cubic')


#%%
#Es = Es/np.linalg.norm(Es)
# Normalize complex coefficeints: 
c_opt = c_opt/np.linalg.norm(c_opt)

# Normalize data for display only: 
I_opt = I_opt/np.max(np.abs(Is))
diff = diff/np.max(np.abs(Is))
Is = Is/np.max(np.abs(Is))

vmax = np.max(Is)
vmin = np.min(Is)

mm = 1e3
    
# TM modes

fig = plt.figure()

gs = gridspec.GridSpec(3, 2,figure = fig)

# Mode content plots:
height = [np.abs(c)**2 for c in c_opt]
keys = [key for key in modes.keys()]

for i in range(len(height)):
    if np.around(height[i],3)>0:
        print(keys[i]+' : '+str(np.around(height[i],3))+'\n')

ax1 = plt.subplot(gs[:,0])
ax1.barh(keys,height)
ax1.set_xlabel('$|c_n|^2$',font)
ax1.grid(True,linestyle='--',alpha=0.4)
ax1.set_yticklabels(ax1.get_yticklabels(),rotation=0,size=10,fontfamily="Times New Roman")
ax1.set_xlim((0,1.1))
ax1.set_title(f'Mode Content, HERD-2 {int(freq*1e-9)} GHz',font)

lvls = 30
size = 3
cmp = 'viridis'
# Intensity plots: 

ax2 = plt.subplot(gs[0,1])
cf1 = ax2.pcolormesh(X*mm,Y*mm,Is,cmap=cmp, vmin = vmin, vmax = vmax)
ax2.set_title('$I_s$',font)
ax2.set_aspect('equal')
#ax2.set_anchor('W')
#ax2.set_xlabel('x (mm)',font)
ax2.set_ylabel('y (mm)',font)
cbar1 = fig.colorbar(cf1, ax=ax2)

ax3 = plt.subplot(gs[1,1])
cf2 = ax3.pcolormesh(X*mm,Y*mm,I_opt,cmap=cmp,vmin = vmin, vmax = vmax)
ax3.set_title('$I_r$',font)
ax3.set_aspect('equal')
#ax3.set_anchor('W')
#ax3.set_xlabel('x (mm)',font)
ax3.set_ylabel('y (mm)',font)
cbar2 = fig.colorbar(cf2, ax=ax3)

ax4 = plt.subplot(gs[2,1])
cf3 = ax4.pcolormesh(X*mm,Y*mm,diff,cmap=cmp, vmin = vmin, vmax = vmax)
ax4.set_title('$|I_{s}-I_{r}|$',font)
ax4.set_aspect('equal')
#ax4.set_anchor('W')
ax4.set_xlabel('x (mm)',font)
ax4.set_ylabel('y (mm)',font)
cbar3 = fig.colorbar(cf2, ax=ax4)

gs.tight_layout(fig,pad=0.05)
#fig.colorbar(cf1, ax = (ax1,ax2,ax3))
name = 'mode_decomp'
#plt.savefig(loc+ r'\\' + filename + name+'.svg',format='svg')
#plt.savefig(loc+ r'\\' + filename + name+'.png',format='png')
plt.show()

#fd.plot_mode_content(modes, c_opt)
filename = ''#loc+ r'\\' + filename
fd.plot_field_components_vec(Es, E_opt,X, Y, x, y, f'{int(freq*1e-9)} GHz', filename)
#fd.plot_field_components_vec(Es, X, Y, x, y, f'Simulated, {int(freq*1e-9)} GHz', filename)
#fd.plot_field_components_vec(E_opt, X,Y,x,y, f'Reconstructed, {int(freq*1e-9)} GHz', filename)