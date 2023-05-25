# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:19:33 2023

@author: anlinus
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import repeat
import field_decomposition as fd
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
import seaborn as sns

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
freq = 105*1e9

loc = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Plots\Thesis Plots'
filename = f'{int(freq*1e-9)}GHz_HERD2_HW1_h5_'

mn = 8 # QxQ grid of modes
remove = 0 # Number of modes removed

modes = fd.get_propagating_modes(a, b, freq, epsR, mn)
#fd.list_modes(modes)
#modes_list = list(modes.items())
#modes_list = modes_list[0:len(modes_list)-remove]
#modes = dict(modes_list)
nr_modes = len(modes)
print('-----------------------------\n')
fd.list_modes(modes)
print('# of propagating modes: '+str(nr_modes))


# Load comsol data: 

# Filter array shallow HW: 
#file_re_ex = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h5_HW1_Ex_real.txt'
#file_im_ex = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h5_HW1_Ex_imag.txt'

#file_re_ey = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h5_HW1_Ey_real.txt'
#file_im_ey = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h5_HW1_Ey_imag.txt'

#file_re_ez = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h5_HW1_Ez_real.txt'
#file_im_ez = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h5_HW1_Ez_imag.txt'

#file_re_hx = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h5_HW1_Hx_real.txt'
#file_im_hx = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h5_HW1_Hx_imag.txt'

#file_re_hy = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h5_HW1_Hy_real.txt'
#file_im_hy = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h5_HW1_Hy_imag.txt'

#file_re_hz = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h5_HW1_Hz_real.txt'
#file_im_hz = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h5_HW1_Hz_imag.txt'

#file_power_flow = r'C:\Users\Linus\Documents\Chalmers\Master Thesis\Scripts\Field Data Array Deep New Mesh\105GHz_array_a20_b14_h5_HW1_power_flow.txt'

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

ROTATE = False

#----------Load power flow data----------

data_power_flow = fd.load_comsol_data(file_power_flow)
power = fd.load_comsol_fields(data_power_flow,ROTATE)[0]
norm = np.sqrt(np.sum(power))
#power = power / np.sum(power)
#----------------------------------------

#----------Electric Field Data-----------
data_re_ex = fd.load_comsol_data(file_re_ex)
data_im_ex = fd.load_comsol_data(file_im_ex)

data_re_ey = fd.load_comsol_data(file_re_ey)
data_im_ey = fd.load_comsol_data(file_im_ey)

data_re_ez = fd.load_comsol_data(file_re_ez)
data_im_ez = fd.load_comsol_data(file_im_ez)

# Load field data: 
Ex_re_data = fd.load_comsol_fields(data_re_ex,ROTATE)
Ex_im_data = fd.load_comsol_fields(data_im_ex,ROTATE)

Ey_re_data = fd.load_comsol_fields(data_re_ey,ROTATE)
Ey_im_data = fd.load_comsol_fields(data_im_ey,ROTATE)

Ez_re_data = fd.load_comsol_fields(data_re_ez,ROTATE)
Ez_im_data = fd.load_comsol_fields(data_im_ez,ROTATE)

Ex_s = Ex_re_data[0]+1j*Ex_im_data[0]
Ey_s = Ey_re_data[0]+1j*Ey_im_data[0]
Ez_s = Ez_re_data[0]+1j*Ez_im_data[0]

Es = np.array([Ex_s,Ey_s,Ez_s])
#---------------------------------------

#----------Magnetic Field Data-----------
data_re_hx = fd.load_comsol_data(file_re_hx)
data_im_hx = fd.load_comsol_data(file_im_hx)

data_re_hy = fd.load_comsol_data(file_re_hy)
data_im_hy = fd.load_comsol_data(file_im_hy)

data_re_hz = fd.load_comsol_data(file_re_hz)
data_im_hz = fd.load_comsol_data(file_im_hz)

# Load field data: 
Hx_re_data = fd.load_comsol_fields(data_re_hx,ROTATE)
Hx_im_data = fd.load_comsol_fields(data_im_hx,ROTATE)

Hy_re_data = fd.load_comsol_fields(data_re_hy,ROTATE)
Hy_im_data = fd.load_comsol_fields(data_im_hy,ROTATE)

Hz_re_data = fd.load_comsol_fields(data_re_hz,ROTATE)
Hz_im_data = fd.load_comsol_fields(data_im_hz,ROTATE)

Hx_s = Hx_re_data[0]+1j*Hx_im_data[0]
Hy_s = Hy_re_data[0]+1j*Hy_im_data[0]
Hz_s = Hz_re_data[0]+1j*Hz_im_data[0]

Hs = np.array([Hx_s,Hy_s,Hz_s])
#---------------------------------------

k = np.sqrt(np.sum(fd.cal_avg_poynting_vector(Es, Hs)))

print('Normalization factor from power: '+str(norm))
print('Normalization factor from field: '+str(k))
# Calculate simulated intensity: 
Is = fd.cal_avg_poynting_vector(Es/k, Hs/k)
#Is = fd.cal_avg_poynting_vector(Es, Hs)
# Normalize field: 
Es = Es / k #np.linalg.norm(Es)
Hs = Hs / k #np.linalg.norm(Hs)


# Exctract COMSOL solution points: 

x = Ez_re_data[1]#[::skip]
y = Ez_re_data[2]#[::skip]

#Is = fd.cal_intensity(Es)#/np.linalg.norm(Es[0:2]))
#Es = Es/np.linalg.norm(Es)

print('Sum of intensity vector: '+str(np.sum(Is))) 
E, H = fd.get_field(a, b, freq, epsR, x, y, modes, False)


#%% Minimize: 
    
def cost(c):
    # insert code below
    c = np.reshape(c, c0_m.shape)  # To get c back in matrix-form
    c_complex = np.vectorize(complex)(c[:, 0], c[:, 1]) # Create array of complex numbers

    E_fit = np.array([np.tensordot(c_complex, E[0], axes=1), 
                      np.tensordot(c_complex, E[1], axes=1), 
                      np.tensordot(c_complex, E[2], axes=1)])  # Field we try to fit
    
    #norm = np.sqrt(np.abs(np.sum(fd.cal_avg_poynting_vector(E_fit, H_fit))))
    
    #norm = np.linalg.norm(E_fit)
    #if norm > 0: 
    #   E_fit = E_fit / norm

    return np.linalg.norm(Es-E_fit)

def constraint(c):
    return np.linalg.norm(c) - 1

con1 = {'type': 'eq', 'fun': constraint}
cons = [con1]

# bounds on optimized parameters
bnds=tuple(repeat((-1,1),2*nr_modes)) # NOTE: bounds must be double the len of modes bcs the unravel

#initial guess
c0_m = fd.get_random_initial_guess(nr_modes)
c0 = 0*c0_m.ravel()

result = minimize(cost,c0,method='SLSQP', bounds=bnds, constraints=cons, tol=1e-12,options={'maxiter':2000}) #bounds=bnds
#result = minimize(cost,c0,method='SLSQP',tol=1e-8,options={'maxiter':2000})
print(result)

c_opt = result.x
c_opt = np.reshape(c_opt,c0_m.shape)
c_opt = np.vectorize(complex)(c_opt[:,0], c_opt[:,1]) 
#%% Experiment with intensities: 

# Calculate reconstructed intensities: 

#I_opt = fd.cal_reconstructed_intensity(E, c_opt, a, b, freq, epsR, modes)

E_opt = np.array([np.tensordot(c_opt, E[0], axes=1), 
                  np.tensordot(c_opt, E[1], axes=1), 
                  np.tensordot(c_opt, E[2], axes=1)])  # Field we try to fit

H_opt = np.array([np.tensordot(c_opt, H[0], axes=1), 
                  np.tensordot(c_opt, H[1], axes=1), 
                  np.tensordot(c_opt, H[2], axes=1)]) 



k_opt = np.sqrt(np.sum(fd.cal_avg_poynting_vector(E_opt, H_opt)))
#E_opt = E_opt / k_opt
#H_opt = H_opt / k_opt
#print('Normalization factor from power: '+str(norm))
#print('Normalization factor from field: '+str(norm_field))
#print('Normalization factor from opt. field: '+str(norm_opt))

I_opt = fd.cal_avg_poynting_vector(E_opt, H_opt)
#I_opt = fd.cal_avg_poynting_vector(E_opt, H_opt)

print('Integral of simulated intensity: '+ str(np.sum(Is)))
print('Integral of reconstructed intensity: '+ str(np.sum(I_opt)))
print('Integral of power flow from COMSOL: '+ str(np.sum(power)))
print('Integral of complex coeff. : '+str(np.sum(np.abs(c_opt)**2))) 
#%% Compare to calculated power: 

for i in range(nr_modes):
    En = c_opt[i]*np.array([E[0][i],E[1][i],E[2][i]]) 
    Hn = c_opt[i]*np.array([H[0][i],H[1][i],H[2][i]]) 
    Pn = np.sum(fd.cal_avg_poynting_vector(En, Hn))
    print(f'P{i} = '+str(Pn)+'\n')

for i in range(len(c_opt)):
    print(f'|c{i}|^2 = '+str(np.abs(c_opt[i])**2)+'\n')
print(np.sum(np.abs(c_opt)**2))

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
Is = griddata((x,y),Is,(X,Y),fill_value = 0,method = 'linear')
P = griddata((x,y),power,(X,Y),fill_value = 0,method = 'linear')
I_opt = griddata((x,y),I_opt,(X,Y),fill_value = 0,method = 'linear')
diff = griddata((x,y),diff,(X,Y),fill_value = 0,method = 'linear')

# Normalize data: 
#Es = Es/np.linalg.norm(Es)
# Normalize complex coefficeints: 


#%% Normalize data for display only: 
I_opt = I_opt/np.max(np.abs(Is))
diff = diff/np.max(np.abs(Is))
Is = Is/np.max(np.abs(Is))
P = P/np.max(np.abs(P))

#%% Plot intensities

fd.plot_intensities(Is, I_opt, X, Y,'105GHz_')

#%% Plot
E_opt = E_opt / np.linalg.norm(E_opt)
Es = Es / np.linalg.norm(Es)
c_opt = c_opt/np.linalg.norm(c_opt)

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
ax1.set_title(f'Mode Content, {int(freq*1e-9)} GHz',font)

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
filename = loc+ r'\\' + filename
fd.plot_field_components_vec(Es, E_opt,X, Y, x, y, f'{int(freq*1e-9)} GHz', filename)
#fd.plot_field_components_vec(E_opt, X,Y,x,y, f'Reconstructed, {int(freq*1e-9)} GHz', filename+'reconstruction_')
