# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:12:43 2023

@author: llha9
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.constants as constants
from scipy.optimize import minimize
from itertools import repeat
from scipy.interpolate import griddata
from cmath import sqrt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

plt.style.use('seaborn-colorblind')
plt.rcParams['figure.dpi'] = 300
font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

# Physical constanst:
eps0 = constants.epsilon_0
mu0 = constants.mu_0

#TODO: Add functionality to import complex data from comsol (might need to be done from two files (real and imaginary))
def load_comsol_data(file):
    return np.loadtxt(file,comments = '%')

def load_comsol_fields(data, ROTATE):
    # Extract field data: 
    x = data[:,0]-np.min(data[:,0]) # NOTE: Make sure you have the same coordinate system as in COMSOL
    y = data[:,1]-np.min(data[:,1])
    E = data[:,3]
    
    if ROTATE: 
        extracted_data = np.array([E,y,x])
    else:
        extracted_data = np.array([E,x,y])
    return extracted_data

def interpolate_data(E_data,X,Y):
    E_interp = griddata((E_data[1],E_data[2]),E_data[0],(X,Y),fill_value = 0,method = 'linear')
    return E_interp

def plot_field(X, Y, Ex, Ey, Ez, mode):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.contourf(X, Y, np.abs(Ex), levels=30)
    ax1.set_title('$E_x$')
    ax1.set_aspect('equal', 'box')

    ax2.contourf(X, Y, np.abs(Ey), levels=30)
    ax2.set_title('$E_y$')
    ax2.set_aspect('equal', 'box')

    ax3.contourf(X, Y, np.abs(Ez), levels=30)
    ax3.set_title('$E_z$')
    ax3.set_aspect('equal', 'box')

    # plt.title(f'TE{mode[0]}{mode[1]}')
    fig.tight_layout()
    plt.show()


def cal_kc(a, b, mode):
    return np.sqrt((mode[0]*np.pi/a)**2 + (mode[1]*np.pi/b)**2)


def cal_k(freq, epsR):
    return 2*np.pi*freq*np.sqrt(mu0*eps0*epsR)


def cal_propagationConstant(k, kc):
    return 1j*np.sqrt(k**2+kc**2)


def cal_fc(a, b, mode, eps_r):
    kc = cal_kc(a, b, mode)
    fc = kc/(2*np.pi*np.sqrt(mu0*eps0*eps_r))
    #print('fc = '+str(fc*1e-9)+' GHz')
    return fc

def cal_beta(a,b,mode,freq, epsR): # mode = [m,n]
    k = cal_k(freq, epsR)
    kc = cal_kc(a, b, mode)
    beta = np.sqrt(k**2-kc**2)
    #print(beta)
    return beta
def cal_eta(epsR):
    return np.sqrt(mu0/(eps0*epsR))

def cal_Z_TE(a,b,mode,freq,epsR):
    k = cal_k(freq, epsR)
    eta = cal_eta(epsR)
    beta = cal_beta(a, b, mode, freq, epsR)
    ZTE = k*eta/beta
    #print(ZTE)
    return ZTE

def cal_Z_TM(a,b,mode,freq,epsR):
    k = cal_k(freq, epsR)
    eta = cal_eta(epsR)
    beta = cal_beta(a, b, mode, freq, epsR)
    ZTM = beta*eta/k
    #print(ZTM)
    return ZTM

def cal_TE_modes(a, b, freq, epsR, X, Y, mode, plot):

    kc = cal_kc(a, b, mode)
    k = cal_k(freq, epsR)
    beta = sqrt(k**2 - kc**2)
    w = 2*np.pi*freq
    
    #----- E-field components -----
    Ez = X*0
    # Calculate x-component:
    if mode[1] != 0:
        Ex = (1j*w*mu0*mode[1]*np.pi/(kc**2 * b)) * np.cos(mode[0]*np.pi*X/a)*np.sin(mode[1]*np.pi*Y/b)
        #Ex = Ex/np.linalg.norm(Ex)
        #Ex = Ex/np.max(np.abs(Ex))
        #Ex = mode[1]*np.cos(mode[0]*np.pi*X/a)*np.sin(mode[1]*np.pi*Y/b)
    else:
        Ex = X*0
    # Calculate y-component:
    if mode[0] != 0:  
        Ey = (-1j*w*mu0*mode[0]*np.pi/(kc**2 * a)) * np.sin(mode[0]*np.pi*X/a)*np.cos(mode[1]*np.pi*Y/b)
        #Ey = Ey/np.linalg.norm(Ey)
        #Ey = Ey/np.max(np.abs(Ey))
        #Ey = -mode[0]*np.sin(mode[0]*np.pi*X/a)*np.cos(mode[1]*np.pi*Y/b)

    else:
        Ey = X*0
    #------------------------------
    
    #----- H-field components -----
    Hz = np.cos(mode[0]*np.pi*X/a)*np.cos(mode[1]*np.pi*Y/b)
    # Calculate x-component:
    #if mode[0] != 0:
    Hx = (1j*beta*mode[0]*np.pi / (kc**2 * a))*np.sin(mode[0]*np.pi*X/a)*np.cos(mode[1]*np.pi*Y/b)
    #else:
        #Hx = X*0
    # Calculate y-component:
    #if mode[1] != 0:    
    Hy = (1j*beta*mode[1]*np.pi / (kc**2 * a))*np.cos(mode[0]*np.pi*X/a)*np.sin(mode[1]*np.pi*Y/b)
    #else:
        #Hy = X*0
    #------------------------------
    
    if plot:
        plot_field(X, Y, Ex, Ey, Ez, mode)
        
    E = np.array([Ex, Ey, Ez])
    H = np.array([Hx,Hy,Hz]) 
    k = np.sqrt(np.sum(cal_avg_poynting_vector(E, H)))
    P = np.sum(cal_avg_poynting_vector(E/k, H/k))
    print('P='+str(P))
    return E/k, H/k#E / np.linalg.norm(E), H / np.linalg.norm(H)


def cal_TM_modes(a, b, freq, epsR, X, Y, mode, plot):

    kc = cal_kc(a, b, mode)
    k = cal_k(freq, epsR)
    beta = sqrt(k**2 - kc**2)
    w = 2*np.pi*freq
    
    #----- E-field components -----
    
    # Calculate x-component:
    Ex = (-1j*beta*mode[0]*np.pi / (kc**2 * a)) * np.cos(mode[0]*np.pi*X/a)*np.sin(mode[1]*np.pi*Y/b)

    # Calculate y-component:

    Ey = (-1j*beta*mode[1]*np.pi / (kc**2 * b)) * np.sin(mode[0]*np.pi*X/a)*np.cos(mode[1]*np.pi*Y/b)

    # Calculate z-component:
    Ez = np.sin(mode[0]*np.pi*X/a)*np.sin(mode[1]*np.pi*Y/b)

    #------------------------------

    #----- H-field components -----
    Hx = (1j*w*eps0*epsR*mode[1]*np.pi/(kc**2 * b))*np.sin(mode[0]*np.pi*X/a)*np.cos(mode[1]*np.pi*Y/b)

    # Calculate y-component:
    Hy = (-1j*w*eps0*epsR*mode[0]*np.pi/(kc**2 * a))*np.cos(mode[0]*np.pi*X/a)*np.sin(mode[1]*np.pi*Y/b)

    # Calculate z-component:
    Hz = X*0
    #------------------------------
    
    if plot:
        plot_field(X, Y, Ex, Ey, Ez, mode)
        
    E = np.array([Ex, Ey, Ez]) 
    H = np.array([Hx,Hy,Hz])
    k = np.sqrt(np.sum(cal_avg_poynting_vector(E, H)))
    return E/k, H/k#E / np.linalg.norm(E), H / np.linalg.norm(H)

def cal_avg_poynting_vector(E,H):
    S_flow = 0.5*np.real( E[0]*np.conj(H[1]) - E[1]*np.conj(H[0]) )
    return S_flow

def cal_reconstructed_intensity(E,c,a,b,freq,epsR,modes):
    
    N = len(modes)
    Z = np.zeros(N)
    i = 0
    for key,value in modes.items(): 
        #print(key)
        if 'TE' in key: 
            Z[i] = cal_Z_TE(a, b, value[0], freq, epsR)
        else: 
            Z[i] = cal_Z_TM(a, b, value[0], freq, epsR)
        i += 1
    # Multiply with elementwise with c, but maintain vector form: 
    #print(Z)
    # Ez = c[:,np.newaxis] * E[2]
    
    # Calculate intensity: 
    Ir = np.zeros(len(E[0][0]))
    for i in range(N):
        Ir = Ir + 0.5*(1/Z[i])*np.abs(c[i])**2 * (np.abs(E[0][i])**2 + np.abs(E[1][i])**2) # (1/(2*Z[i]))*
        
    return Ir 
        


def get_propagating_modes(a, b, freq, epsR, N):

    m = np.arange(0, N, 1)
    n = np.arange(0, N, 1)
    mv, nv = np.meshgrid(m, n)
    grid = np.stack((mv, nv), axis=-1)
    flat_grid = grid.reshape(-1, 2)
    sums = np.sum(flat_grid, axis=1)
    sorted_indices = np.argsort(sums)
    vmodes = flat_grid[sorted_indices][1:]  # Vector contraining modes. Sorted grid (may not be needed). [m,n]=[0,0] removed

    modes = {}

    for mode in vmodes:
        fc = cal_fc(a, b, mode, epsR)
        if freq > fc:
            modes[f'TE{mode[0]}{mode[1]}'] = [mode,fc]
            print(f'TE{mode[0]}{mode[1]} is propagating with fc = '+str(fc*1e-9)+' GHz\n')
            if mode[0] >= 1 and mode[1] >= 1:
                modes[f'TM{mode[0]}{mode[1]}'] = [mode,fc]
                print(f'TM{mode[0]}{mode[1]} is propagating with fc = '+str(fc*1e-9)+' GHz\n')
    modes = dict(sorted(modes.items(), key=lambda x: x[1][1]))
    """
    for mode in vmodes:
        fc = cal_fc(a, b, mode, epsR)
        #if freq > fc:
        modes[f'TE{mode[0]}{mode[1]}'] = [mode,fc]
        #print(f'TE{mode[0]}{mode[1]} is propagating with fc = '+str(fc*1e-9)+' GHz\n')
        if mode[0] >= 1 and mode[1] >= 1:
            modes[f'TM{mode[0]}{mode[1]}'] = [mode,fc]
        #print(f'TM{mode[0]}{mode[1]} is propagating with fc = '+str(fc*1e-9)+' GHz\n')
    modes = dict(sorted(modes.items(), key=lambda x: x[1][1]))
    """
    return modes

def get_modes(a, b, freq, epsR, N):

    m = np.arange(0, N, 1)
    n = np.arange(0, N, 1)
    mv, nv = np.meshgrid(m, n)
    grid = np.stack((mv, nv), axis=-1)
    flat_grid = grid.reshape(-1, 2)
    sums = np.sum(flat_grid, axis=1)
    sorted_indices = np.argsort(sums)
    vmodes = flat_grid[sorted_indices][1:]  # Vector containing modes. Sorted grid (may not be needed). [m,n]=[0,0] removed
    
    modes = {}

    for mode in vmodes:
        fc = cal_fc(a, b, mode, epsR)
        #if freq > fc:
        modes[f'TE{mode[0]}{mode[1]}'] = [mode,fc]
        #print(f'TE{mode[0]}{mode[1]} with fc = '+str(fc*1e-9)+' GHz\n')
        if mode[0] >= 1 and mode[1] >= 1:
            modes[f'TM{mode[0]}{mode[1]}'] = [mode,fc]
            #print(f'TM{mode[0]}{mode[1]} with fc = '+str(fc*1e-9)+' GHz\n')
    modes = dict(sorted(modes.items(), key=lambda x: x[1][1]))
    #TE_modes = [modes[key][0] for key in modes.keys() if 'TE' in key]
    #TM_modes = [modes[key][0] for key in modes.keys() if 'TM' in key]
    
    return modes

def list_modes(modes):
    for key,value in modes.items():
        print(key+' with fc = '+str(value[-1]*1e-9)+' GHz\n')
        

def cal_intensity(Etot):
    I = np.sqrt(np.abs(Etot[0])**2 + np.abs(Etot[1])**2)**2# + np.abs(Etot[2])**2)
    return I

def get_complex_set(N): # Function to generate a normalized set of complex numbers

    # Generate N random complex numbers with real and imaginary parts between -1 and 1
    z = np.random.uniform(-1, 1, (N,)) + 1j * np.random.uniform(-1, 1, (N,))

    # Calculate the sum of the norm squared of all N complex numbers
    S = np.sum(np.abs(z)**2)

    # Normalize each complex number
    z_normalized = z / np.sqrt(S)
    print('Normalized set of complex numbers: \n'+str(z_normalized))
    
    return z_normalized

def get_field(a,b,freq,epsR,X,Y,modes,plot):
    
    Ex = []
    Ey = []
    Ez = []
    
    Hx = []
    Hy = []
    Hz = []
    
    for key in modes.keys(): 
        if "TE" in key:
            E, H = cal_TE_modes(a, b, freq, epsR, X, Y, modes[key][0], plot)
            
            Ex.append(E[0])
            Ey.append(E[1])
            Ez.append(E[2])
            
            Hx.append(H[0])
            Hy.append(H[1])
            Hz.append(H[2])
            
        if 'TM' in key: 
            E, H = cal_TM_modes(a, b, freq, epsR, X, Y, modes[key][0], plot)
            
            Ex.append(E[0])
            Ey.append(E[1])
            Ez.append(E[2])
            
            Hx.append(H[0])
            Hy.append(H[1])
            Hz.append(H[2])
            
    E = np.array([Ex,Ey,Ez])
    H = np.array([Hx,Hy,Hz])
    
    return E,H

def get_random_field(E, H,nr_modes):
    
    c = get_complex_set(nr_modes)
    Ex = np.tensordot(c,E[0],axes=1)
    Ey = np.tensordot(c,E[1],axes=1)
    Ez = np.tensordot(c,E[2],axes=1)
    
    Hx = np.tensordot(c,H[0],axes=1)
    Hy = np.tensordot(c,H[1],axes=1)
    
    return np.array([Ex,Ey,Ez]), np.array([Hx,Hy]), c

def get_initial_guess(nr_modes):
    # default guess zero
    c0 = np.zeros(2*nr_modes)
    c0 = np.reshape(c0,(nr_modes,-1))
    return c0

def get_even_initial_guess(nr_modes):
    angles = np.random.rand(nr_modes) * 2 * np.pi  # random phase angles
    c0 = np.exp(1j * angles)  # array of complex numbers with unit amplitude
    c0 = c0 / np.linalg.norm(c0)
    c0_re = np.real(c0)
    c0_im = np.imag(c0)
    c0 = np.array([c0_re,c0_im])
    c0 = np.reshape(c0,(nr_modes,-1))
    return c0

def get_random_initial_guess(nr_modes):
    c0 = get_complex_set(nr_modes)
    c0_re = np.real(c0)
    c0_im = np.imag(c0)
    c0 = np.array([c0_re,c0_im])
    c0 = np.reshape(c0,(nr_modes,-1))
    return c0

def plot_mode_content(modes, c_opt):
    height = [np.abs(c)**2 for c in c_opt]
    keys = [key for key in modes.keys()]
    plt.bar(keys, height)
    plt.ylabel('$|c_n|^2$')
    plt.ylim((0,1))

def normalize_field(E):
    norm_factor = np.max(np.abs(E))
    if norm_factor == 0:
        E_norm = E
    else: 
        E_norm = E/norm_factor
    return E_norm

def create_files(E, X, Y, filename):
    flat_x = np.ravel(X)
    flat_y = np.ravel(Y)
    flat_z = flat_x*0
    # Extract both real and imaginary part of all field components:
    
    # Ex:
    data_Ex_real = np.column_stack((flat_x, flat_y, flat_z, np.ravel(np.real(E[0]))))
    data_Ex_imag = np.column_stack((flat_x, flat_y, flat_z,np.ravel(np.imag(E[0]))))
    np.savetxt(filename+'_Ex_real'+'.txt', data_Ex_real, delimiter='\t')
    np.savetxt(filename+'_Ex_imag'+'.txt', data_Ex_imag, delimiter='\t')
    
    # Ey: 
    data_Ey_real = np.column_stack((flat_x, flat_y, flat_z, np.ravel(np.real(E[1]))))
    data_Ey_imag = np.column_stack((flat_x, flat_y, flat_z, np.ravel(np.imag(E[1]))))
    np.savetxt(filename+'_Ey_real'+'.txt', data_Ey_real, delimiter='\t')
    np.savetxt(filename+'_Ey_imag'+'.txt', data_Ey_imag, delimiter='\t')
    
    # Ez:
    data_Ez_real = np.column_stack((flat_x, flat_y, flat_z, np.ravel(np.real(E[2]))))
    data_Ez_imag = np.column_stack((flat_x, flat_y, flat_z, np.ravel(np.imag(E[2]))))
    np.savetxt(filename+'_Ez_real'+'.txt', data_Ez_real, delimiter='\t')
    np.savetxt(filename+'_Ez_imag'+'.txt', data_Ez_imag, delimiter='\t')
    
    
def plot_field_components_vec(Es,Eopt,X,Y,x,y,T,filename):
    
    # Interpolate simulated field components: 
    Ex_re = griddata((x,y),np.real(Es[0]),(X,Y),fill_value = 0, method = 'cubic')
    Ex_im = griddata((x,y),np.imag(Es[0]),(X,Y),fill_value = 0, method = 'cubic')
    
    Ey_re = griddata((x,y),np.real(Es[1]),(X,Y),fill_value = 0, method = 'cubic')
    Ey_im = griddata((x,y),np.imag(Es[1]),(X,Y),fill_value = 0, method = 'cubic')
    
    Ez_re = griddata((x,y),np.real(Es[2]),(X,Y),fill_value = 0, method = 'cubic')
    Ez_im = griddata((x,y),np.imag(Es[2]),(X,Y),fill_value = 0, method = 'cubic')
    #------------------------------  
    
    # Interpolate simulated field components: 
    Ex_re_opt = griddata((x,y),np.real(Eopt[0]),(X,Y),fill_value = 0, method = 'cubic')
    Ex_im_opt = griddata((x,y),np.imag(Eopt[0]),(X,Y),fill_value = 0, method = 'cubic')
    
    Ey_re_opt = griddata((x,y),np.real(Eopt[1]),(X,Y),fill_value = 0, method = 'cubic')
    Ey_im_opt = griddata((x,y),np.imag(Eopt[1]),(X,Y),fill_value = 0, method = 'cubic')
    
    Ez_re_opt = griddata((x,y),np.real(Eopt[2]),(X,Y),fill_value = 0, method = 'cubic')
    Ez_im_opt = griddata((x,y),np.imag(Eopt[2]),(X,Y),fill_value = 0, method = 'cubic')
    #------------------------------  
    
    labels = ['Re($E_x$)',
              'Im($E_x$)',
              'Re($E_y$)',
              'Im($E_y$)',
              'Re($E_z$)',
              'Im($E_z$)',
              'Re($E_x$)',
                        'Im($E_x$)',
                        'Re($E_y$)',
                        'Im($E_y$)',
                        'Re($E_z$)',
                        'Im($E_z$)']
    
    mm = 1e3
    
    field_data = np.array([Ex_re,
                           Ex_im,
                           Ey_re,
                           Ey_im,
                           Ez_re,
                           Ez_im,
                           Ex_re_opt,
                           Ex_im_opt,
                           Ey_re_opt,
                           Ey_im_opt,
                           Ez_re_opt,
                           Ez_im_opt])
    
    fig, axs = plt.subplots(2, 6)
    
    overall_min = np.min([np.min(vmin) for vmin in field_data.flat])
    overall_max = np.max([np.max(vmax) for vmax in field_data.flat])
    print(overall_min)
    print(overall_max)
    
    for i, ax in enumerate(axs.flat):
        # Plot the data
        pcm = ax.pcolormesh(X*mm,
                            Y*mm,
                            field_data[i], 
                            norm=colors.SymLogNorm(linthresh=0.0005, 
                                                   linscale=0.05, 
                                                   vmin=overall_min, 
                                                   vmax=overall_max, 
                                                   base=10),
                            shading='auto')
        ax.set_title(labels[i],fontdict=font)
        
        #ax.set_xlabel('x (mm)',fontdict=font)
        #ax.set_xlabel('y (mm)',fontdict=font)
        ax.set_aspect('equal')
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0)
    cbar = fig.colorbar(pcm, ax=axs[:,:],location='right', shrink = 0.5,ticks=[-0.01, 0, 0.01])
    #cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
    # Set the aspect ratio
    axs[0,0].set_ylabel('Simulation', rotation='horizontal', ha='right', fontdict = font)
    axs[1,0].set_ylabel('Reconstructed', rotation='horizontal', ha='right', fontdict = font)
    
    
    # Set labels
        
    # Adjust the layout of the subplots
    #fig.suptitle(T, fontfamily='Times New Roman')
    #fig.tight_layout()

    
    plt.savefig(filename + 'vec_comp' + '.svg',bbox_inches = 'tight',format='svg')
    plt.savefig(filename + 'vec_comp' + '.png',bbox_inches = 'tight',format='png')
    plt.show()
    
def plot_intensities(Is,I_fit,X,Y,filename):
    mm = 1e3
    diff = Is-I_fit
    vmax = np.max(Is)
    vmin = np.min(Is)
    fig, axs = plt.subplots(1, 3, layout='constrained')

    p1 = axs[0].pcolormesh(X*mm, Y*mm, Is, vmin=vmin, vmax=vmax)
    axs[0].set_title('$I_s$')
    axs[0].set_aspect('equal', 'box')
    axs[0].set_xlabel('x (mm)')
    axs[0].set_ylabel('y (mm)')
    #cbar1 = fig.colorbar(p1,ax=axs[0])

    p2 = axs[1].pcolormesh(X*mm, Y*mm, I_fit, vmin=vmin, vmax=vmax)
    axs[1].set_title('$I_{r}$')
    axs[1].set_aspect('equal', 'box')
    axs[1].set_xlabel('x (mm)')
    axs[1].set_ylabel('y (mm)')
    #cbar2 = fig.colorbar(p2,ax=axs[1])
    
    p3 = axs[2].pcolormesh(X*mm, Y*mm, diff, vmin=vmin, vmax=vmax)
    axs[2].set_title('$I_s$-$I_{r}$')
    axs[2].set_aspect('equal', 'box')
    axs[2].set_xlabel('x (mm)')
    axs[2].set_ylabel('y (mm)')
    cbar3 = fig.colorbar(p1,ax=axs,location='bottom',shrink=0.6)
    
    #fig.tight_layout()
    #plt.savefig(filename + 'I_comp' + '.svg',format='svg')
    #plt.savefig(filename + 'I_comp' + '.png',format='png')
    plt.show()

def plot_all_eigenmodes(E,X,Y,modes):
    mm = 1e3
    i=0
    for mode in modes.keys():
        
        fig, axs = plt.subplots(3, 2)

        axs[0,0].contourf(X*mm, Y*mm, np.real(E[0][i]), levels=30)
        axs[0,0].set_title('$Re\{E_x\}$')
        axs[0,0].set_aspect('equal', 'box')
        axs[0,0].set_xlabel('x (mm)')
        axs[0,0].set_ylabel('y (mm)')

        axs[0,1].contourf(X*mm, Y*mm, np.imag(E[0][i]), levels=30)
        axs[0,1].set_title('$Im\{E_x\}$')
        axs[0,1].set_aspect('equal', 'box')
        axs[0,1].set_xlabel('x (mm)')
        axs[0,1].set_ylabel('y (mm)')
        
        axs[1,0].contourf(X*mm, Y*mm, np.real(E[1][i]), levels=30)
        axs[1,0].set_title('$Re\{E_y\}$')
        axs[1,0].set_aspect('equal', 'box')
        axs[1,0].set_xlabel('x (mm)')
        axs[1,0].set_ylabel('y (mm)')

        axs[1,1].contourf(X*mm, Y*mm, np.imag(E[1][i]), levels=30)
        axs[1,1].set_title('$Im\{E_y\}$')
        axs[1,1].set_aspect('equal', 'box')
        axs[1,1].set_xlabel('x (mm)')
        axs[1,1].set_ylabel('y (mm)')

        axs[2,0].contourf(X*mm, Y*mm, np.real(E[2][i]), levels=30)
        axs[2,0].set_title('$Re\{E_z\}$')
        axs[2,0].set_aspect('equal', 'box')
        axs[2,0].set_xlabel('x (mm)')
        axs[2,0].set_ylabel('y (mm)')

        axs[2,1].contourf(X*mm, Y*mm, np.imag(E[2][i]), levels=30)
        axs[2,1].set_title('$Im\{E_z\}$')
        axs[2,1].set_aspect('equal', 'box')
        axs[2,1].set_xlabel('x (mm)')
        axs[2,1].set_ylabel('y (mm)')
        
        fig.suptitle(mode)
        fig.tight_layout()
        plt.show()
        
        i += 1
        
    


