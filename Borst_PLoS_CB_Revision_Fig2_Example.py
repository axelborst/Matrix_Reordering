# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:07:30 2022

@author: aborst
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sort_library_NEW as sl
import time

mode='rec'
mydim=50
counter=0

save_switch=1

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'

titlesize=8
numbersize=6
legendsize=6
labelsize=7

plt.rc('xtick',labelsize=6)
plt.rc('ytick',labelsize=6)

z=np.arange(mydim)
z_bounds=[(0,mydim)]
for i in range(mydim-1): z_bounds.append((0,mydim))

x=np.arange(mydim)
search_list=np.zeros((20000,3+mydim))
    
if mode=='rec':
    oriM=sl.init_lower_M(mydim,fb_trld=0.96)
    tolerance = 1e-8
    if mydim >  500: tolerance = 1e-8
    if mydim > 1000: tolerance = 1e-9
    if mydim > 2000: tolerance = 1e-10
    if mydim > 5000: tolerance = 1e-11
    
if mode=='all':
    oriM=sl.init_stripe_M(mydim)
    oriM=sl.init_block_M(mydim)
    tolerance = 1e-14
    
scrlist=np.random.permutation(mydim)
corlist=sl.reverse_permutation(scrlist)
M=sl.renumberM(oriM,scrlist)

yval,xval = np.where(M != 0)

pauli_fac           = 20.0   
density             = np.sum(M)/((1.0*mydim)**2)
recurrency_weight   = 1.0/density
bandwidth_weight    = 1.0/((1.0*mydim)**2.0*density)
pauli_weight        = 1.0/((1.0*mydim)**2.0)*pauli_fac*1.0/(1.0+np.exp(-(mydim-50)))

def setmyaxes(myxpos,myypos,myxsize,myysize):
    
    ax=plt.axes([myxpos,myypos,myxsize,myysize])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def plotmatrix(M,title):
    
    plt.imshow(M)
    plt.axis('off')
    plt.title(title,fontsize=titlesize)   
    
def plot_all(M1,M2,M3,z,nofswitches):
    
    boxsize=0.22
    boxypos=0.70
    
    setmyaxes(0.1,boxypos,boxsize,boxsize)
    plotmatrix(M1,'original')
    setmyaxes(0.4,boxypos,boxsize,boxsize)
    plotmatrix(M2,'scrambled')
    setmyaxes(0.7,boxypos,boxsize,boxsize)
    plotmatrix(M3,'sorted')
    
    xsize=0.22
    ysize=0.15
    
    boxypos=0.52
    
    setmyaxes(0.1,boxypos,xsize,ysize)
    
    plt.plot(search_list[0:counter,0],label='total')
    plt.plot(search_list[0:counter,1],label='length')
    plt.plot(search_list[0:counter,2],label='pauli')
    plt.legend(loc=7,frameon=False,fontsize=labelsize)
    
    plt.ylabel('cost',fontsize=labelsize)
    plt.xlabel('iteration',fontsize=labelsize)
    plt.xlim(0,int(counter/3))
    plt.yscale('log')
    
    setmyaxes(0.4,boxypos,xsize,ysize)
    
    for i in range(mydim):
    
        plt.plot(search_list[0:counter,i+3])
    
    plt.ylabel('position z',fontsize=labelsize)
    plt.xlabel('iteration',fontsize=labelsize)
    plt.xlim(0,int(counter/3))
    
    setmyaxes(0.7,boxypos,xsize,ysize)
    
    plt.plot(nofswitches)
    plt.ylabel('# of switched places',fontsize=labelsize)
    plt.xlabel('iteration',fontsize=labelsize)
    plt.xlim(0,int(counter/3))
    
def calc_positions_switched():
    
    z_vals=search_list[0:counter,3:53]
    
    positions=np.zeros((counter,50))
    
    nofswitches=np.zeros(counter)
    
    for i in range(counter):
        
        positions[i]=np.argsort(z_vals[i])
        
    for i in range(1,counter):
        
        nofswitches[i]=np.where(positions[i]!=positions[i-1])[0].size
        
    return nofswitches
        
def create_rand_params():

    z=np.zeros(mydim)
    
    for i in range(mydim):
            
            z_mean  = (z_bounds[i][1]+z_bounds[i][0])/2.0
            z_range = (z_bounds[i][1]-z_bounds[i][0])/2.0
            z[i]    = z_mean+(np.random.rand()-0.5)*z_range
            
    return z

def alpha(x,dim):
    
    y = 1.0 / (1.0 + np.exp(-10.0 / dim * x ))
    
    return y 

def calc_rec_length(z):
    
    D       = z[xval]-z[yval]+1
    D_rect  = D * (D > 0)
    
    total_length = np.sum(alpha(D_rect,mydim)-0.5) / np.sum(M) * recurrency_weight
        
    return total_length

def calc_rec_gradient(z):
    
    dz=np.zeros(mydim)
        
    for i in range(mydim):
        
        D1 = (z-z[i]+1) 
        D2 = (z[i]-z+1) 
        
        D1_rect = D1 * (D1 > 0)
        D2_rect = D2 * (D2 > 0)
        
        dz1   = - np.sum(M[i,:]*alpha(D1,mydim)*(1-alpha(D1,mydim))*10.0/mydim*(D1_rect != 0))
        dz2   = + np.sum(M[:,i]*alpha(D2,mydim)*(1-alpha(D2,mydim))*10.0/mydim*(D2_rect != 0))
        
        dz[i] = dz1 + dz2
        
    dz = dz / np.sum(M) * recurrency_weight
    
    return dz

def calc_all_length(z):
    
    D = z[xval]-z[yval]+1
        
    total_length = np.sum(D**2) / np.sum(M) * bandwidth_weight
        
    return total_length

def calc_all_gradient(z):
    
    dz = np.zeros(mydim)
    
    for i in range(mydim):
        
        D1 = (z[i]-z+1) 
        D2 = (z-z[i]+1) 
        
        dz[i] = 2.0*(np.sum(M[i,:]*D1)-np.sum(M[:,i]*D2))
        
    dz = dz / np.sum(M) * bandwidth_weight
    
    return dz

def calc_pauli(z):
    
    z_sort = np.sort(z)
    pauli  = np.mean((z_sort-x)**2) * pauli_weight
    
    return pauli

def calc_pauli_gradient(z):
    
    z_positions = sl.reverse_permutation(np.argsort(z))
    
    pauli_gradient = 2.0*(z-z_positions)/(1.0*mydim) * pauli_weight
    
    return pauli_gradient

def calc_error(z):
        
    global counter
    
    if mode == 'rec':
        length = calc_rec_length(z)
        
    if mode == 'all':
        length = calc_all_length(z)
        
    pauli = calc_pauli(z)
    
    error = length + pauli

    counter+=1
    
    search_list[counter-1,0:3]=np.array([error,length,pauli])
    search_list[counter-1,3:3+mydim]=z
        
    return error

def calc_gradient(z):
    
    if mode == 'rec':
        
        length_gradient = calc_rec_gradient(z)
        
    if mode == 'all':
        
        length_gradient = calc_all_gradient(z)
        
    pauli_gradient = calc_pauli_gradient(z)
    
    gradient = length_gradient+pauli_gradient
        
    return gradient

def send_message(res):
    
    jac_norm = np.sqrt(np.sum(res.jac**2))
    
    print('Optimization Success   :', res.success)
    print('Optimization Report    :', res.message)
    print('Last Value of cost fct :', format(res.fun,'.5f'))
    print('Norm of Last Jacobian  :', format(jac_norm,'.1e'))
    print('Number of cost fct use :', res.nfev)
    print('Number of Jacobian cal :', res.njev)
    print()

def fit_params(jac=1): 
    
    a=time.time()
        
    method='L-BFGS-B'
    options = {'maxiter':2000}
    
    z=create_rand_params()
    
    if jac==0:

        res = minimize(calc_error, z, method=method, tol=tolerance, bounds=z_bounds, options=options)
        
    if jac==1:

        res = minimize(calc_error, z, jac=calc_gradient, method=method, tol=tolerance, bounds=z_bounds, options=options)
        
    z = res.x
    
    b=time.time()
    print()
    print('CPU time for N =', mydim, ':  ', format(b-a,'.5f'), ' sec')
          
    send_message(res)
        
    return z

def report_result_quality(z):
    
    srtM=sl.renumberM(M,np.argsort(z))
    
    if mode == 'rec':
    
        all_ori=np.sum(oriM)
        upp_ori=np.sum(np.triu(oriM))
        upp_srt=np.sum(np.triu(srtM))
        
        ratio_ori=upp_ori/(1.0*all_ori)*100.0
        ratio_srt=upp_srt/(1.0*all_ori)*100.0
        
        print('ratio ori: ',format(ratio_ori,'.2f'),' %')
        print('ratio srt: ',format(ratio_srt,'.2f'),' %')
        
    if mode == 'all':
        
        z=np.arange(mydim)
        
        yval,xval = np.where(oriM != 0)
        Diff      = z[xval]-z[yval]+1
        total     = np.mean(Diff**2)
        
        print('mean sqared length ori: ',format(total,'.2f'))
        
        yval,xval = np.where(srtM != 0)
        Diff      = z[xval]-z[yval]+1
        total     = np.mean(Diff**2)
        
        print('mean sqared length srt: ',format(total,'.2f'))
        
def plot_figure():
    
    global counter
    
    counter=0
    
    z=fit_params(jac=1)
        
    srtM=sl.renumberM(M,np.argsort(z))
    
    nofswitches = calc_positions_switched()
    
    plt.figure(figsize=(7.5,10))
            
    plot_all(oriM,M,srtM,z,nofswitches)
    
    if save_switch==1:
        
        plt.savefig('Figure_2.tiff',dpi=300)
    
    report_result_quality(z)
    
def show_movie(nofframes=30,startframe=0,step=100):
    
    movie=np.zeros((mydim,mydim,nofframes))
    
    for i in range(nofframes):
        
        plt.figure(figsize=(10,4))
        
        frame=startframe+i*step
        
        print(frame)
        
        setmyaxes(0.1,0.1,0.5,0.8)
        
        plt.plot(search_list[0:frame+1,0],label='total')
        plt.plot(search_list[0:frame+1,1],label='length')
        plt.plot(search_list[0:frame+1,2],label='pauli')
        plt.legend(loc=1,frameon=False)
        
        plt.ylabel('cost')
        plt.xlim(0,counter)
        plt.yscale('log')
        plt.ylim(0.001,10)
        
        setmyaxes(0.65,0.1,0.3,0.8)
        
        movie[:,:,i] = sl.renumberM(M,np.argsort(search_list[frame,3:]))
        plt.imshow(movie[:,:,i])
        plt.axis('off')
        plt.pause(0.01)
        plt.title('i='+str(frame))
        plt.savefig('movie pics/movie'+np.str(i)+'.jpeg')
        plt.close()
    
plot_figure()
    





    

            

            
            
        