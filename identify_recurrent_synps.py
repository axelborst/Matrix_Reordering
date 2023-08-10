# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:07:30 2022

@author: aborst
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
import blindschleiche_py3 as bs
import sort_library_NEW as sl

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'

filename='P:/My Python 3/Medulla Network/Connectivity Matrix Srini/avg_n_syn_paper.csv'
data=np.genfromtxt(filename,delimiter=',',dtype=None)

nofcells=65
myxval=100

connM=np.zeros((nofcells,nofcells))
ctype = ["" for x in range(nofcells)]

for i in range(nofcells):
    ctype[i]=data[i+1,0].decode()

for i in range(1,nofcells+1,1):
    for j in range(1,nofcells+1,1):
        connM[i-1,j-1]=float(data[i,j])
        
connM=np.transpose(connM)

# --------- plot-params -----------

fontsize_title=12

def calcAdjM(M,thr=0,mode='both'):
    
    mydim=M.shape[0]

    AdjM=np.zeros((mydim,mydim))
    
    if mode=='both':
    
        AdjM=1*(abs(M)>thr)
                    
    if mode=='exc':
    
        AdjM=1*(M>thr)
        
    if mode=='inh':
    
        AdjM=1*(M<(-thr))
                
    return AdjM 

M=calcAdjM(connM, thr=4)

def calc_synthetic_M(mydim,ff_trld=0.60,fb_trld=0.96):
    
    M=sl.init_lower_M(mydim,ff_trld=ff_trld, fb_trld=fb_trld)
    
    M[np.arange(mydim),np.arange(mydim)]=0
    
    nof_totl_synps = np.sum(M)
    nof_recu_synps = np.sum(np.triu(M))
    nof_ffwd_synps = np.sum(np.tril(M))
    nof_reci_synps = np.sum(M*np.transpose(M))
    
    print('# of totl synapses:', nof_totl_synps)
    print('# of recu synapses:', nof_recu_synps)
    print('# of ffwd synapses:', nof_ffwd_synps)
    print('# of reci synapses:', nof_reci_synps)
    
    plt.imshow(M)
    
    return M

def load_synthetic_M():
    
    M    = np.load('synthetic_M.npy')
    x    = np.load('random_vector.npy')
    scrM = sl.renumberM(M,x)
    
    nof_totl_synps = np.sum(M)
    nof_recu_synps = np.sum(np.triu(M))
    nof_ffwd_synps = np.sum(np.tril(M))
    nof_reci_synps = np.sum(M*np.transpose(M))
    
    print('# of totl synapses:', nof_totl_synps)
    print('# of recu synapses:', nof_recu_synps)
    print('# of ffwd synapses:', nof_ffwd_synps)
    print('# of reci synapses:', nof_reci_synps)
    
    plt.figure(figsize=(10,5))
    
    plt.subplot(1,2,1)
    
    plt.imshow(M)
    
    plt.subplot(1,2,2)
    
    plt.imshow(scrM)
    
    return M,scrM,x

def identify_rec_synapses(M,n, save_it = 0, fname = 'rec_synpses_synth_M.npy'):
    
    mydim = M.shape[0]
    
    rec_synM=np.zeros((mydim,mydim))
    
    for i in range(n):
        
        srtM,argsort_z=sl.SI_sort(M)
        recM=np.triu(srtM,1)
        
        print('Run ', i, ': nof rec synpases = ', np.sum(recM))
        
        rev_z=sl.reverse_permutation(argsort_z)
        rec_synM += sl.renumberM(recM,rev_z)
        
    rec_synM = rec_synM/(1.0*n)
    
    plt.imshow(rec_synM,cmap='Reds')
    plt.title('recurrent synapses', fontsize=fontsize_title)
    
    cbar = plt.colorbar()
    cbar.set_label('probability', rotation=90, fontsize=10)
    
    if save_it == 1:

        np.save(fname,rec_synM)
        
    return rec_synM



        
    
                
    
    
            

    

            

            
            
        