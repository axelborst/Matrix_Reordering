# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:07:30 2022

@author: aborst
"""

import numpy as np
import matplotlib.pyplot as plt
import blindschleiche_py3 as bs
import sort_library_NEW as sl

# --------- general params --------

nofcells=65
myxval=100
thr=4

save_switch=0

# --------- plot-params -----------

fontsize_title  = 8
fontsize_legend = 8

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'

# --------------------------------

def read_data():
    
    M      = np.load('synthetic_M.npy')
    x      = np.load('random_vector.npy')
    scrM   = sl.renumberM(M,x)
    recuPM = np.load('rec_synpses_synth_M.npy')
    
    nof_totl_synps = np.sum(M)
    nof_recu_synps = np.sum(np.triu(M,1))
    nof_ffwd_synps = np.sum(np.tril(M))
    nof_reci_synps = np.sum(M*np.transpose(M))
    
    print('# of totl synapses:', nof_totl_synps)
    print('# of recu synapses:', nof_recu_synps)
    print('# of ffwd synapses:', nof_ffwd_synps)
    print('# of reci synapses:', nof_reci_synps)
    
    return M,scrM,recuPM,x

trld = 0.45

oriM,scrM,recuPM,x = read_data()
rev_x      = sl.reverse_permutation(x)
rev_recuPM = sl.renumberM(recuPM,rev_x)
reciM      = oriM*np.transpose(oriM)
fbM        = 1.0*(rev_recuPM>trld)
ffM        = oriM - fbM

correct    = fbM * np.triu(oriM,1)
therest    = oriM - fbM * np.triu(oriM,1)

nof_rec_correct = np.sum(correct)

percent_correct = nof_rec_correct / np.sum(np.triu(oriM,1))*100.0

print('correctly identified:'+np.str(percent_correct)+' %')
    
def test_propagation(myM,nofsteps):
    
    nofcells = myM.shape[0]
    
    inp=np.zeros(nofcells)
    inp[0]=1.0
    
    activity=np.zeros((nofsteps,nofcells))
    
    for i in range(nofsteps):
        
        activity[i]=1.0*inp
        
        inp=np.dot(myM,1.0*inp)>0
  
    plt.imshow(bs.rebin(activity,50,50),interpolation=None)
    plt.xticks(np.arange(10)*5,'')
    plt.ylabel('n of steps',fontsize=fontsize_legend)
    
def check_reci_synps():
    
    yval,xval = np.where(reciM != 0)
    
    n = yval.size
    
    for i in range(n):
        
        print(xval[i],yval[i],rev_recuPM[yval[i],xval[i]])
        print(yval[i],xval[i],rev_recuPM[xval[i],yval[i]])
        print()
    
def plot_figure():
    
    plt.figure(figsize=(7.5,10))
    
    xsize=0.22
    ysize=0.19
    
    cbar_fac=1.08
    
    xpos=np.array([0.1,0.4,0.7,0.1,0.4,0.7])-0.02
    ypos=np.array([0.35,0.35,0.35,0.10,0.10,0.10])
    
    def draw_box():
        
        plt.plot([-0.5,49.5],[-0.5,-0.5],color='black')
        plt.plot([-0.5,49.5],[49.5,49.5],color='black')
        plt.plot([-0.5,-0.5],[-0.5,49.5],color='black')
        plt.plot([49.5,49.5],[-0.5,49.5],color='black')
        
    # ---------------------------------------
    
    bs.setmyaxes(xpos[0],ypos[0],xsize,ysize)

    plt.imshow(oriM)
    plt.title('original',fontsize=fontsize_title)
    plt.axis('off')
    draw_box()
    
    # ---------------------------------------
        
    bs.setmyaxes(xpos[1],ypos[1],xsize,ysize)
        
    plt.imshow(scrM)
    plt.title('scrambled',fontsize=fontsize_title)
    plt.axis('off')
    draw_box()
    
    # ---------------------------------------
    
    bs.setmyaxes(xpos[2],ypos[2],xsize*cbar_fac,ysize)
        
    plt.imshow(recuPM,cmap='Reds')
    plt.title('scrambled',fontsize=fontsize_title)
    plt.axis('off')
    draw_box()
    
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    
    cbar.set_ticks([0,0.5,1])
    cbar.set_ticklabels(['0.0','0.5','1.0'])
    cbar.ax.tick_params(labelsize=6) 
    cbar.set_label('recurrence probability', rotation=90, fontsize=6)
    
    # ---------------------------------------
    
    bs.setmyaxes(xpos[3],ypos[3],xsize*cbar_fac,ysize)

    plt.imshow(rev_recuPM,cmap='Reds')
    plt.title('original',fontsize=fontsize_title)
    plt.axis('off')
    draw_box()
    
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    
    cbar.set_ticks([0,0.5,1])
    cbar.set_ticklabels(['0.0','0.5','1.0'])
    cbar.ax.tick_params(labelsize=6) 
    cbar.set_label('recurrence probability', rotation=90, fontsize=6)
    
    # row 2
    
    # ---------------------------------------
    
    bs.setmyaxes(xpos[4],ypos[4],xsize*cbar_fac,ysize)
        
    plt.imshow(reciM*(rev_recuPM-0.5),cmap='coolwarm')
    plt.title('reciprocal synapses',fontsize=fontsize_title)
    plt.axis('off')
    draw_box()
    
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    
    cbar.set_ticks([-0.5,0,0.5])
    cbar.set_ticklabels(['0.0','0.5','1.0'])
    cbar.ax.tick_params(labelsize=6) 
    cbar.set_label('recurrence probability', rotation=90, fontsize=6)
    
    # ---------------------------------------
        
    bs.setmyaxes(xpos[5],ypos[5],xsize*cbar_fac,ysize)
        
    plt.imshow(fbM-ffM,vmin=-1,vmax=1,cmap='coolwarm')
    plt.title('all synapses',fontsize=fontsize_title)
    plt.axis('off')
    draw_box()
    
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    
    cbar.set_ticks([-1,-0.1,1])
    cbar.set_ticklabels(['forw','trld','recu'])
    cbar.ax.tick_params(labelsize=6) 
    cbar.set_label('classification', rotation=90, fontsize=6)
    
    if save_switch == 1:
        
        plt.savefig('Borst_Denk_NBC_Fig2CDEFGH.tiff',dpi=300)

plot_figure()

       
        
        
    
                
    
    
            

    

            

            
            
        