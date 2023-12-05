# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:07:30 2022

@author: aborst
"""

import numpy as np
import matplotlib.pyplot as plt
import sort_library_NEW as sl
from timeit import default_timer as timer

mydim=100

ff_trld=0.50
fb_trld=0.96
   
mylw=2
titlesize=8
numbersize=6
legendsize=6
labelsize=7

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'

plt.rc('xtick',labelsize=6)
plt.rc('ytick',labelsize=6)

jac = 1
save_switch=0

direct='SIPaper_Data/Recurr_CPU_time/'

def setmyaxes(myxpos,myypos,myxsize,myysize):
    
    ax=plt.axes([myxpos,myypos,myxsize,myysize])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    
def renumberM(M,inparray):
        
    newM=np.zeros_like(M)
    tmp=M[:,inparray]
    newM=tmp[inparray,:]
            
    return newM


def init_M(mydim):
    
    line=np.arange(1,mydim)
    col=np.arange(0,mydim-1)
    R=np.random.random((mydim,mydim))
    R[line,col]=1
    L=np.tril(((R-ff_trld)>0),-1)
    U=np.triu((R-fb_trld)>0)
    
    return (L+U)*1.0


def calc_recurrency(maxN,save_switch=1):
    
    maxnofruns = 50
    
    dim     = np.array([10,20,50,100,200,500,1000,2000,5000,10000],dtype=int)
    nofruns = np.array([50,50,20,20,10,10,5,2,1,1])
    
    recurrency=np.zeros((maxnofruns,10,3))
    CPU_time=np.zeros((maxnofruns,10,2))
    
    for j in range(maxN):
        
        mydim=dim[j]
        
        print('N =',mydim)
        
        if mydim < 50: 
            
            pauli_fac = 0.1
            
        else:
            
            pauli_fac = 1.0
        
    
        for i in range(nofruns[j]):
            
            print('run # ',i)
            
            M_OR=init_M(mydim)                                  # original
            
            scrlist=np.random.permutation(mydim)
            M_SC=sl.renumberM(M_OR,scrlist)                     # scrambled

            start=timer()
            M_OD=sl.OD_sort(M_SC)                               # out degree sorted
            end=timer()
            
            OD_time=1.0*(end - start)
            
            start=timer()
            M_SI=sl.SI_sort(M_SC,jac=jac,pauli_fac=pauli_fac)   # smooth index sorted
            end=timer()

            SI_time=1.0*(end - start)
            
            rec_OR=int(np.sum(np.triu(M_OR)))
            rec_OD=int(np.sum(np.triu(M_OD)))
            rec_SI=int(np.sum(np.triu(M_SI)))

            total_nofconnections=np.sum(M_OR)
            
            recurrency[i,j]=np.array([rec_OR,rec_OD,rec_SI])/(1.0*total_nofconnections)
            CPU_time[i,j]=np.array([OD_time,SI_time])
            
    
    if save_switch==1:
    
        np.save(direct+str(ff_trld)+'_recurrency.npy',recurrency)
        np.save(direct+str(ff_trld)+'_CPU_time.npy',CPU_time)
        
        
def calc_all():
    
    global ff_trld
    
    for j in range(3):
        
        print()
        
        ff_trld = 0.5 + 0.15*j
        
        print('ff_trld = ', ff_trld)
        
        print()
        
        calc_recurrency(10)
        
        
def plotmatrix(M,title):
    
    plt.imshow(M,interpolation=None)
    plt.axis('off')
    plt.title(title,fontsize=titlesize)   
    
    
def plot_all(M1,M2,M3,M4,ypos):

    m_size=0.1
    ydelta=0.05
    
    setmyaxes(0.04,ypos,m_size,m_size)
    plotmatrix(M1,'original')
    
    mytext = r'$p_{lower} = $'+str(format(1-ff_trld,'.2f'))
    plt.text(35,-20,mytext,fontsize=labelsize)
    
    setmyaxes(0.16,ypos,m_size,m_size)
    plotmatrix(M2,'scrambled')
    
    setmyaxes(0.28,ypos+ydelta,m_size,m_size)
    plotmatrix(M3,'OD-sorted',)
    
    setmyaxes(0.28,ypos-ydelta,m_size,m_size)
    plotmatrix(M4,'SI-sorted')
    
    
def plot_example(ypos):
    
    mydim = 50
    
    M_OR=init_M(mydim)                      # original
    
    scrlist=np.random.permutation(mydim)
    M_SC=sl.renumberM(M_OR,scrlist)         # scrambled

    M_OD=sl.OD_sort(M_SC)                   # out degree sorted

    M_SI=sl.SI_sort(M_SC,jac=jac)           # smooth index sorted
            
    plot_all(M_OR,M_SC,M_OD,M_SI,ypos)
    

def plot_fraction(ff_trld):
    
    if ff_trld == 0.50: up_limit = 0.3
    if ff_trld == 0.65: up_limit = 0.4
    if ff_trld == 0.80: up_limit = 0.5
    
    x=np.array([10,20,50,100,200,500,1000,2000,5000,10000])
    nofruns = np.array([50,50,20,20,10,10,5,2,1,1])
    
    fname = str(ff_trld)+'_recurrency.npy'
    
    recurrency = np.load(direct+fname)
    
    mean_rec = np.zeros((10,3))
    std_rec  = np.zeros((10,3))
    
    for i in range(10):
        
        for j in range(3):
        
            mean_rec[i,j] = np.mean(recurrency[0:nofruns[i],i,j])
            std_rec[i,j]  = np.std(recurrency[0:nofruns[i],i,j])
    
    low_rec  = mean_rec-std_rec
    high_rec = mean_rec+std_rec
    
    mycolor=['black','red','blue']
    myfillc=['lightgrey','#FFBBBB','lightblue']
    mylabel=['original','OD-sorted','SI-sorted']
    
    for i in range(3):
    
        plt.fill_between(x,low_rec[:,i],high_rec[:,i],color=myfillc[i])
        
    for i in range(3):
        
        plt.plot(x,mean_rec[:,i],color=mycolor[i],label=mylabel[i],linewidth=mylw)
        plt.scatter(x,mean_rec[:,i],color=mycolor[i],linewidth=1)
    
    plt.legend(loc=1,frameon=False,fontsize=legendsize)

    plt.yscale('linear')
    plt.xscale('log')
    plt.ylim(0.0,up_limit)
    plt.xlim(5,20000)
      
    plt.xlabel('# of neurons',fontsize=labelsize)
    plt.ylabel('fraction of recurrent synapses',fontsize=labelsize)
    

def plot_CPU_time(ff_trld):
    
    x=np.array([10,20,50,100,200,500,1000,2000,5000,10000])
    nofruns = np.array([50,50,20,20,10,10,5,2,1,1])
    
    fname = str(ff_trld)+'_CPU_time.npy'
    
    CPU_time = np.load(direct+fname)
    
    mean_CPU_time = np.zeros((10,2))
    std_CPU_time  = np.zeros((10,2)) 
    
    for i in range(10):
        
        for j in range(2):
        
            mean_CPU_time[i,j] = np.mean(CPU_time[0:nofruns[i],i,j])
            std_CPU_time[i,j]  = np.std(CPU_time[0:nofruns[i],i,j])
    
    low_CPU  = mean_CPU_time-std_CPU_time
    high_CPU = mean_CPU_time+std_CPU_time
    
    mycolor=['red','blue']
    myfillc=['#FFBBBB','lightblue']
    mylabel=['OD-sorted','SI-sorted']
    
    for i in range(2):
    
        plt.fill_between(x,low_CPU[:,i],high_CPU[:,i],color=myfillc[i])
        plt.plot(x,mean_CPU_time[:,i],color=mycolor[i],label=mylabel[i],linewidth=mylw)
        plt.scatter(x,mean_CPU_time[:,i],color=mycolor[i],linewidth=1)
    
    plt.legend(loc=4,frameon=False,fontsize=legendsize)
    
    plt.xlabel('# of neurons',fontsize=labelsize)
    plt.xscale('log')
    plt.xlim(5,20000)
    
    plt.ylabel('CPU time [s]',fontsize=labelsize)
    plt.yscale('log')
    plt.ylim(0.000001,10000)
    
    
def plot_figure():
    
    global ff_trld
    
    plt.figure(figsize=(7.5,10))
    
    xsize=0.21
    ysize=0.17
    
    # upper row
    
    ff_trld = 0.5
    
    plot_example(0.75)
    
    setmyaxes(0.47,0.72,xsize,ysize)
    plot_fraction(ff_trld)
    
    setmyaxes(0.76,0.72,xsize,ysize)
    plot_CPU_time(ff_trld)
    
    
    # middle row
    
    ff_trld = 0.65
    
    plot_example(0.50)
    
    setmyaxes(0.47,0.47,xsize,ysize)
    plot_fraction(ff_trld)
    
    setmyaxes(0.76,0.47,xsize,ysize)
    plot_CPU_time(ff_trld)
    
    # middle row
    
    ff_trld = 0.8
    
    plot_example(0.25)
    
    setmyaxes(0.47,0.22,xsize,ysize)
    plot_fraction(ff_trld)
    
    setmyaxes(0.76,0.22,xsize,ysize)
    plot_CPU_time(ff_trld)
    
    
    if save_switch==1:
        
        plt.savefig('Figure_3.tiff',dpi=300)
        
plot_figure()
      
        
        
    

    

            

            
            
        