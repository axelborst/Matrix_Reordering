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
save_switch=1

direct='SIPaper_Data/'

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

def plotmatrix(M,title):
    
    plt.imshow(M,interpolation=None)
    plt.axis('off')
    plt.title(title,fontsize=titlesize)   
    
def plot_all(M1,M2,M3,i):
    
    N=M1.shape[0]
    
    m_box=0.09

    setmyaxes(0.12,0.82-i*0.08,m_box,m_box)
    
    plt.text(-0.7*N,N/2,'N='+np.str(N),fontsize=numbersize)
    
    if i==0: 
        plotmatrix(M1,'original')
    else:
        plotmatrix(M1,'')
        
    setmyaxes(0.23,0.82-i*0.08,m_box,m_box)
    if i==0: 
        plotmatrix(M2,'scrambled')
    else:
        plotmatrix(M2,'')
        
    setmyaxes(0.34,0.82-i*0.08,m_box,m_box)
    if i==0: 
        plotmatrix(M3,'SI-sorted')
    else:
        plotmatrix(M3,'')
        
def calc_recurrency(maxN):
    
    # maxN = 10
    
    dim=np.array([10,20,50,100,200,500,1000,2000,5000,10000],dtype=int)
    
    recurrency=np.zeros((dim.shape[0],4))
    CPU_time=np.zeros((dim.shape[0],2))
    
    for i in range(maxN):
        
        mydim=dim[i]
        
        print('N =',mydim)
        
        M_OR=init_M(mydim)                      # original
        scrlist=np.random.permutation(mydim)
        M_SC=sl.renumberM(M_OR,scrlist)         # scrambled

        start=timer()
        M_OD=sl.OD_sort(M_SC)                   # out degree sorted
        end=timer()
        
        OD_time=1.0*(end - start)
        
        start=timer()
        M_SI=sl.SI_sort(M_SC,jac=jac)           # smooth index sorted
        end=timer()

        SI_time=1.0*(end - start)
        
        print(OD_time,SI_time)
        
        rec_OR=int(np.sum(np.triu(M_OR)))
        rec_SC=int(np.sum(np.triu(M_SC)))
        rec_OD=int(np.sum(np.triu(M_OD)))
        rec_SI=int(np.sum(np.triu(M_SI)))

        total_nofconnections=np.sum(M_OR)
        
        recurrency[i]=np.array([rec_OR,rec_SC,rec_OD,rec_SI])/(1.0*total_nofconnections)
        CPU_time[i]=np.array([OD_time,SI_time])
        
        plt.pause(0.1)
        
    return recurrency,CPU_time

def do_many_runs(maxN,nofruns=10,save_switch=1):
    
    recurrency=np.zeros((nofruns,10,4))
    CPU_time=np.zeros((nofruns,10,2))
    
    for i in range(nofruns):
        
        print(i)
        recurrency[i],CPU_time[i]=calc_recurrency(maxN)
        
    mean_recurrency=np.mean(recurrency,axis=0)
    std_recurrency=np.std(recurrency,axis=0)
    
    mean_CPU_time=np.mean(CPU_time,axis=0)
    std_CPU_time=np.std(CPU_time,axis=0)
    
    if save_switch==1:
    
        np.save(direct+'Fig3_mean_recurrency.npy',mean_recurrency)
        np.save(direct+'Fig3_std_recurrency.npy',std_recurrency)
        np.save(direct+'Fig3_mean_CPU_time.npy',mean_CPU_time)
        np.save(direct+'Fig3_std_CPU_time.npy',std_CPU_time)
        
def create_data_Fig3A():
    
    global mydim,M
    
    dim=np.array([10,50,100,500,1000,5000,10000],dtype=int)
    
    for i in range(7):
        
        print(i)
        
        mydim=dim[i]
        
        oriM=init_M(mydim)
        scrlist=np.random.permutation(mydim)
        scrM=renumberM(oriM,scrlist)    
        srtM=sl.SI_sort(scrM,jac = jac)
        
        fname1=direct+'Fig3A_oriM_dim'+np.str(mydim)+'.npy'
        fname2=direct+'Fig3A_scrM_dim'+np.str(mydim)+'.npy'
        fname3=direct+'Fig3A_srtM_dim'+np.str(mydim)+'.npy'
        
        np.save(fname1,oriM)
        np.save(fname2,scrM)
        np.save(fname3,srtM)

def plot_Fig_2A():
    
    x=np.array([10,20,50,100,200,500,1000,2000,5000,10000])
    
    mean_rec=np.load(direct+'Fig3_mean_recurrency.npy')
    std_rec=np.load(direct+'Fig3_std_recurrency.npy')
    low_rec=mean_rec-std_rec
    high_rec=mean_rec+std_rec
    
    mycolor=['black','gray','red','blue']
    myfillc=['lightgrey','lightgrey','#FFBBBB','lightblue']
    mylabel=['original','scrambled','OD-sorted','SI-sorted']
    
    for i in range(4):
    
        plt.fill_between(x,low_rec[:,i],high_rec[:,i],color=myfillc[i])
        plt.plot(x,mean_rec[:,i],color=mycolor[i],label=mylabel[i],linewidth=mylw)
        plt.scatter(x,mean_rec[:,i],color=mycolor[i],linewidth=1)
    
    plt.legend(loc=4,frameon=False,fontsize=legendsize)

    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(0.01,1)
    plt.xlim(5,20000)
      
    plt.xlabel('# of neurons',fontsize=labelsize)
    plt.ylabel('fraction of recurrent synapses',fontsize=labelsize)
    
def plot_Fig_2A_performance():
    
    x=np.array([10,20,50,100,200,500,1000,2000,5000,10000])
    
    mean_rec=np.load(direct+'Fig3_mean_recurrency.npy')
    std_rec=np.load(direct+'Fig3_std_recurrency.npy')
    
    mean_perf = np.zeros((10,2))
    std_perf = np.zeros((10,2))
    
    for i in range(2):
        
        mean_perf[:,i]=mean_rec[:,0]/mean_rec[:,i+2]
        std_perf[:,i]=np.sqrt(std_rec[:,0]**2+std_rec[:,i+2]**2)
    
    low_perf=mean_perf-std_perf
    high_perf=mean_perf+std_perf
    
    mycolor=['red','blue']
    myfillc=['#FFBBBB','lightblue']
    mylabel=['OD-sorted','SI-sorted']
    
    for i in range(2):
    
        plt.fill_between(x,low_perf[:,i],high_perf[:,i],color=myfillc[i])
        plt.plot(x,mean_perf[:,i],color=mycolor[i],label=mylabel[i],linewidth=mylw)
        plt.scatter(x,mean_perf[:,i],color=mycolor[i],linewidth=1)
    
    plt.legend(loc=4,frameon=False,fontsize=legendsize)

    plt.xscale('log')
    plt.ylim(0,1.1)
    plt.xlim(5,20000)
      
    plt.xlabel('# of neurons',fontsize=labelsize)
    plt.ylabel('performance',fontsize=labelsize)

def plot_Fig_2B():
    
    x=np.array([10,20,50,100,200,500,1000,2000,5000,10000])
    mean_CPU_time=np.load(direct+'Fig3_mean_CPU_time.npy')
    std_CPU_time=np.load(direct+'Fig3_std_CPU_time.npy')
    low_CPU=mean_CPU_time-std_CPU_time
    high_CPU=mean_CPU_time+std_CPU_time
    
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
    
def plot_Fig_2AB():
    
    plt.figure(figsize=(7.5,10))
    
    xsize=0.37
    ysize=0.25
    
    setmyaxes(0.09,0.65,xsize,ysize)
    plot_Fig_2A_performance()
    
    setmyaxes(0.56,0.65,xsize,ysize)
    plot_Fig_2B()
    
    if save_switch==1:
        
        plt.savefig('Borst_Denk_NBC_Fig2AB_NEW.tiff',dpi=300)
        
plot_Fig_2AB()
      
        
        
    

    

            

            
            
        