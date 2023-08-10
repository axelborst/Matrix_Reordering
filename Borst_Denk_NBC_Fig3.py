# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:07:30 2022

@author: aborst
"""

import numpy as np
import matplotlib.pyplot as plt
import sort_library_NEW as sl

jac = 1

mylw=2
titlesize=8
numbersize=6
legendsize=6
labelsize=7

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'

plt.rc('xtick',labelsize=6)
plt.rc('ytick',labelsize=6)

save_switch=1

def setmyaxes(myxpos,myypos,myxsize,myysize):
    
    ax=plt.axes([myxpos,myypos,myxsize,myysize])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def renumberM(M,inparray):
        
    newM=np.zeros_like(M)
    tmp=M[:,inparray]
    newM=tmp[inparray,:]
            
    return newM

def init_stripe_M(mydim,trld=0.5):
    
    bandwidth=int(mydim*0.2)

    R=np.random.random((mydim,mydim))
    R=1.0*(R>trld)
    
    ones=np.ones((mydim,mydim))
    
    L=np.tril(ones,-bandwidth)
    U=np.triu(ones,bandwidth)
    
    stripe=1-(L+U)
    
    R=stripe*R
    
    line=np.arange(1,mydim)
    col=np.arange(0,mydim-1)
    R[line,col]=1
    
    return R

def init_tight_M(mydim,trld=0.5):
    
    R=np.random.random((mydim,mydim))
    
    Mzi=np.zeros((mydim,mydim))
    Mzj=np.zeros((mydim,mydim))
    
    for i in range(mydim):
        Mzi[i,:]=np.arange(mydim)+1
        Mzj[:,i]=np.arange(mydim)
        
    Dist=abs(Mzi-Mzj)
    Dist=1-Dist/(1.0*np.max(Dist))
    
    R=1.0*(R<Dist-trld)
    
    line=np.arange(1,mydim)
    col=np.arange(0,mydim-1)
    R[line,col]=1
    
    return R

def init_block_M(mydim,nofblocks=4,trld=0.5):
    
    R=np.random.random((mydim,mydim))
    
    Mask=np.zeros((mydim,mydim))
    
    step=int(mydim/nofblocks)
    
    for i in range(nofblocks):
        
        Mask[step*i:step*(i+1),step*i:step*(i+1)]=1
        
    R=(R>trld)*Mask
    
    line=np.arange(1,mydim)
    col=np.arange(0,mydim-1)
    R[line,col]=1
    
    return R

def plotmatrix(M,title):
    
    plt.imshow(M)
    plt.axis('off')
    plt.title(title,fontsize=titlesize)   
    
def plot_all(M1,M2,M3,M4,ypos):

    m_size=0.1
    ydelta=0.05
    
    setmyaxes(0.1,ypos,m_size,m_size)
    plotmatrix(M1,'original')
    
    setmyaxes(0.22,ypos,m_size,m_size)
    plotmatrix(M2,'scrambled')
    
    setmyaxes(0.34,ypos+ydelta,m_size,m_size)
    plotmatrix(M3,'RC-sorted',)
    
    setmyaxes(0.34,ypos-ydelta,m_size,m_size)
    plotmatrix(M4,'SI-sorted')
    
def calc_total_length_M(M):
    
    z         = np.arange(M.shape[0])
    yval,xval = np.where(M != 0)
    Diff      = z[xval]-z[yval]+1
    total     = np.mean(Diff**2)
            
    return total

def print_all(M_OR,M_SC,M_RC,M_BD):
    
    length_M_OR=calc_total_length_M(M_OR)
    length_M_SC=calc_total_length_M(M_SC)
    length_M_RC=calc_total_length_M(M_RC)
    length_M_BD=calc_total_length_M(M_BD)
    
    relBW_RC=length_M_RC/length_M_OR*100.0
    relBW_BD=length_M_BD/length_M_OR*100.0
    
    print()
    print('length_M_OR',length_M_OR)
    print('length_M_SC',length_M_SC)
    print('length_M_RC',length_M_RC)
    print('length_M_BD',length_M_BD)
    print()
    print('relative BW RC', format(relBW_RC,'.3f'),' [%]')
    print('relative BW BD', format(relBW_BD,'.3f'),' [%]')
    
def one_run(mydim,switch):
    
    if switch=='stripe': 
        M_OR=init_stripe_M(mydim)
        
    if switch=='tight': 
        M_OR=init_tight_M(mydim)
        
    if switch=='block':
        M_OR=init_block_M(mydim)
        
    scrlist=np.random.permutation(mydim)
    M_SC=renumberM(M_OR,scrlist)
    
    M_BD,x=sl.SI_sort(M_SC,mode='bandwidth',jac=jac)
    M_RC=sl.RC_sort(M_SC)
    
    length_M_OR=calc_total_length_M(M_OR)
    length_M_RC=calc_total_length_M(M_RC)
    length_M_BD=calc_total_length_M(M_BD)
    
    relBW_RC=length_M_RC/length_M_OR*100.0
    relBW_BD=length_M_BD/length_M_OR*100.0
    
    return relBW_RC, relBW_BD
    
def do_many_runs(nofruns,switch,save_switch=1):
    
    # switch is 'stripe',' 'tight' or 'block'
    
    dim=np.array([10,20,50,100,200,500,1000,2000,5000,10000])
    
    relBW_RC=np.zeros((nofruns,10))
    relBW_BD=np.zeros((nofruns,10))
    
    for i in range(nofruns):
        
        print()
        print('loop ',i)
        print()
        
        for j in range(10):
            
            print('dim=',dim[j])
        
            relBW_RC[i,j],relBW_BD[i,j]=one_run(dim[j],switch)
    
    mean_RC=np.mean(relBW_RC,axis=0)
    mean_BD=np.mean(relBW_BD,axis=0)
    
    std_RC=np.std(relBW_RC,axis=0)
    std_BD=np.std(relBW_BD,axis=0)
    
    direct='SIPaper_Data'
    
    if save_switch==1:
    
        np.save(direct+'/Fig4_mean_RC_'+switch+'.npy',mean_RC)
        np.save(direct+'/Fig4_std_RC_'+switch+'.npy',std_RC)
        np.save(direct+'/Fig4_mean_BD_'+switch+'.npy',mean_BD)
        np.save(direct+'/Fig4_std_BD_'+switch+'.npy',std_BD)
        
def calc_data_Fig3():
    
    do_many_runs(10,'stripe',save_switch=1)
    do_many_runs(10,'block',save_switch=1)
        
def plot_Fig_3AB(mydim):
    
    # ---------- stripe M -------------
    
    M_OR=init_stripe_M(mydim)
    scrlist=np.random.permutation(mydim)
    M_SC=renumberM(M_OR,scrlist)
    
    M_BD,x=sl.SI_sort(M_SC,mode='bandwidth',jac=jac) 
    M_RC=sl.RC_sort(M_SC)
            
    plot_all(M_OR,M_SC,M_RC,M_BD,0.75)
    
    print_all(M_OR,M_SC,M_RC,M_BD)
    
     # ---------- clustered M -------------
    
    M_OR=init_block_M(mydim)
    scrlist=np.random.permutation(mydim)
    M_SC=renumberM(M_OR,scrlist)
    
    M_BD,x=sl.SI_sort(M_SC,mode='bandwidth',jac=jac) 
    M_RC=sl.RC_sort(M_SC)
            
    plot_all(M_OR,M_SC,M_RC,M_BD,0.45)
    
    print_all(M_OR,M_SC,M_RC,M_BD)
    
    
def plot_Fig_3C(switch,myrange):
    
    dim=np.array([10,20,50,100,200,500,1000,2000,5000,10000])
    
    direct='SIPaper_Data'
    
    mean_RC=np.load(direct+'/Fig4_mean_RC_'+switch+'.npy')
    std_RC=np.load(direct+'/Fig4_std_RC_'+switch+'.npy')
    
    mean_SI=np.load(direct+'/Fig4_mean_BD_'+switch+'.npy')
    std_SI=np.load(direct+'/Fig4_std_BD_'+switch+'.npy')
    
    low_RC=mean_RC-std_RC
    high_RC=mean_RC+std_RC
    
    low_SI=mean_SI-std_SI
    high_SI=mean_SI+std_SI
    
    mycolor=['red','blue']
    myfillc=['#FFBBBB','lightblue']
    mylabel=['RC-sorted','SI-sorted']
    
    if myrange=='small':
        
        dim=np.array([100,200,500,1000,2000,5000,10000])
        
        mean_RC=mean_RC[3:10]
        low_RC=low_RC[3:10]
        high_RC=high_RC[3:10]
        
        mean_SI=mean_SI[3:10]
        low_SI=low_SI[3:10]
        high_SI=high_SI[3:10]
        
    plt.fill_between(dim,low_RC,high_RC,color=myfillc[0])
    plt.plot(dim,mean_RC,color=mycolor[0],label=mylabel[0],linewidth=mylw)
    plt.scatter(dim,mean_RC,color=mycolor[0],linewidth=1)
    
    plt.fill_between(dim,low_SI,high_SI,color=myfillc[1])
    plt.plot(dim,mean_SI,color=mycolor[1],label=mylabel[1],linewidth=mylw)
    plt.scatter(dim,mean_SI,color=mycolor[1],linewidth=1)
    
    plt.legend(loc=4,frameon=False,fontsize=legendsize)

    plt.xscale('log')
    
    if switch == 'stripe':
        plt.ylim(80,120)
    else:
        plt.ylim(0,200)
        
    plt.xlim(5,20000)
    
    if myrange=='small':plt.xlim(50,20000)
        
    plt.xlabel('# of neurons',fontsize=labelsize)
    plt.ylabel('relative bandwidth [% of original]',fontsize=labelsize)
    plt.title(switch+' matrix',fontsize=titlesize)
    
        
def plot_Fig_3():
    
    plt.figure(figsize=(7.5,10))
    
    plot_Fig_3AB(100)
    
    setmyaxes(.56,.68,.35,.23)
    plot_Fig_3C(switch='stripe',myrange='small')
    
    setmyaxes(.56,.38,.35,.23)
    plot_Fig_3C(switch='block',myrange='small')
    
    if save_switch==1:
        
        plt.savefig('Borst_Denk_NBC_Fig3.tiff',dpi=600)

plot_Fig_3()


            

            
            
        