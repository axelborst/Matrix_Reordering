# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 08:43:49 2018

@author: aborst
"""

import numpy as np
import matrixlibrary_py3 as mtx
import matplotlib.pyplot as plt

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
    
def add_arrow(x1,x2,y1,y2,mycolor):
    
    hl=0.5
    hw=0.5
    radius=0.5
    offs=radius+hl
    oril=np.sqrt((x2-x1)**2+(y2-y1)**2)
    ratio=(oril-offs)/oril*0.97
    dx=(x2-x1)*0.00001
    dy=(y2-y1)*0.00001
    x0=x1+ratio*(x2-x1)
    y0=y1+ratio*(y2-y1)
    plt.arrow(x0,y0,dx,dy,head_width=hw,head_length=hl,color=mycolor)

def add_blob(x1,y1,mycolor):
    
    plt.scatter(x1,y1,linewidth=5,color=mycolor)    
    plt.scatter(x1,y1,linewidth=10,color=mycolor)
    
def connectpoints(x1,y1,x2,y2,lw,mysign):
    
    radius=0.80
    
    if y1>y2: 
        amp=-2.0
        mycolor='grey'
    else:
        amp=+2.0
        mycolor='red'
        
    myx=x1+np.sin(np.linspace(0,np.pi,200))*amp
    myy=np.linspace(y1,y2,200)*1.0
        
    for i in range(199):
        dist1=np.sqrt((myx[i]-myx[0])**2+(myy[i]-myy[0])**2)
        dist2=np.sqrt((myx[i]-myx[196])**2+(myy[i]-myy[196])**2)
        if (dist1>radius and dist2>radius):
            plt.plot([myx[i],myx[i+1]],[myy[i],myy[i+1]],'k-',color=mycolor, linewidth=lw)

    add_arrow(myx[190],myx[196],myy[190],myy[196],mycolor)
    
def show_variable_circuit(mixed,box_xpos,box_ypos):
    
    setmyaxes(box_xpos,box_ypos,0.1,0.20)

    mydim=6    
    M=np.zeros((mydim,mydim))

    for i in range(mydim-1):
        M[i+1,i]=1
    
    M[1,5]=1
        
    newnumbers=np.arange(mydim)
    newlabels=['a','b','c','d','e','f']

    if mixed==1: 
        newnumbers=np.array([3,0,2,5,4,1])
        newlabels=['d','a','c','f','e','b']
        M=mtx.renumberM(M,newnumbers)
        
    var_y=np.array([0.51, 0.22, 0.77, 0.53, 0.89, 0.89])
    
    ypos=19.0-np.arange(mydim)*3.0+(var_y-0.5)*2.0
    
    xpos=np.arange(mydim)*0+17

    # plot all cells
                    
    for i in range(mydim):
        
        for mylw in range(5,15,5):
    
            plt.scatter(xpos[i],ypos[i],linewidth=mylw,color='blue')
            
        plt.text(xpos[i]+0.05,ypos[i]-0.1,newlabels[i],horizontalalignment='center',verticalalignment='center',fontsize=10, weight='bold',color='white')
        
    # draw connections
    
    mylw=2
    
    for pre in range(mydim):
        for post in range(mydim):
            if M[post,pre]==1:
                connectpoints(xpos[pre],ypos[pre],xpos[post],ypos[post],mylw,1)
    
    plt.xlim(14,20)
    plt.ylim(2,22)
    plt.axis('off')
    
def showcircuit(mixed,recurr,mytitle,box_ypos):
    
    setmyaxes(0.38,box_ypos,0.1,0.20)

    mydim=6    
    M=np.zeros((mydim,mydim))
    
    for i in range(mydim-1):
        M[i+1,i]=1
        
    if recurr==1: M[1,5]=1
        
    newnumbers=np.arange(mydim)
    newlabels=['a','b','c','d','e','f']

    if mixed==1: 
        newnumbers=np.array([3,0,2,5,4,1])
        newlabels=['d','a','c','f','e','b']
        M=mtx.renumberM(M,newnumbers)
    
    ypos=19.0-np.arange(mydim)*3.0
    xpos=np.arange(mydim)*0+17

    # plot all cells
                    
    for i in range(mydim):
        
        for mylw in range(5,15,5):
    
            plt.scatter(xpos[i],ypos[i],linewidth=mylw,color='blue')
            
        plt.text(xpos[i]+0.05,ypos[i]-0.1,newlabels[i],horizontalalignment='center',verticalalignment='center',fontsize=10, weight='bold',color='white')
        
    # draw connections
    
    mylw=2
    
    for pre in range(mydim):
        for post in range(mydim):
            if M[post,pre]==1:
                connectpoints(xpos[pre],ypos[pre],xpos[post],ypos[post],mylw,1)
    
    plt.xlim(14,20)
    plt.ylim(2,22)
    plt.axis('off')
    
    setmyaxes(0.1,box_ypos,0.22,0.19)
    
    plt.imshow(M)
    plt.xticks(np.arange(mydim),np.arange(mydim))
    plt.yticks(np.arange(mydim),np.arange(mydim))
    plt.title(mytitle,fontsize=titlesize)
    plt.tight_layout(rect=[0.1,0.1,0.9, 0.95])

def alpha(x,dim):
    
    y = 1.0 / (1.0 + np.exp(-10.0 / dim * x ))
    
    return y 

def show_alpha(box_xpos,box_ypos):
    
    box_xsiz=0.15
    box_ysiz=0.08
    
    setmyaxes(box_xpos,box_ypos,box_xsiz,box_ysiz)
    
    dim=50
    x=np.arange(2*dim+1)-dim
    y = alpha(x,dim)-0.5
    y=y*(y>0)
    plt.plot(x,y,linewidth=2,color='black')
    
    plt.xticks((np.arange(3)-1)*dim,'')
    plt.yticks(np.arange(6)*0.1,'')

        
def plot_figure():
    
    plt.figure(figsize=(7.5,10))
    
    showcircuit(mixed=1,recurr=1,mytitle='scrambled',box_ypos=0.70)
    showcircuit(mixed=0,recurr=1,mytitle='sorted',box_ypos=0.45)
    show_variable_circuit(mixed=1,box_xpos=0.6,box_ypos=0.70)
    show_alpha(box_xpos=0.78,box_ypos=0.76)
    
    if save_switch==1:
        
        plt.savefig('Figure_1.tiff',dpi=300)
    
plot_figure()
        

            
        
        



