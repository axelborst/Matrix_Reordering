# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 08:10:50 2023

@author: aborst
"""

"""

The algorithm from
'A FAST & EFFECTIVE HEURISTIC FOR THE FEEDBACK ARC SET PROBLEM'
article
Peter Eades
Xuemin Lin
W. F. Smyth

"""
import random
import scipy as scipy
import numpy as np
import matplotlib.pyplot as plt
import sort_library_NEW as sl
import networkx as nx

mylw=2
titlesize=8
numbersize=6
legendsize=6
labelsize=7

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'

plt.rc('xtick',labelsize=6)
plt.rc('ytick',labelsize=6)

direct = 'SIPaper_data/'

save_switch=1

def setmyaxes(myxpos,myypos,myxsize,myysize):
    
    ax=plt.axes([myxpos,myypos,myxsize,myysize])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def algorithm1(GG):
    
    G = GG.copy()

    S1 = []
    S2 = [] #REVERSED!!!!


    while G.nodes:
        
        A = list(G.out_degree(G.nodes()))
        B = list(G.in_degree(G.nodes()))

        for u in A:
            if u[1] == 0:
                S2.append(u[0])
                G.remove_node(u[0])
                A.remove(u)
                for i in B:
                    if i[0] == u[0]:
                        B.remove(i)

                #dictionary changed size during iteration

        for u in B:
            if u[1] == 0:
                S1.append(u[0])
                G.remove_node(u[0])
                B.remove(u)
                for i in A:
                    if i[0] == u[0]:
                        A.remove(i)


        delta_max = 0
        node_to_delete = -1

        " May be for non weighted "

        for out_deg in A:
            for in_deg in B:
                if out_deg[0] == in_deg[0]:
                    a = out_deg[0]
                    b = (out_deg[1]-in_deg[1])
                    #                delta.append((a, b))
                    if b >= delta_max:
                        delta_max = b
                        node_to_delete = a


        for u in G.nodes:
            if G.out_degree(u)-G.in_degree(u) > delta_max:
                delta_max = G.out_degree(u)-G.in_degree(u)
                node_to_delete = u


        if node_to_delete != -1:
            S1.append(node_to_delete)
            G.remove_node(node_to_delete)

    S2.reverse()
    S1.extend(S2)

    remove=[]

    for u in GG.edges():
        
        a = u[0]
        b = u[1]
        if S1.index(a) >= S1.index(b):
            remove.append(u)

    GG.remove_edges_from(remove)

    return GG

def algorithm2(G):

    S = list(G.nodes())
    l = 1
    u = G.number_of_nodes()
    pi = []

    #while l!=n:
    for i in S[:]:
        w_out = 0
        w_in = 0
        S.remove(i)
        for j in S:
            if (j, u) in G.edges():
                w_out += 1

        for j in S:
            if (j, i) in G.edges():
               w_in += 1

        if w_in <= w_out:
            pi.append((i, l))
            l += 1
        else:
            pi.append((i, u))
            u -= 1



    remove = []
    for u in G.edges():
        a = u[0]
        b = u[1]
        a_pi = -1
        b_pi = -1
        for c in pi:
            if c[0] == a:
                a_pi = c[1]

            if c[0] == b:
                b_pi = c[1]

        if a_pi >= b_pi:
            remove.append(u)

    G.remove_edges_from(remove)

    return G

def algorithm3(G):
    
    V1 = []
    V2 = []
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()
    E1 = []
    E2 = []

    for u in G.nodes():
        random.choice([V1, V2]).append(u)

    G1.add_nodes_from(V1)
    G2.add_nodes_from(V2)

    for u in G.edges():
        if u[0] in V1 and u[1] in V1:
            E1.append(u)

        elif (u[0] in V2 and u[1] in V2):
            E2.append(u)


    G1.add_edges_from(E1)
    G2.add_edges_from(E2)


    G1 = algorithm2(G1)
    G2 = algorithm2(G2)

    G_fin1 = nx.DiGraph()
    G_fin2 = nx.DiGraph()

    G_fin1.add_nodes_from(G1)
    G_fin1.add_nodes_from(G2)
    G_fin2.add_nodes_from(G1)
    G_fin2.add_nodes_from(G2)

    G_fin1.add_edges_from(G1.edges)
    G_fin1.add_edges_from(G2.edges)
    G_fin2.add_edges_from(G1.edges)
    G_fin2.add_edges_from(G2.edges)


    for u in G.edges():
        if u[0] in V1 and u[1] in V2:
            G_fin1.add_edge(u[0], u[1])
        elif u[0] in V2 and u[1] in V1:
            G_fin2.add_edge(u[0], u[1])


    if G_fin1.number_of_edges() > G_fin2.number_of_edges():
        return G_fin1

    else:
        return G_fin2

    
def go_loop(choice=1,mydim=50,ff_trld=0.8,fb_trld=0.96):
    
    nofiter = 10
    
    fractions = np.zeros((nofiter,3))
    
    for i in range(nofiter):
    
        scrlist=np.random.permutation(mydim)
        
        oriM = sl.init_lower_M(mydim,ff_trld=ff_trld)
        scrM = sl.renumberM(oriM,scrlist)
        
        GG = nx.DiGraph(np.transpose(scrM))
        
        if choice == 1:  redG = algorithm1(GG)
        if choice == 2:  redG = algorithm2(GG)
        if choice == 3:  redG = algorithm3(GG)
    
        redM = np.transpose(nx.adjacency_matrix(redG).toarray())
        
        SchM,U = scipy.linalg.schur(redM)
        SchM   = np.transpose(SchM)
        pi     = sl.create_pi(np.transpose(U))
        srtM   = np.transpose(sl.renumberM(scrM,pi))
        
        sisM = sl.SI_sort(scrM)
        
        fraction_oriM = np.sum(np.triu(oriM,1))/np.sum(oriM)
        fraction_srtM = np.sum(np.triu(srtM,1))/np.sum(srtM)
        fraction_sisM = np.sum(np.triu(sisM,1))/np.sum(sisM)
        
        fractions[i]=np.array([fraction_oriM,fraction_srtM,fraction_sisM])
        
        density = np.sum(oriM)/(mydim*mydim)
        
        print('density = ',density*100,'%')
        
        print('fraction oriM = ', format(fraction_oriM*100,'.2f'),'%')
        print('fraction srtM = ', format(fraction_srtM*100,'.2f'),'%')
        print('fraction sisM = ', format(fraction_sisM*100,'.2f'),'%')
        
    mean_fraction = np.mean(fractions,axis=0)
    std_fraction  = np.std(fractions,axis=0)
    
    plt.bar(np.arange(3),mean_fraction*100,yerr=std_fraction*100,color=('grey','red','green'))
    plt.xticks(np.arange(3),('original','FAS-sorted','SI-sorted'))
    plt.ylabel('fraction of recurrent synapses')
    
    fname = 'ff_trld_'+str(ff_trld)+'_mean_fraction.npy'
    np.save(direct+fname,mean_fraction)
    
    fname = 'ff_trld_'+str(ff_trld)+'_std_fraction.npy'
    np.save(direct+fname,std_fraction)

def go(choice=1,mydim=50,ff_trld=0.5,fb_trld=0.96):
    
    scrlist=np.random.permutation(mydim)
    
    oriM = sl.init_lower_M(mydim,ff_trld=ff_trld)
    scrM = sl.renumberM(oriM,scrlist)
    
    GG = nx.DiGraph(np.transpose(scrM))
    
    if choice == 1:  redG = algorithm1(GG)
    if choice == 2:  redG = algorithm2(GG)
    if choice == 3:  redG = algorithm3(GG)

    redM = np.transpose(nx.adjacency_matrix(redG).toarray())
    
    SchM,U = scipy.linalg.schur(redM)
    SchM   = np.transpose(SchM)
    pi     = sl.create_pi(np.transpose(U))
    srtM   = np.transpose(sl.renumberM(scrM,pi))
    
    sisM = sl.SI_sort(scrM)
    
    print('oriM: total number of edges:', np.sum(oriM))
    print('oriM: number of recur edges:', np.sum(np.triu(oriM,1))) 
    print('srtM: number of rmovd edges:', np.sum(oriM)-np.sum(redM))
    
    plt.figure(figsize=(7.5,10))
    
    boxsize=0.22
    boxypos=0.70
    
    setmyaxes(0.1,boxypos,boxsize,boxsize)
    
    plt.imshow(oriM)
    plt.axis('off')
    plt.title('original',fontsize=titlesize)
    
    setmyaxes(0.4,boxypos,boxsize,boxsize)
    
    plt.imshow(scrM)
    plt.axis('off')
    plt.title('scrambled',fontsize=titlesize)
    
    setmyaxes(0.7,boxypos,boxsize,boxsize)
    
    plt.imshow(redM)
    plt.axis('off')
    plt.title('FAS reduced',fontsize=titlesize)
    
    boxypos=0.48
    
    setmyaxes(0.1,boxypos,boxsize,boxsize)
    
    plt.imshow(SchM)
    plt.axis('off')
    plt.title('Schur(FAS)',fontsize=titlesize)
    
    setmyaxes(0.4,boxypos,boxsize,boxsize)
    
    plt.imshow(srtM)
    plt.axis('off')
    plt.title('FAS sorted',fontsize=titlesize)
    
    setmyaxes(0.7,boxypos,boxsize,boxsize)
    
    plt.imshow(sisM)
    plt.axis('off')
    plt.title('SI sorted',fontsize=titlesize)
    
    boxypos  = 0.29
    boxxsize = 0.22
    boxysize = 0.16

    for i in range(3):
        
        ff_trld = np.array([0.5,0.65,0.8])
        
        boxxpos = 0.1+i*0.3
        
        fname = 'ff_trld_'+str(ff_trld[i])+'_mean_fraction.npy'
        mean_fraction = np.load(direct+'FAS Comparison/'+fname)
        
        fname = 'ff_trld_'+str(ff_trld[i])+'_std_fraction.npy'
        std_fraction = np.load(direct+'FAS Comparison/'+fname)
        
        setmyaxes(boxxpos,boxypos,boxxsize,boxysize)
    
        plt.bar(np.arange(3),mean_fraction*100,yerr=std_fraction*100,width=0.5,color=('grey','red','blue'))
        plt.xticks(np.arange(3),('original','FAS-sorted','SI-sorted'))
        plt.ylabel('fraction of recurrent synapses [%]',fontsize=labelsize)
        
        mytext = r'$p_{lower} = $'+str(format(1-ff_trld[i],'.2f'))
        plt.text(0.4,22,mytext,fontsize=labelsize)
        
        plt.ylim(0,25)
        plt.xlim(-0.6,2.6)
    
    if save_switch==1:
        
        plt.savefig('Figure_4.tiff',dpi=300)
        
go()
    
