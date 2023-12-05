# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:47:35 2018

@author: aborst
"""

import numpy as np
import scipy as scipy
from scipy import sparse
from scipy.optimize import minimize
import tarjan
import networkx as nx
import copy

# ------  matrix generation -----------------
#
# init_lower_M
# init_stripe_M
# init_tight_M
# init_block_M
#
# ------  matrix generation -----------------

def init_lower_M(mydim, ff_trld=0.50, fb_trld=0.96):
    
    line=np.arange(1,mydim)
    col=np.arange(0,mydim-1)
    R=np.random.random((mydim,mydim))
    R[line,col]=1
    L=np.tril(((R-ff_trld)>0),-1)
    U=np.triu((R-fb_trld)>0)
    
    return (L+U)*1.0

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

# ------------end of matrix generation --------------

# -------------permutation functions ----------------
# ---------------------------------------------------
# create_random_vector(dim)
# create_P(pi)
# create_pi(P)
# renumberM(M,inparray)
# swapcolsnrows(M,x,y)
# swapnumbers(a,x,y)
# reverse_permutation(inplist)
# countlength(M)
# sortlength(M)
# count_rcs(M)
# ---------------------------------------------------
# -------------permutation functions ----------------

def create_random_vector(dim):
    
    x=np.random.permutation(np.arange(dim))
    
    return x

def create_P(pi):
    
    dim=pi.shape[0]
    P=np.zeros((dim,dim))
    
    for i in range(dim):
        P[i,int(pi[i])]=1
        
    return P

def create_pi(P):
    
    x,y=np.where(P==1)
    
    return y
    
def renumberM(M,inparray):
        
    newM=np.zeros_like(M)
    tmp=M[:,inparray]
    newM=tmp[inparray,:]
            
    return newM
       
def swapcolsnrows(M,x,y):
    
    newM=1.0*M
    newM[:,[x,y]]=newM[:,[y,x]]
    newM[[x,y],:]=newM[[y,x],:]
    
    return newM
    
def swapnumbers(a,x,y):
    
    b=a.copy()
    b[x],b[y]=a[y],a[x]
    
    return b

def reverse_permutation(inplist):
    
    dim=inplist.shape[0]
    outlist=np.zeros(dim)
    outlist[inplist]=np.arange(dim)
    
    return outlist.astype(int)
    
def countlength(M):
    
    newM=np.triu(M)
            
    line,col=np.where(newM==1)
    length=0
    for i in range(len(line)):
        length+=np.abs(col[i]-line[i])
            
    return length

def count_rcs(M):
    
    # counts number of recurrent entries in upper triangle
    
    upper=np.triu(M)
    nofupper=np.sum(upper)
    
    return nofupper

#---------  end of permutation functions ------------------

# -------------- sort functions ---------------------------
# AB_sort(M)
# MP_sort(M) Matrix Power
# RC_sort(M) Reverse Cuthill McKee
# DF_sort(M) Depth First
# SS_sort(M) Sebastian Seung
# OD_sort(M) Outdegree
# SI_sort(M) Smooth Index
# --------------------------------------------------------

def sortlength(M):
    
    newM=np.triu(M)
            
    line,col=np.where(newM==1)
    length=np.abs(col-line)
    print(length)
    length_index=np.argsort(length)[::-1]
            
    return line[length_index[0]],col[length_index[0]]
    
def sortloop(M):
    
    mydim=M.shape
    mydim=int(mydim[0])
    sortlist=np.arange(mydim)
    inplength=countlength(M)
        
    outlength=inplength-1
    outM=1.0*M
    nofpermutations=0

    # xchl from 1 to mydim/2
    for xchl in range(1,int(mydim/2+1),1):
        # jump from mydim-xcl-1 to 0
        for jump in range(mydim-xchl-1,0,-1):
            # going down the list
            for i in range(mydim-jump-(xchl-1)):
                
                intM=1.0*outM
                nofpermutations+=1
                
                # exchange every element of the xchange package
                for j in range(xchl):
                    
                    intM=swapcolsnrows(intM,i+j,i+jump+j)
                    
                intlength=countlength(intM)
                
                if intlength<outlength:
                    
                    outM=1.0*intM
                    for j in range(xchl):
                        sortlist=swapnumbers(sortlist,i+j,i+jump+j)
                    outlength=intlength

    return outM, sortlist

def sortloop_new(M,sortlist):
    
    new_rcs=count_rcs(renumberM(M,sortlist))
    mydim=int(M.shape[0])
    new_list=sortlist.copy()
    
    for i in range(mydim-1):

        interim_list=swapnumbers(new_list,i,i+1)
        interim_rcs=count_rcs(renumberM(M,interim_list))
        
        if interim_rcs<new_rcs:
            
            new_rcs=1.0*interim_rcs
            new_list=interim_list.copy()
            
    new_M=renumberM(M,new_list)

    return new_M, new_list

def AB_sort(M):
    
    ABA,sortlist=sortloop(M)
    ABA,new_list=sortloop_new(M,sortlist)
    
    return ABA

#------------------Matrix Power------------------------

def MP_sort(M):

    def shuffelids(M,ip):

        Mnew=np.zeros_like(M)
        tmp=M[:,ip]
        Mnew=tmp[ip,:]
    
        return Mnew
 
    def powermat(M,a):

        Mnew=np.eye(M.shape[0])
        
        for a in range(a):
            Mnew=np.dot(Mnew,M)
    
        return Mnew

    def iteratepot(M):
    
        permopt=[]
        a=0
        while len(permopt)< M.shape[0]:
            a+=1
            Mpot=powermat(M,a)
            outdegrees=np.sum(Mpot,axis=0)
            outdegrees[permopt]=np.max(outdegrees)+1
            dmin=np.min(outdegrees)
            id0=np.where(outdegrees==dmin)[0]
            permopt.extend(id0)
    
        permopt=np.flip(permopt)
        
        return np.array(permopt), shuffelids(M,np.array(permopt))
    
    popt, Mest=iteratepot(M)
    
    in_dim=M.shape[0]
    out_dim=Mest.shape[0]
    
    Mest=Mest[out_dim-in_dim:out_dim,out_dim-in_dim:out_dim]
    
    return Mest

# ------------------Reverse_Cuthill_McKee---------------------------

def RC_sort(M):
    
    # reverse_cuthill_mckee
    
    S=sparse.csr_matrix(M)
    sortlist = scipy.sparse.csgraph.reverse_cuthill_mckee(S)
    
    outM=renumberM(M,sortlist)
    
    if np.sum(np.tril(outM))<np.sum(np.triu(outM)):
        outM=np.transpose(outM)
    
    return outM

# -------------- Depth First ----------------------------

def DF_sort(M):
    
    # Tarjan modified by Christian Leibold

    def initlowZ(mydim):

        lowZ=np.zeros((mydim,mydim))+1

        for j in range(1,mydim):

            for i in range(0,j):

                lowZ[j,i]=0


        return lowZ


    def shuffelids(M,ip):


        Mnew=copy.deepcopy(M[:,ip])
        Mnew=Mnew[ip,:]

        return Mnew

    def dfs(graph, start, end): #depth first search, RE Tarjan 1972

        fringe = [(start, [])]
        while fringe:
            state, path = fringe.pop()
            if path and state == end:
                yield path
                continue
            for next_state in graph[state]:
                if next_state in path:
                    continue
                fringe.append((next_state, path+[next_state]))


    def get_cycles(Mtmp):

       	Gnx=nx.from_numpy_matrix(Mtmp.transpose(), create_using=nx.DiGraph)
        narr=np.array(Gnx.nodes)
        earr=np.array(Gnx.edges)
        Mgraph={}

        for nn in narr:
            lst=[]
            for edge in earr:
                if edge[0]==nn:
                    lst.append(edge[1])

            Mgraph[nn]=lst


        rawcyc= [[node]+path  for node in Mgraph for path in dfs(Mgraph, node, node)]

        cycles=[]
        for cyc in rawcyc:

            check=0
            for j in cycles:
                check+=(sum([np.sum(j==jj) for jj in cyc])>0)

            if check==0:
                cycles.append(cyc[:-1])

        return cycles

    Gnx=nx.from_numpy_matrix(M.transpose(),create_using=nx.DiGraph)
    narr=np.array(Gnx.nodes)
    earr=np.array(Gnx.edges)
    Mgraph={}

    for nn in narr: #graph as dict
        lst=[]
        for edge in earr:
            if edge[0]==nn:
                lst.append(edge[1])

        Mgraph[nn]=lst


    td=tarjan.tarjan(Mgraph)
    
    raw=[]
    for dom in td:
        raw.extend(dom)

    modf=[]
    for dom in td:
        obj=np.zeros(len(dom))
        perms=[]
        adom=np.array(dom)
        for i0 in range(len(dom)):
            iperm=adom[np.mod(np.arange(len(dom))+i0,len(dom))]
            perms.append(iperm)
            U=M[iperm,:]
            U=U[:,iperm]
            obj[i0]=np.sum(U.transpose()*initlowZ(len(dom)))

        iopt=np.argmin(obj)
        modf.extend(perms[iopt])

    modf=np.flip(modf)
    
    return shuffelids(M,modf)

# -------------- Sebastian Seung ----------------------------

def SS_sort(M):
    
    L=np.zeros((M.shape[0],M.shape[0]))
    b=np.zeros(M.shape[0])
    
    din=np.sum(M,axis=1)
    dout=np.sum(M,axis=0)
            
    L=-M-np.transpose(M)
            
    for i in range(M.shape[0]):
        L[i,i]+=din[i]+dout[i]
        
    for i in range(M.shape[0]):
        b[i]=din[i]-dout[i]
        
    # calculates the pseudi-inverse
    
    Q=np.linalg.pinv(L)
    
    z=Q@b
    
    newM=renumberM(M,np.argsort(z))
            
    return newM   

# ----------- sorts according to outdegree -----------

def OD_sort(M):
    
    outdeg=np.sum(M,axis=0)
    x=np.flip(np.argsort(outdeg))
    newM=renumberM(M,x)
    
    return newM
    
# -----------------------------------------------------------------------------
# ----------------------------- Smooth Index ----------------------------------
# -----------------------------------------------------------------------------

def SI_sort(M, mode = 'recurrency', jac = 1, response='silent', pauli_fac=1.0, return_object = 'single'):
    
    # Smooth Index Sorting  works in two modes: -------------------------------
    # -------------------------------------------------------------------------
    # in 'recurrency' mode, it minimizes the length of recurrent connections
    # in 'bandwidth'  mode, it minimizes the length of all connections
    # -------------------------------------------------------------------------
    # if jac = 0, it calculates the gradient by evaluating the cost fct along
    #             all parameter axes
    # -------------------------------------------------------------------------
    # if jac = 1, it uses the jacobian function provided explicitly -----------
    # -------------------------------------------------------------------------

    mydim=M.shape[0]
    
    # ------- weights for the different cost functions and jacobians ----------
    
    recurrency_weight   = 5.0
    bandwidth_weight    = 10.0/((1.0*mydim)**2.0) 
    pauli_weight        = 20.0/((1.0*mydim)**2.0)*pauli_fac
    
    # ------------------------- bounds for z values ---------------------------
    
    z        = np.arange(mydim)
    z_bounds = [(0,mydim)]
    
    for i in range(mydim-1): 
        z_bounds.append((0,mydim))
        
    # ------------ the integers from 0 to N - 1 for the Pauli Function --------

    x=np.arange(mydim)
    
    # -----------the x and y coordinates of non-zero entries in M  ------------
    
    yval, xval = np.where(M != 0)
    
    # ------- initialize random values for z ----------------------------------
    
    def create_rand_params():

        z=np.zeros(mydim)
        
        for i in range(mydim):
                
            z_mean  = (z_bounds[i][1]+z_bounds[i][0])/2.0
            z_range = (z_bounds[i][1]-z_bounds[i][0])/2.0
            z[i]    = z_mean+(np.random.rand()-0.5) * z_range
                
        return z
    
    # -------- alpha function -------------------------------------------------
    
    def alpha(x,dim):
    
        y = 1.0 / (1.0 + np.exp(-10.0 / dim * x ))
        
        return y 
    
    # ---- definition of cost functions ---------------------------------------
    
    def calc_pauli(z):
        
        pauli = np.mean((np.sort(z)-x)**2) * pauli_weight
        
        return pauli
    
    def calc_length_of_recurrent_connections(z):
    
        D       = z[xval]-z[yval]+1
        D_rect  = D * (D > 0)
        
        length = np.sum(alpha(D_rect,mydim)-0.5) / np.sum(M) * recurrency_weight
            
        return length
    
    def calc_length_of_all_connections(z):
    
        D = z[xval]-z[yval]+1
            
        length = np.sum(D**2) / np.sum(M) * bandwidth_weight
            
        return length

    def calc_recurrency_cost(z):
        
        length = calc_length_of_recurrent_connections(z)
        pauli  = calc_pauli(z)
        cost   = length + pauli
 
        return cost
    
    def calc_bandwidth_cost(z):
        
        length = calc_length_of_all_connections(z)
        pauli  = calc_pauli(z)
        cost   = length + pauli
 
        return cost
    
    # -------- definition of jacobians ----------------------------------------
    
    def calc_recurrency_gradient(z):
    
        dz=np.zeros(mydim)
            
        for i in range(mydim):
            
            D1 = (z-z[i]+1) 
            D2 = (z[i]-z+1) 
            
            dz1   = - np.sum(M[i,:] * alpha(D1,mydim) * (1-alpha(D1,mydim)) * 10.0/mydim * (D1 > 0))
            dz2   = + np.sum(M[:,i] * alpha(D2,mydim) * (1-alpha(D2,mydim)) * 10.0/mydim * (D2 > 0))
            
            dz[i] = dz1 + dz2
            
        dz = dz / np.sum(M) * recurrency_weight
        
        return dz
    
    def calc_bandwidth_gradient(z):
    
        dz = np.zeros(mydim)
        
        for i in range(mydim):
            
            D1 = (z-z[i]+1) 
            D2 = (z[i]-z+1) 
            
            dz[i] = 2.0*(-np.sum(M[i,:]*D1)+np.sum(M[:,i]*D2))
            
        dz = dz / np.sum(M) * bandwidth_weight
        
        return dz
    
    def calc_pauli_gradient(z):
        
        z_positions = reverse_permutation(np.argsort(z))
        
        pauli_gradient = 2.0*(z-z_positions)/(1.0*mydim)*pauli_weight
        
        return pauli_gradient
    
        
    def calc_recurrency_jacobian(z):

        length_gradient = calc_recurrency_gradient(z)
            
        pauli_gradient = calc_pauli_gradient(z)
        
        gradient = length_gradient+pauli_gradient
            
        return gradient
    
    
    def calc_bandwidth_jacobian(z):

        length_gradient = calc_bandwidth_gradient(z)
            
        pauli_gradient = calc_pauli_gradient(z)
        
        gradient = length_gradient+pauli_gradient
            
        return gradient
    
    # -------------------------------------------------------------------------
    
    def send_message(res):
    
        jac_norm = np.sqrt(np.sum(res.jac**2))
        
        print()
        print('Optimization Success  :', res.success)
        print('Last Value of cost fct:', format(res.fun,'.5f'))
        print('Norm of Last Jacobian :', format(jac_norm,'.1e'))
        print('Number of cost fct use:', res.nfev)
        print('Number of Jacobian cal:', res.njev)
        print()
        
    # ----------- here, SI sorting starts doing the job -----------------------
        
    z=create_rand_params()
    
    method    = 'L-BFGS-B'
    options   = {'maxiter':2000}
    rec_tol   = 1e-8
    bwt_tol   = 1e-14

    if mydim > 1000:  rec_tol = 1e-9
    if mydim > 2000:  rec_tol = 1e-10
    if mydim > 5000:  rec_tol = 1e-11
    if mydim > 10000: rec_tol = 1e-12
    if mydim > 20000: rec_tol = 1e-14
      
    if jac == 0:
    
        if mode == 'recurrency':
            
            res = minimize(calc_recurrency_cost, z, method=method, tol=rec_tol, bounds=z_bounds, options=options)
        
        if mode == 'bandwidth':
            
            res = minimize(calc_bandwidth_cost,  z, method=method, tol=bwt_tol, bounds=z_bounds, options=options)
            
    if jac == 1:
        
        if mode=='recurrency':
            
            res = minimize(calc_recurrency_cost, z, jac=calc_recurrency_jacobian, method=method, tol=rec_tol, bounds=z_bounds, options=options)
            
        if mode=='bandwidth':

            res = minimize(calc_bandwidth_cost,  z, jac=calc_bandwidth_jacobian,  method=method, tol=bwt_tol, bounds=z_bounds, options=options)
        
    z = res.x
    
    newM=renumberM(M,np.argsort(z))
    
    if response != 'silent':  send_message(res)
    
    if return_object == 'single':
        
        return newM
    
    else:

        return newM, np.argsort(z)



    

