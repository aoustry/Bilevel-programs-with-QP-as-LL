# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:22:49 2020

@author: aoust
"""

from mosek.fusion import *
import sys
import numpy as np 

# Sample input data
n = 5
asym = np.array([[2.7,-0.2,0,1,3],[2,2,1,0,0],[0,0,0.3,1,2],[1,0,0.3,5,2],[1.2,0,0.3,-1,0.4]])
Qref = 0.5*(asym + asym.transpose())
qref = np.array([-2.3,-2,1,2,1])
cref = 1.2
sigma = 0.4
p = 5000
wlist = np.random.rand(p,n)
noise = np.random.normal(loc = 0, scale = sigma, size = p)
noiseless_z=0.5 *np.array([ w.dot(Qref).dot(w) for w in wlist]) + wlist.dot(qref) + cref
print(noiseless_z.min())
z = noiseless_z + noise
wlist_square_flattened = np.array([(w.reshape(n,1).dot(w.reshape(1,n))).reshape(n**2) for w in wlist]) 
#for each vector w, we create a matrix n x n (given by w*w^T) and then a vector n**2

#Definition of the LL polytope : [0,1]^n box
A = np.concatenate([np.eye(n),-np.eye(n)])
b = np.concatenate([np.ones(n),np.zeros(n)])
rho = n

# Create a model with n semidefinite variables od dimension d x d

with Model("App1") as M:
    
    #Upper level var
    obj = M.variable("t", 1, Domain.unbounded())
    Q = M.variable("Q", [n,n], Domain.unbounded())
    q = M.variable("q", n, Domain.unbounded())
    c = M.variable("c", Domain.unbounded())
    
    #LL variables
    lam = M.variable("lambda", len(b), Domain.greaterThan(0.0))
    alpha = M.variable("alpha", Domain.greaterThan(0.0))
    beta = M.variable("beta", Domain.unbounded())
    
    #Vars for PSD constraint
    PSDVar = M.variable(Domain.inPSDCone(n+1)) #the whole matrix that must be PSD in (27)
    PSDVar_main = PSDVar.slice([0,0], [n,n]) #we take the first submatrix n x n in each component of the sum in (27) i.e. 0.5*Q, 0_n, alpha*I_n
    PSDVar_vec = Var.flatten(PSDVar.slice([0,n], [n,n+1])) #we take the second submatrix n x 1 i.e. 0.5*q, \sum_r(lambda_r*A)
    PSDVar_offset = PSDVar.slice([n,n+1], [n,n+1])   #we take the third submatrix 1 x 1 i.e. \beta, alpha*1
    
    #Objective
    deg0term = Expr.mul(c,np.ones(p))
    deg1term = Expr.mul(wlist, q)
    deg2term = Expr.mul(0.5,Expr.mul(wlist_square_flattened, Var.flatten(Q)))
    
    prediction_term =  Expr.add(deg2term,Expr.add(deg1term,deg0term)) #predicted z (p points)
    M.constraint( Expr.vstack(obj, Expr.sub(prediction_term, z)), Domain.inQCone() ) #see below -- vstack put obj "above" (z-zpredicted) 
    M.objective( ObjectiveSense.Minimize, obj )
    #we minimize obj s.t. obj**2 >= \sum_p (z_p-zpredicted_p)**2 -> It is like I am minimizing the square root of LSE 
    
    #Symmetry constraint for Q
    M.constraint( Expr.sub(Q, Q.transpose()),  Domain.equalsTo(0,n,n) ) #Q-Q^T = 0
   
    # c - (\lambda^T b + \alpha (1+ rho^2) + beta) >=0 
    LL_obj_expr = Expr.add(Expr.dot(lam,b),Expr.add(Expr.mul((1+rho**2),alpha),beta))
    M.constraint(Expr.sub(c,LL_obj_expr),Domain.greaterThan(0.0))
    
    #Constraints to define the several parts of the PSD matrix
    M.constraint(Expr.sub(Expr.add(Expr.mul(0.5,Q), Expr.mul(alpha,np.eye(n))), PSDVar_main),  Domain.equalsTo(0,n,n) )  
    M.constraint( Expr.sub(Expr.add(Expr.mul(0.5,q), Expr.mul(lam,A)), PSDVar_vec),  Domain.equalsTo(0,n) )
    M.constraint( Expr.sub(Expr.add(beta, alpha), PSDVar_offset),  Domain.equalsTo(0) )


    # Solve
    #M.setLogHandler(sys.stdout)            # Add logging
    M.writeTask("App1.ptf")                # Save problem in readable format
    M.solve()

    #Get results
    print("Objective value ={0}".format(obj.level()))
    print(Q.level().reshape(n,n))
    print(Qref)
    print(q.level())
    print(qref)
    print("Average square error reconstruction ={0}".format(obj.level()**2/p))
    print("Average square error data ={0}".format(np.array([n**2 for n in noise]).sum()/p))
    print("TODO : also print error between ref and reconstructed output")
