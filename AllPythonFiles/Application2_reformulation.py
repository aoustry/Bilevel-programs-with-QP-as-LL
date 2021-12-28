# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 15:33:35 2021

@author: aoust
"""

from mosek.fusion import *
import sys
import numpy as np 
import networkx
from scipy.linalg import sqrtm
from DimacsReader import *

def save(name, value,soltime, xsol):
    f = open("../Application2_data/"+name+"/reformulation_obj_value.txt","w+")
    f.write("Obj: "+str(value)+"\n")
    f.write("SolTime: "+str(soltime)+"\n")
    f.write("Upper level solution: "+str(xsol)+"\n")
    f.close()

def main(name_dimacs,name):
    #Reading graph file
    f = DimacsReader("../DIMACS/"+name_dimacs)
    M = f.M
    n = f.n

    Q1= np.load("../Application2_data/"+name+"/bigQ1.npy")
    Q2= np.load("../Application2_data/"+name+"/bigQ2_fix.npy")
    q1= np.load("../Application2_data/"+name+"/q1.npy")
    q2= np.load("../Application2_data/"+name+"/q2_fix.npy")    
    diagonalQ2x = np.load("../Application2_data/"+name+"/diagQ2x.npy")
    Mcheck = np.load("../Application2_data/"+name+"/M.npy")
    
    assert(np.linalg.norm(M-Mcheck)<1E-6)
    assert(np.linalg.norm(M-M.T)<1E-6)
    assert(np.linalg.norm(Q1-Q1.T)<1E-6)
    assert(np.linalg.norm(Q2-Q2.T)<1E-6)
    
    # Create a model with n semidefinite variables od dimension d x d
    
    with Model("App2") as model:
        #y must be greater than 0
        A = -np.eye(n)
        #b = np.zeros(n)
        
        #Upper level var
        v = model.variable("v", 1, Domain.unbounded())
        x = model.variable("x", n, Domain.greaterThan(0.0))
            
        #LL variables
        lam = model.variable("lambda", Domain.unbounded()) #lagrangian multiplier related to the equality constraint (simplex)
        lam2 = model.variable("lambda2", n, Domain.greaterThan(0.0)) #lagrangian multiplier related to the nonnegativity of y
        alpha = model.variable("alpha", Domain.greaterThan(0.0))
        beta = model.variable("beta", Domain.unbounded())
        
        #Vars for PSD constraint
        PSDVar = model.variable(Domain.inPSDCone(n+1))
        PSDVar_main = PSDVar.slice([0,0], [n,n])
        PSDVar_vec = Var.flatten(PSDVar.slice([0,n], [n,n+1]))
        PSDVar_offset = PSDVar.slice([n,n], [n+1,n+1])
        #other auxiliary variables
        t = model.variable("t", 1, Domain.unbounded()) #upper level variable
        P1 = sqrtm(Q1) #necessary for the following constraint
        ##t >= 0.5 x^TQ_1x iif t >= 0.5 ||P_1 x ||^2   iif (t,1, P_1x) \in RotatedCone(n+2)
        ## This constraint is necessary saturated at the optimum, thus we have t = 0.5 x^TQ_1x
        model.constraint(Expr.vstack(t,1, Expr.mul(P1,x)), Domain.inRotatedQCone(n+2))
        v_and_player1_cost = Expr.add(v, Expr.add(t,Expr.dot(q1,x))) #upper level objective function
        
        #Objective
        model.objective( "objfunct", ObjectiveSense.Minimize, v_and_player1_cost )
    
        #Simplex constraint for x
        model.constraint( Expr.sum(x),  Domain.equalsTo(1) )
         
        # -v + lambda1 + 2 alpha + beta \leq 0 
        sum_of_duals = Expr.add(lam,Expr.add(Expr.mul(2,alpha),beta))
        model.constraint(Expr.add(Expr.mul(-1,v),sum_of_duals),Domain.lessThan(0.0))
        
        #Constraints to define the several parts of the PSD matrix
        Q2x = Expr.add([Expr.mul(x.index(i),Matrix.sparse(n, n, [i], [i], [0.5*diagonalQ2x[i]])) for i in range(n)])
        model.constraint(Expr.sub(Expr.add(Expr.add(0.5*Q2,Q2x), Expr.mul(alpha,np.eye(n))), PSDVar_main),  Domain.equalsTo(0,n,n) )
        model.constraint(Expr.sub(Expr.add(Expr.add(0.5*q2, Expr.add(Expr.mul(0.5*M.T,x),Expr.mul(lam,0.5*np.ones(n)))),Expr.mul(lam2,0.5*A)), PSDVar_vec),  Domain.equalsTo(0,n) )
        model.constraint(Expr.sub(Expr.add(beta, alpha), PSDVar_offset),  Domain.equalsTo(0) )
    
        # Solve
        model.setLogHandler(sys.stdout)            # Add logging
        model.writeTask("App2.ptf")                # Save problem in readable format
        model.solve()
        soltime =  model.getSolverDoubleInfo("optimizerTime")
        
        #Get results
        xres = x.level()
        tres = t.level()[0]
        vres = v.level()
        objres = 0.5*xres.dot(Q1).dot(xres) + vres + q1.dot(xres)
        print("Min eigenvalue = {0}".format(min(np.linalg.eigvalsh(Qsol))))
        assert(abs(tres-0.5*xres.dot(Q1).dot(xres))<1E-7)
        assert(abs(PSDVar.level()[-1] - (alpha.level()[0]+beta.level()[0]))<1E-7)
        print("Upper level solution : ",x.level())
        print("Objective value =",objres)
        print("v :", v.level())
        save(name,objres,soltime, x.level())
            

            
    
if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])  
        
