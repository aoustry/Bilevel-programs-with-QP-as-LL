# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

from mosek.fusion import *
import sys
import numpy as np 
import networkx
from scipy.linalg import sqrtm
from DimacsReader import *

def save(name, value,soltime, xsol):
    f = open("Application2bis_data/"+name+"/reformulation_obj_value.txt","w+")
    f.write("Obj: "+str(value)+"\n")
    f.write("SolTime: "+str(soltime)+"\n")
    f.write("Upper level solution: "+str(xsol)+"\n")
    f.close()
    
def solve_subproblem_App2(n,Q,b,c):
    m = gp.Model("LL problem")
    y = m.addMVar(n, lb = 0.0, ub = 1.0, name="y")
    m.addConstr(np.ones(n)@y==1)
    m.setObjective(y@(0.5*Q)@y+  b@y +c, GRB.MINIMIZE)
    m.setParam('NonConvex', 2)
    m.optimize()
    return y.X, m.objVal

def main(name_dimacs,name,mu):
    #Reading graph file
    f = DimacsReader("DIMACS/"+name_dimacs)
    M = f.M
    n = f.n

    Q1= np.load("Application2bis_data/"+name+"/bigQ1.npy")
    Q2= np.load("Application2bis_data/"+name+"/bigQ2_fix.npy")
    q1= np.load("Application2bis_data/"+name+"/q1.npy")
    q2= np.load("Application2bis_data/"+name+"/q2_fix.npy")    
    diagonalQ2x = np.load("Application2bis_data/"+name+"/diagQ2x.npy")
    Mcheck = np.load("Application2bis_data/"+name+"/M.npy")
    assert(np.linalg.norm(M-Mcheck)<1E-6)
    assert(np.linalg.norm(M-M.T)<1E-6)
    assert(np.linalg.norm(Q1-Q1.T)<1E-6)
    assert(np.linalg.norm(Q2-Q2.T)<1E-6)
    
    Qxk_list, qxk_list, vxk_list,yklist,running = [],[],[],[],True
    it_count = 0
    while running:
        x,c,xrelax,crelax,dist = master(M,n,Q1,Q2,q1,q2,diagonalQ2x,Qxk_list,qxk_list, np.array(vxk_list),yklist,mu)
        Qrelax = Q2+np.diag(diagonalQ2x*xrelax)
        brelax = q2 + (M.T)@xrelax
        yrelax,valrelax = solve_subproblem_App2(n,Qrelax,brelax,crelax)
        Qxk_list.append(Qrelax)
        qxk_list.append(brelax)
        vxk_list.append(valrelax-crelax)
        yklist.append(yrelax)

        if valrelax>-1E-6 and dist<1E-6:
            running=False
        it_count+=1
        print("Iteration number {0}".format(it_count))
    #save(name,objres,soltime, x.level(),v.level())

def master(M,n,Q1,Q2,q1,q2,diagonalQ2x,Qxk_list,qxk_list, vxk_vector,yklist,mu):
        
    # Create a model with n semidefinite variables od dimension d x d
    K = len(yklist)
    with Model("App2bis") as model:
        #y must be greater than 0
        A = -np.eye(n)
        #b = np.zeros(n)
        
        #Upper level var
        c = model.variable("v", 1, Domain.unbounded())
        x = model.variable("x", n, Domain.greaterThan(0.0))
        crelax = model.variable("vrelax", 1, Domain.unbounded())
        xrelax = model.variable("xrelax", n, Domain.greaterThan(0.0))
        distance_term = model.variable("dist", 2, Domain.unbounded())
            
        #LL variables
        lam = model.variable("lambda", Domain.unbounded()) #lagrangian multiplier related to the equality constraint (simplex)
        lam2 = model.variable("lambda2", n, Domain.greaterThan(0.0)) #lagrangian multiplier related to the nonnegativity of y
        alpha = model.variable("alpha", Domain.greaterThan(0.0))
        beta = model.variable("beta", Domain.unbounded())
        eta = model.variable("eta", K, Domain.greaterThan(0.0))
        
        #Vars for PSD constraint
        PSDVar = model.variable(Domain.inPSDCone(n+1))
        PSDVar_main = PSDVar.slice([0,0], [n,n])
        PSDVar_vec = Var.flatten(PSDVar.slice([0,n], [n,n+1]))
        PSDVar_offset = PSDVar.slice([n,n], [n+1,n+1])
        
        #other auxiliary variables
        t = model.variable("t", 1, Domain.unbounded()) #upper level variable
        trelax = model.variable("trelax", 1, Domain.unbounded())
        P1 = sqrtm(Q1) #necessary for the following constraint
        ##t >= 0.5 x^TQ_1x iif t >= 0.5 ||P_1 x ||^2   iif (t,1, P_1x) \in RotatedCone(n+2)
        ## This constraint is necessary saturated at the optimum, thus we have t = 0.5 x^TQ_1x
        model.constraint(Expr.vstack(t,1, Expr.mul(P1,x)), Domain.inRotatedQCone(n+2))
        model.constraint(Expr.vstack(trelax,1, Expr.mul(P1,xrelax)), Domain.inRotatedQCone(n+2))
        c_and_player1_cost = Expr.add(c, Expr.add(t,Expr.dot(q1,x))) #upper level objective function
        c_and_player1_cost_relax = Expr.add(crelax, Expr.add(trelax,Expr.dot(q1,xrelax))) #upper level objective function
        
        
        #Objective
        model.constraint( Expr.vstack(1.0,Expr.vstack(distance_term.index(0), Expr.sub(x, xrelax))), Domain.inRotatedQCone() ) 
        model.constraint( Expr.vstack(1.0,Expr.vstack(distance_term.index(1), Expr.sub(c, crelax))), Domain.inRotatedQCone() ) 
        
        model.objective( "objfunct", ObjectiveSense.Minimize, Expr.add(c_and_player1_cost,Expr.add(c_and_player1_cost_relax,Expr.mul(mu, Expr.sum(distance_term))) ))
    
        #Simplex constraint for x
        model.constraint( Expr.sum(x),  Domain.equalsTo(1) )
        model.constraint( Expr.sum(xrelax),  Domain.equalsTo(1) )
         
        # -v -\eta^t v + lambda1 + 2 alpha + beta  \leq 0 
        sum_of_duals = Expr.add(lam,Expr.add(Expr.mul(2,alpha),beta))
        model.constraint(Expr.add(Expr.mul(-1,Expr.add(Expr.dot(eta,vxk_vector),c)),sum_of_duals),Domain.lessThan(0.0))
        Q2x = Expr.add([Expr.mul(x.index(i),Matrix.sparse(n, n, [i], [i], [0.5*diagonalQ2x[i]])) for i in range(n)])
            
        #Constraints to define the several parts of the PSD matrix
        if K>=1:
            combiliMat = np.zeros((n,n))
            combiliVect = np.zeros(n)
            for i in range(K):
                combiliMat = Expr.add(combiliMat, Expr.mul(eta.index(i),Qxk_list[i]))
                combiliVect = Expr.add(combiliVect, Expr.mul(eta.index(i),qxk_list[i]))
            model.constraint(Expr.sub(Expr.add(Expr.sub(Expr.add(0.5*Q2,Q2x),Expr.mul(0.5,combiliMat)), Expr.mul(alpha,np.eye(n))), PSDVar_main),  Domain.equalsTo(0,n,n) )
            model.constraint(Expr.sub(Expr.add(Expr.add(Expr.sub(0.5*q2,Expr.mul(0.5,combiliVect)), Expr.add(Expr.mul(0.5*M.T,x),Expr.mul(lam,0.5*np.ones(n)))),Expr.mul(lam2,0.5*A)), PSDVar_vec),  Domain.equalsTo(0,n) )
            model.constraint(Expr.sub(Expr.add(beta, alpha), PSDVar_offset),  Domain.equalsTo(0) )
    
        else:
        
            model.constraint(Expr.sub(Expr.add(Expr.add(0.5*Q2,Q2x), Expr.mul(alpha,np.eye(n))), PSDVar_main),  Domain.equalsTo(0,n,n) )
            model.constraint(Expr.sub(Expr.add(Expr.add(0.5*q2, Expr.add(Expr.mul(0.5*M.T,x),Expr.mul(lam,0.5*np.ones(n)))),Expr.mul(lam2,0.5*A)), PSDVar_vec),  Domain.equalsTo(0,n) )
            model.constraint(Expr.sub(Expr.add(beta, alpha), PSDVar_offset),  Domain.equalsTo(0) )
    
        
        Q2x_relax = Expr.add([Expr.mul(xrelax.index(i),Matrix.sparse(n, n, [i], [i], [0.5*diagonalQ2x[i]])) for i in range(n)])
        quad = Expr.add(0.5*Q2,Q2x_relax)
        
        for k in range(K):
            y = yklist[k]
            Y = (y.reshape(n,1).dot(y.reshape(1,n))).reshape(n**2)
            froeb_prod = Expr.dot(Expr.flatten(quad),Y.flatten())
            scal_prod = Expr.dot(y,Expr.add(q2,Expr.mul(M,xrelax)))
            model.constraint(Expr.add(Expr.add(froeb_prod,scal_prod),crelax), Domain.greaterThan(0))
    
    
        #Solve
        model.setLogHandler(sys.stdout)            # Add logging
        model.writeTask("App2.ptf")                # Save problem in readable format
        model.solve()
        soltime =  model.getSolverDoubleInfo("optimizerTime")
        
        #Get results
        xsol,csol,xrelaxsol,crelaxsol = x.level(), c.level(),xrelax.level(),crelax.level()
        dist = np.linalg.norm(xsol-xrelaxsol,2)**2 + (csol-crelaxsol)**2
        
        print(csol+0.5*(xsol@Q1@xsol)+q1@xsol,crelaxsol+0.5*(xrelaxsol@Q1@xrelaxsol)+q1@xrelaxsol)
        print(dist)
        print(eta.level())
        return xsol,csol,xrelaxsol,crelaxsol, dist
        
            

            
main('queen8_12.col','queen8_12_random1',1)
# if __name__ == "__main__":
#     main(sys.argv[1],sys.argv[2])  
        
