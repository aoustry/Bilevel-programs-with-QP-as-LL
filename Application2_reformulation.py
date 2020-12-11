from mosek.fusion import *
import sys
import numpy as np 
import networkx
from scipy.linalg import sqrtm
from DimacsReader import *

#Reading graph file
f = DimacsReader("DIMACS/jean.col")
M = f.M
n = f.n


#Cost param 
quadcostlevel = 0.1
linear_cost1 = 0.1
linear_cost2 = 0.1

# Sample input data
aux1 =np.random.rand(n,n)
aux2 =np.random.rand(n,n)
P1 = quadcostlevel*(aux1 + aux1.T)
Q1 = P1.dot(P1.transpose())
Q2 = quadcostlevel*(aux2 + aux2.T)
q1 = linear_cost1*np.ones(n)
q2 = linear_cost2*np.ones(n)



# Create a model with n semidefinite variables od dimension d x d

with Model("App1") as model:
    
    #Upper level var
    v = model.variable("v", 1, Domain.unbounded())
    x = model.variable("x", n, Domain.greaterThan(0.0))
        
    #LL variables
    lam = model.variable("lambda", Domain.greaterThan(0.0))
    alpha = model.variable("alpha", Domain.greaterThan(0.0))
    beta = model.variable("beta", Domain.unbounded())
    
    #Vars for PSD constraint
    PSDVar = model.variable(Domain.inPSDCone(n+1))
    PSDVar_main = PSDVar.slice([0,0], [n,n])
    PSDVar_vec = Var.flatten(PSDVar.slice([0,n], [n,n+1]))
    PSDVar_offset = PSDVar.slice([n,n+1], [n,n+1])
    
    #Objective
    model.objective( ObjectiveSense.Minimize, v )

    
    #Simplex constraint for x
    model.constraint( Expr.sum(x),  Domain.equalsTo(1) )
   
    
    SOCvariable = model.variable(Domain.inQCone(n+1))
    SOCvariable_1n = SOCvariable.slice(1,n+1)
    #t >= (SOC[0])**2 >= x^TQ_1x
    model.constraint(Expr.sub(SOCvariable_1n,Expr.mul(P1,x)),Domain.equalsTo(0.0))
    t = model.variable("t", 1, Domain.unbounded())
    model.constraint(Expr.hstack(0.5, t, SOCvariable.index(0)), Domain.inRotatedQCone())
    
    # -v + 0.5 t +q_1^Tx + lambda + 2 alpha + beta \leq 0 
    
    v_and_player1_opt = Expr.add( Expr.mul(-1,v), Expr.add(Expr.mul(0.5,t),Expr.dot(q1,x)))
    sum_of_duals = Expr.add(lam,Expr.add(Expr.mul(2,alpha),beta))
    model.constraint(Expr.add(v_and_player1_opt,sum_of_duals),Domain.lessThan(0.0))
    
    #Constraints to define the several parts of the PSD matrix
    model.constraint(Expr.sub(Expr.add(0.5*Q2, Expr.mul(alpha,np.eye(n))), PSDVar_main),  Domain.equalsTo(0,n,n) )
    model.constraint( Expr.sub(Expr.add(0.5*q2, Expr.add(Expr.mul(M.T,x),Expr.mul(lam,np.ones(n)))), PSDVar_vec),  Domain.equalsTo(0,n) )
    model.constraint( Expr.sub(Expr.add(beta, alpha), PSDVar_offset),  Domain.equalsTo(0) )


    # Solve
    model.setLogHandler(sys.stdout)            # Add logging
    model.writeTask("App1.ptf")                # Save problem in readable format
    model.solve()

    #Get results
    print("Objective value ={0}".format(v.level()))
    xres = x.level()
    tres = t.level()
    assert(abs(tres-xres.dot(Q1).dot(xres))<=1E-6)
    print(xres.dot(Q1).dot(xres))
    print(x.level())
   
    