from mosek.fusion import *
import sys
import numpy as np 
import networkx
from scipy.linalg import sqrtm
from DimacsReader import *

def save(name, value,soltime, xsol):
    f = open("Application2_data/"+name+"/reformulation_obj_value.txt","w+")
    f.write("Obj: "+str(value)+"\n")
    f.write("SolTime: "+str(soltime)+"\n")
    f;write("Upper level solution: "+str(xsol)+"\n")
    f.close()

def main(name_dimacs,name):
    #Reading graph file
    f = DimacsReader("DIMACS/"+name_dimacs)
    M = f.M
    n = f.n
    
       
    Q1= np.load("Application2_data/"+name+"/bigQ1.npy")
    Q2= np.load("Application2_data/"+name+"/bigQ2_fix.npy")
    q1= np.load("Application2_data/"+name+"/q1.npy")
    q2= np.load("Application2_data/"+name+"/q2_fix.npy")     
    Mcheck = np.load("Application2_data/"+name+"/M.npy")
    
    print(np.linalg.norm(M-Mcheck))
    
    # Create a model with n semidefinite variables od dimension d x d
    
    with Model("App2") as model:
        
        #Upper level var
        v = model.variable("v", 1, Domain.unbounded())
        x = model.variable("x", n, Domain.greaterThan(0.0))
            
        #LL variables
        lam = model.variable("lambda", Domain.unbounded())
        alpha = model.variable("alpha", Domain.greaterThan(0.0))
        beta = model.variable("beta", Domain.unbounded())
        
        #Vars for PSD constraint
        PSDVar = model.variable(Domain.inPSDCone(n+1))
        PSDVar_main = PSDVar.slice([0,0], [n,n])
        PSDVar_vec = Var.flatten(PSDVar.slice([0,n], [n,n+1]))
        PSDVar_offset = PSDVar.slice([n,n], [n+1,n+1])
        
        #Objective
        model.objective( ObjectiveSense.Minimize, v )
    
        #Simplex constraint for x
        model.constraint( Expr.sum(x),  Domain.equalsTo(1) )
        
        ##t >= 0.5 x^TQ_1x iif t >= 0.5 ||P_1 x ||^2   iif (t,1, P_1x) \in RotatedCone(n+2)
        ## This constraint is necessary saturated at the optimum, thus we have t = 0.5 x^TQ_1x
        
        P1 = sqrtm(Q1)
        t = model.variable("t", 1, Domain.unbounded())
        model.constraint(Expr.vstack(t,1, Expr.mul(P1,x)), Domain.inRotatedQCone(n+2))
        
        # -v + t +q_1^Tx + lambda + 2 alpha + beta \leq 0 
        v_and_player1_cost = Expr.add( Expr.mul(-1,v), Expr.add(t,Expr.dot(q1,x)))
        sum_of_duals = Expr.add(lam,Expr.add(Expr.mul(2,alpha),beta))
        model.constraint(Expr.add(v_and_player1_cost,sum_of_duals),Domain.lessThan(0.0))
        
        #Constraints to define the several parts of the PSD matrix
        model.constraint(Expr.sub(Expr.add(0.5*Q2, Expr.mul(alpha,np.eye(n))), PSDVar_main),  Domain.equalsTo(0,n,n) )
        model.constraint( Expr.sub(Expr.add(0.5*q2, Expr.add(Expr.mul(0.5*M.T,x),Expr.mul(lam,0.5*np.ones(n)))), PSDVar_vec),  Domain.equalsTo(0,n) )
        model.constraint( Expr.sub(Expr.add(beta, alpha), PSDVar_offset),  Domain.equalsTo(0) )
    
    
        # Solve
        model.setLogHandler(sys.stdout)            # Add logging
        model.writeTask("App2.ptf")                # Save problem in readable format
        model.solve()
        soltime =  model.getSolverDoubleInfo("optimizerTime")
        
        #Get results
        print("Objective value ={0}".format(v.level()))
        save(name,v.level()[0],soltime, x.level())
        xres = x.level()
        tres = t.level()[0]
        print("Check rotated cone constraint (t = 0.5 x^TQ_1x) : ", abs(tres-0.5*xres.dot(Q1).dot(xres)))
        print("Check last coefficient constraint :", PSDVar.level()[-1] - (alpha.level()[0]+beta.level()[0]))
        print("Upper level solution : ",x.level())
        print(lam.level() + 2 * alpha.level() + beta.level())
        print(alpha.level() + beta.level())
            
    
if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])  
        