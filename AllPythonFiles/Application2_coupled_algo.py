import gurobipy as gp
from gurobipy import GRB
from mosek.fusion import *
import time,sys
import numpy as np 
import pandas as pd
from scipy.linalg import sqrtm
from DimacsReader import *

def save(name,finished,value,soltime,xsol):
    f = open("../output/Application2/"+name+"/coupledAlgo.txt","w+")
    if finished==True:
        f.write("Finished before time limit.\n")
    else:
        f.write("Time limit reached.\n")
    f.write("Obj: "+str(value)+"\n")
    f.write("SolTime: "+str(soltime)+"\n")
    f.write("\nUpper level solution: "+str(xsol)+"\n")
    f.close()

def main_app2(name_dimacs,name,mu,timelimit=18000):
    #Logs
    ValueLogRes, ValueLogRel, EpsLogs, MasterTimeLogs, LLTimeLogs = [],[],[],[],[]
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
    
    print("We are solving instance:", name)
    """Solve the restriction. If sufficient condition of GOPT is satisfied, stop"""
    t0 = time.time()
    xres, zres, obj = restriction(M,n,Q1,Q2,q1,q2,diagonalQ2x)
    x = xres
    mastertime = time.time() - t0
    
    #we check if the matrix Q2 is PD (i.e. sufficient condition satisfied) using Cholesky factorization:
    try:
        np.linalg.cholesky(Q2+np.diag(diagonalQ2x*xres))
        posdef = True
    except np.linalg.LinAlgError:
        posdef = False
    running = not(posdef)
    
    ValueLogRes.append(obj)
    ValueLogRel.append(-np.inf)
    EpsLogs.append(0)
    MasterTimeLogs.append(mastertime)
    LLTimeLogs.append(0)

    """If not, we run the inner/outer approximation algorithm """
    iteration = 0
    mu2 = mu * 100
    Qxk_list, qxk_list, vxk_list, yklist = [],[],[],[]
    while running and (time.time()-t0 < timelimit):
        print("Iteration number {0}".format(iteration))
        t1 = time.time()
        #we solve the master problem
        x,z,xrelax,crelax,obj,obj_relax,dist = master(M,n,Q1,Q2,q1,q2,diagonalQ2x,Qxk_list,qxk_list,np.array(vxk_list),yklist,mu)
        mastertime = time.time() - t1
        
        Qrelax = Q2+np.diag(diagonalQ2x*xrelax)
        brelax = q2 + (M.T)@xrelax
        tl = 10+max(0,timelimit-(time.time()-t0))

        #we solve the inner problem
        t1 = time.time()
        yrelax,epsrel = solve_subproblem_App2(n,Qrelax,brelax,crelax,tl) #we get epsrel = (v+crelax)
        LLtime = time.time() - t1
        Qxk_list.append(Qrelax)
        qxk_list.append(brelax)
        vxk_list.append(epsrel-crelax)#we solve the LL problem involving h(xrelax)=crelax, thus, to obtain v, we have to subtract from the LLobj h(xrelax)
        yklist.append(yrelax)
        #Logs
        ValueLogRes.append(obj)
        ValueLogRel.append(obj_relax)
        EpsLogs.append(epsrel)
        MasterTimeLogs.append(mastertime)
        LLTimeLogs.append(LLtime)
        print("ObjRes, ObjRel, Average = {0},{1},{2}".format(obj,obj_relax,0.5*obj+0.5*obj_relax))
        print("Distance term (check) = {0}".format(dist))
        print("Epsilon term (check) = {0}".format(epsrel))
        if epsrel>-1E-6 and dist<1E-6:
            running=False
        if abs(obj-obj_relax)/abs(obj)<0.001:
            mu = mu2
        iteration+=1
    
    soltime = time.time() - t0
    save(name,not(running),obj,soltime,x)
    df = pd.DataFrame()
    df['MasterObjRes'],df['MasterObjRel'],df["Epsilon"],df["MasterTime"],df['LLTime'] = ValueLogRes,ValueLogRel, EpsLogs, MasterTimeLogs, LLTimeLogs
    df.to_csv("../output/Application2/"+name+"/coupledAlgo.csv")

def restriction(M,n,Q1,Q2,q1,q2,diagonalQ2x):
    "Solve the single level restriction"
    with Model("App2") as model:
        A = -np.eye(n)
        
        #Upper level var
        z = model.variable("z", 1, Domain.unbounded())
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
        z_and_player1_cost = Expr.add(z, Expr.add(t,Expr.dot(q1,x))) #upper level objective function
        
        #Objective
        model.objective( "objfunct", ObjectiveSense.Minimize, z_and_player1_cost )
    
        #Simplex constraint for x
        model.constraint( Expr.sum(x),  Domain.equalsTo(1) )
         
        # -z + lambda1 + 2 alpha + beta \leq 0 
        sum_of_duals = Expr.add(lam,Expr.add(Expr.mul(2,alpha),beta))
        model.constraint(Expr.add(Expr.mul(-1,z),sum_of_duals),Domain.lessThan(0.0))
        
        #Constraints to define the several parts of the PSD matrix
        Q2x = Expr.add([Expr.mul(x.index(i),Matrix.sparse(n, n, [i], [i], [0.5*diagonalQ2x[i]])) for i in range(n)])
        model.constraint(Expr.sub(Expr.add(Expr.add(0.5*Q2,Q2x), Expr.mul(alpha,np.eye(n))), PSDVar_main),  Domain.equalsTo(0,n,n) )
        model.constraint(Expr.sub(Expr.add(Expr.add(0.5*q2, Expr.add(Expr.mul(0.5*M.T,x),Expr.mul(lam,0.5*np.ones(n)))),Expr.mul(lam2,0.5*A)), PSDVar_vec),  Domain.equalsTo(0,n) )
        model.constraint(Expr.sub(Expr.add(beta, alpha), PSDVar_offset),  Domain.equalsTo(0) )
    
        # Solve
        model.solve()
        soltime =  model.getSolverDoubleInfo("optimizerTime")
        
        #Get results
        xres = x.level()
        tres = t.level()[0]
        zres = z.level()[0]
        assert(abs(tres-0.5*xres.dot(Q1).dot(xres))<1E-7)
        assert(abs(PSDVar.level()[-1] - (alpha.level()[0]+beta.level()[0]))<1E-7)
        return xres, zres , zres + tres + xres.dot(q1)


def master(M,n,Q1,Q2,q1,q2,diagonalQ2x,Qxk_list,qxk_list, vxk_vector,yklist,mu):
    "Solve the master problem"
    K = len(yklist)
    with Model("App2") as model:
        A = -np.eye(n)
        #Upper level var
        z = model.variable("z", 1, Domain.unbounded())
        x = model.variable("x", n, Domain.greaterThan(0.0))
        zrelax = model.variable("zrelax", 1, Domain.unbounded())
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
        trelax = model.variable("trelax", 1, Domain.unbounded()) #upper level variable
        P1 = sqrtm(Q1) #necessary for the following constraints
        ## t >= 0.5 x^TQ_1x iif t >= 0.5 ||P_1 x ||^2   iif (t,1, P_1x) \in RotatedCone(n+2)
        ## This constraint is necessary saturated at the optimum, thus we have t = 0.5 x^TQ_1x
        model.constraint(Expr.vstack(t,1, Expr.mul(P1,x)), Domain.inRotatedQCone(n+2))
        model.constraint(Expr.vstack(trelax,1, Expr.mul(P1,xrelax)), Domain.inRotatedQCone(n+2))

        z_and_player1_cost = Expr.add(z, Expr.add(t,Expr.dot(q1,x))) #upper level objective function
        z_and_player1_cost_relax = Expr.add(zrelax, Expr.add(trelax,Expr.dot(q1,xrelax))) #upper level objective function
        
        #Objective
        model.constraint( Expr.vstack(1.0,Expr.vstack(distance_term.index(0), Expr.sub(x, xrelax))), Domain.inRotatedQCone() ) #distance_term first component
        model.constraint( Expr.vstack(1.0,Expr.vstack(distance_term.index(1), Expr.sub(z, zrelax))), Domain.inRotatedQCone() ) #distance_term second component
        
        model.objective( "objfunct", ObjectiveSense.Minimize, Expr.add(z_and_player1_cost,Expr.add(z_and_player1_cost_relax,Expr.mul(mu, Expr.sum(distance_term)))) )
    
        #Simplex constraint for x
        model.constraint( Expr.sum(x),  Domain.equalsTo(1) )
        model.constraint( Expr.sum(xrelax),  Domain.equalsTo(1) )
         
        # -z -\eta^t v + lambda1 + 2 alpha + beta  \leq 0 
        sum_of_duals = Expr.add(lam,Expr.add(Expr.mul(2,alpha),beta))
        model.constraint(Expr.add(Expr.mul(-1,Expr.add(Expr.dot(eta,vxk_vector),z)),sum_of_duals),Domain.lessThan(0.0))
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
            model.constraint(Expr.add(Expr.add(froeb_prod,scal_prod),zrelax), Domain.greaterThan(0))
    
    
        #Solve
        model.acceptedSolutionStatus(AccSolutionStatus.Anything)
        model.writeTask("App2.ptf")                # Save problem in readable format
        model.solve()
        soltime =  model.getSolverDoubleInfo("optimizerTime")
        
        #Get results
        xsol,zsol,xrelaxsol,zrelaxsol = x.level(), z.level()[0],xrelax.level(),zrelax.level()[0]
        dist = np.linalg.norm(xsol-xrelaxsol,2)**2 + (zsol-zrelaxsol)**2
        return xsol,zsol,xrelaxsol,zrelaxsol,zsol+0.5*(xsol@Q1@xsol)+q1@xsol,zrelaxsol+0.5*(xrelaxsol@Q1@xrelaxsol)+q1@xrelaxsol,dist
              
def solve_subproblem_App2(n,Q,b,c,tl):
    m = gp.Model("LL problem")
    m.Params.LogToConsole = 0
    y = m.addMVar(n, lb = 0.0, ub = 1.0, name="y")
    m.addConstr(np.ones(n)@y==1)
    m.setObjective(y@(0.5*Q)@y+  b@y +c, GRB.MINIMIZE)
    m.setParam('NonConvex', 2)
    m.setParam('TimeLimit', tl)
    m.optimize()
    return y.X, m.objVal
        
    # if name[-2]=='t':
    #     running=False
    # else:
