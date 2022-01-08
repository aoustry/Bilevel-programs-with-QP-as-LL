import pandas as pd
from mosek.fusion import *
import gurobipy as gp
from gurobipy import GRB
import sys
import numpy as np
import time

def save(name,finished,p,value,relax,soltime,iteration, bigQ,q,c):
    f = open("../output/Application1/"+name+"/InnerOuterApproxAlgo.txt","w+")
    if finished==True:
        f.write("Finished before time limit.\n")
    else:
        f.write("Time limit reached.\n")
    f.write("Obj: "+str(value)+"\n")
    if iteration>0 and finished==False:
        f.write("Obj relaxation: "+str(relax)+"\n")
    f.write("Average LSE: {0}\n".format(value/p))
    f.write("SolTime: "+str(soltime)+"\n")
    f.write("It. number: "+str(iteration)+"\n")
    f.write("\nQ matrix recovered: " +str(bigQ) +"\n")
    f.write("q vector recovered: " +str(q) +"\n")
    f.write("Scalar c recovered: " +str(c) +"\n")
    f.close()
    

def main_app1(name,mu,timelimit = 18000):
    #Logs
    ValueLogRes, ValueLogRel, EpsLogs, MasterTimeLogs, LLTimeLogs = [],[],[],[],[]
    #Loading data
    wlist = np.load("../Application1_data/"+name+"/w.npy")
    n = wlist.shape[1]
    z = np.load("../Application1_data/"+name+"/z.npy")
    
    print("We are solving instance:", name)
    """Solve the restriction. If sufficient condition of GOPT is satisfied, stop"""
    t0 = time.time()
    Qsol,qsol,csol,obj=restriction(name,n,wlist,z)
    mastertime = time.time() - t0
    obj_relax=0 #random number
    
    #we check if the matrix Q2 is PD (i.e. sufficient condition satisfied) using Cholesky factorization:
    try:
        np.linalg.cholesky(Qsol)
        running = False
        if min(np.linalg.eig(Qsol)[0])<1E-7: #the matrix can be considered positive SEMIdefinite
            running = True
    except np.linalg.LinAlgError: #Cholesky factorization function will return this error if the matrix is not positive definite
        running = True

    ValueLogRes.append(obj)
    ValueLogRel.append(-np.inf)
    EpsLogs.append(0)
    MasterTimeLogs.append(mastertime)
    LLTimeLogs.append(0)

    """If not, we run the inner/outer approximation algorithm """
    iteration = 0
    mu2 = 100*mu
    Qxk_list,qxk_list,vxk_list,yklist = [],[],[],[]
    while running and (time.time()-t0<timelimit):
        print("Iteration number {0}".format(iteration+1))
        t1 = time.time()
        #we solve the master problem
        Qsol,qsol,csol,Qsolrelax,qsolrelax,csolrelax,obj,obj_relax,dist = master(name,n,wlist,z,Qxk_list,qxk_list,np.array(vxk_list),yklist,mu)
        mastertime = time.time() - t1
        
        tl = 10+max(0,timelimit-(time.time()-t0))
        #we solve the inner problem
        t1 = time.time()
        yrelax,epsrel = solve_subproblem_App1(n,Qsolrelax,qsolrelax,csolrelax,tl)
        LLtime = time.time() - t1
        Qxk_list.append(Qsolrelax)
        qxk_list.append(qsolrelax)
        vxk_list.append(epsrel-csolrelax)
        yklist.append(yrelax)
        #Logs
        ValueLogRes.append(obj)
        ValueLogRel.append(obj_relax)
        EpsLogs.append(epsrel)
        MasterTimeLogs.append(mastertime)
        LLTimeLogs.append(LLtime)
        if abs(obj-obj_relax)/abs(obj)<0.001:
            mu = mu2
        if epsrel>-1E-6 and dist<1E-6:
            running=False
        iteration+=1
        print("ObjRes, ObjRel, Average = {0},{1},{2}".format(obj,obj_relax,0.5*obj+0.5*obj_relax))
        print("Distance term (check) = {0}".format(dist))
        print("Epsilon term (check) = {0}".format(epsrel))
    soltime = time.time() - t0
    save(name,not(running),len(z),obj,obj_relax,soltime,iteration, Qsol,qsol,csol)
    df = pd.DataFrame()
    df['MasterObjRes'],df['MasterObjRel'],df["Epsilon"],df["MasterTime"],df['LLTime'] = ValueLogRes, ValueLogRel, EpsLogs, MasterTimeLogs, LLTimeLogs
    df.to_csv("../output/Application1/"+name+"/InnerOuterApproxAlgo.csv")
    

def restriction(name,n,wlist,z):
    "Solve the single level restriction"
    p = len(z) 
    wlist_square_flattened = np.array([(w.reshape(n,1).dot(w.reshape(1,n))).reshape(n**2) for w in wlist]) 
    
    #Definition of the LL polytope : [0,1]^n box
    A = np.concatenate([np.eye(n),-np.eye(n)])
    b = np.concatenate([np.ones(n),np.zeros(n)])
    rho = np.sqrt(n)
    
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
        PSDVar_offset = PSDVar.slice([n,n], [n+1,n+1])   #we take the third submatrix 1 x 1 i.e. \beta, alpha*1
        
        #Objective
        deg0term = Expr.mul(c,np.ones(p))
        deg1term = Expr.mul(wlist, q)
        deg2term = Expr.mul(0.5,Expr.mul(wlist_square_flattened, Var.flatten(Q)))
        
        prediction_term =  Expr.add(deg2term,Expr.add(deg1term,deg0term)) #predicted z (p points)
        M.constraint( Expr.vstack(obj, Expr.sub(prediction_term, z)), Domain.inQCone() ) #see below -- vstack put obj "above" (z-zpredicted) 
        M.objective( ObjectiveSense.Minimize, obj )
        #we minimize obj s.t. obj**2 >= \sum_p (z_p-zpredicted_p)**2 -> It is like we are minimizing the square root of LSE 
        
        #Symmetry constraint for Q
        M.constraint( Expr.sub(Q, Q.transpose()),  Domain.equalsTo(0,n,n) ) #Q-Q^T = 0
       
        # c - (\lambda^T b + \alpha (1+ rho^2) + beta) >=0 
        LL_obj_expr = Expr.add(Expr.dot(lam,b),Expr.add(Expr.mul((1+rho**2),alpha),beta))
        M.constraint(Expr.sub(c,LL_obj_expr),Domain.greaterThan(0.0))
        
        #Constraints to define the several parts of the PSD matrix
        M.constraint(Expr.sub(Expr.add(Expr.mul(0.5,Q), Expr.mul(alpha,np.eye(n))), PSDVar_main),  Domain.equalsTo(0,n,n) )  
        M.constraint( Expr.sub(Expr.add(Expr.mul(0.5,q), Expr.mul(lam,0.5*A)), PSDVar_vec),  Domain.equalsTo(0,n) )
        M.constraint( Expr.sub(Expr.add(beta, alpha), PSDVar_offset),  Domain.equalsTo(0) )
    
        # Solve
        M.writeTask("App1.ptf")                # Save problem in readable format
        M.solve()
    
        #Get results
        print("Objective value restriction ={0}".format(obj.level()**2))
        #print(PSDVar.level().reshape(n+1,n+1))
        test = 0
        Qsol = Q.level().reshape(n,n)
        qsol = q.level()
        csol = c.level()
        for i in range(p):
            w = wlist[i]
            test+= (z[i]-0.5*w.dot(Qsol).dot(w) - w.dot(qsol) - csol)**2
        sol_time =  M.getSolverDoubleInfo("optimizerTime")
        return Qsol,qsol,csol,obj.level()[0]**2


def master(name,n,wlist,z,Qxk_list,qxk_list, vxk_vector,yklist,mu):
    "Solve the master problem"
    p = len(z)
    wlist_square_flattened = np.array([(w.reshape(n,1).dot(w.reshape(1,n))).reshape(n**2) for w in wlist]) 
    # #for each vector w, we create a matrix n x n (given by w^T * w) and then a vector n**2
    
    #Definition of the LL polytope : [0,1]^n box
    A = np.concatenate([np.eye(n),-np.eye(n)])
    b = np.concatenate([np.ones(n),np.zeros(n)])
    rho = np.sqrt(n)
    
    with Model("App1") as M:
        
        #Auxiliary var
        obj = M.variable("obj", 1, Domain.unbounded())
        objrelax = M.variable("objrelax", 1, Domain.unbounded())
        distance_term = M.variable("dist", 3, Domain.unbounded())
        
        #Upper level var
        Q = M.variable("Q", [n,n], Domain.unbounded())
        q = M.variable("q", n, Domain.unbounded())
        c = M.variable("c", Domain.unbounded())
        Qrelax = M.variable("Qrelax", [n,n], Domain.unbounded())
        qrelax = M.variable("qrelax", n, Domain.unbounded())
        crelax = M.variable("crelax", Domain.unbounded())
        K = len(vxk_vector)
        
        #LL variables
        lam = M.variable("lambda", len(b), Domain.greaterThan(0.0))
        eta = M.variable("eta", K, Domain.greaterThan(0.0))
        alpha = M.variable("alpha", Domain.greaterThan(0.0))
        beta = M.variable("beta", Domain.unbounded())
        
        #Vars for PSD constraint
        PSDVar = M.variable(Domain.inPSDCone(n+1)) #the whole matrix that must be PSD in (27)
        PSDVar_main = PSDVar.slice([0,0], [n,n]) #we take the first submatrix n x n in each component of the sum in (27) i.e. 0.5*Q, 0_n, alpha*I_n
        PSDVar_vec = Var.flatten(PSDVar.slice([0,n], [n,n+1])) #we take the second submatrix n x 1 i.e. 0.5*q, \sum_r(lambda_r*A)
        PSDVar_offset = PSDVar.slice([n,n], [n+1,n+1])   #we take the third submatrix 1 x 1 i.e. \beta, alpha*1
                
        #Objective
        deg0term_obj = Expr.mul(c,np.ones(p))
        deg1term_obj = Expr.mul(wlist, q)
        deg2term_obj = Expr.mul(0.5,Expr.mul(wlist_square_flattened, Var.flatten(Q)))
        prediction_term =  Expr.add(deg2term_obj,Expr.add(deg1term_obj,deg0term_obj)) #predicted z (p points)
        M.constraint( Expr.vstack(obj, Expr.sub(prediction_term, z)), Domain.inQCone() ) #see below -- vstack put obj "above" (z-zpredicted) 
        
        deg0term_objrelax = Expr.mul(crelax,np.ones(p))
        deg1term_objrelax = Expr.mul(wlist, qrelax)
        deg2term_objrelax = Expr.mul(0.5,Expr.mul(wlist_square_flattened, Var.flatten(Qrelax)))
        prediction_term_objrelax =  Expr.add(deg2term_objrelax,Expr.add(deg1term_objrelax,deg0term_objrelax)) #predicted z (p points)
        M.constraint( Expr.vstack(objrelax, Expr.sub(prediction_term_objrelax, z)), Domain.inQCone() ) #see below -- vstack put obj "above" (z-zpredicted) 
        
        
        M.constraint( Expr.vstack(1.0,Expr.vstack(distance_term.index(0), Expr.sub(Var.flatten(Q), Var.flatten(Qrelax)))), Domain.inRotatedQCone() ) #see below -- vstack put obj "above" (z-zpredicted) 
        M.constraint( Expr.vstack(1.0,Expr.vstack(distance_term.index(1), Expr.sub(q, qrelax))), Domain.inRotatedQCone() ) #see below -- vstack put obj "above" (z-zpredicted) 
        M.constraint( Expr.vstack(1.0,Expr.vstack(distance_term.index(2), Expr.sub(c, crelax))), Domain.inRotatedQCone() ) #see below -- vstack put obj "above" (z-zpredicted) 
        
        
        M.objective(ObjectiveSense.Minimize, Expr.add(Expr.add(obj,objrelax),Expr.mul(mu,Expr.sum(distance_term)) ))
        
        #Symmetry constraint for Q
        M.constraint( Expr.sub(Q, Q.transpose()),  Domain.equalsTo(0,n,n) ) #Q-Q^T = 0
       
        # c - (\lambda^T b + \alpha (1+ rho^2) + beta) + \eta^T v >=0 
        LL_obj_expr = Expr.add(Expr.dot(lam,b),Expr.add(Expr.mul((1+rho**2),alpha),beta))
        
        #Constraints to define the several parts of the PSD matrix
        if K>=1:
            M.constraint(Expr.add(Expr.dot(eta,vxk_vector),Expr.sub(c,LL_obj_expr)),Domain.greaterThan(0.0))
        
            combiliMat = np.zeros((n,n))
            combiliVect = np.zeros(n)
            for i in range(K):
                combiliMat = Expr.add(combiliMat, Expr.mul(eta.index(i),Qxk_list[i]))
                combiliVect = Expr.add(combiliVect, Expr.mul(eta.index(i),qxk_list[i]))
            M.constraint(Expr.sub(Expr.add(Expr.sub(Expr.mul(0.5,Q),Expr.mul(0.5,combiliMat)), Expr.mul(alpha,np.eye(n))), PSDVar_main),  Domain.equalsTo(0,n,n) )  
            M.constraint( Expr.sub(Expr.add(Expr.sub(Expr.mul(0.5,q),Expr.mul(0.5,combiliVect)), Expr.mul(lam,0.5*A)), PSDVar_vec),  Domain.equalsTo(0,n) )
            M.constraint( Expr.sub(Expr.add(beta, alpha), PSDVar_offset),  Domain.equalsTo(0) )
        else:
            M.constraint(Expr.sub(c,LL_obj_expr),Domain.greaterThan(0.0))
        
            #Constraints to define the several parts of the PSD matrix
            M.constraint(Expr.sub(Expr.add(Expr.mul(0.5,Q), Expr.mul(alpha,np.eye(n))), PSDVar_main),  Domain.equalsTo(0,n,n) )  
            M.constraint( Expr.sub(Expr.add(Expr.mul(0.5,q), Expr.mul(lam,0.5*A)), PSDVar_vec),  Domain.equalsTo(0,n) )
            M.constraint( Expr.sub(Expr.add(beta, alpha), PSDVar_offset),  Domain.equalsTo(0) )
        
    
        #Constraints for the relaxation variables
        for k in range(K):
            y = yklist[k]
            Y = (y.reshape(n,1).dot(y.reshape(1,n))).reshape(n**2)
            M.constraint(Expr.add(Expr.add(Expr.dot(Var.flatten(Qrelax),0.5*Y.flatten()),Expr.dot(qrelax,y)),crelax), Domain.greaterThan(0))
    
        # Solve
        M.acceptedSolutionStatus(AccSolutionStatus.Anything)
        M.writeTask("App1.ptf")                # Save problem in readable format
        M.solve()

        
        test = 0
        Qsol = Q.level().reshape(n,n)
        qsol = q.level()
        csol = c.level()
        Qsolrelax = Qrelax.level().reshape(n,n)
        qsolrelax = qrelax.level()
        csolrelax = crelax.level()
        S = 0.5*(np.linalg.norm(Qsol-Qsolrelax,2)**2 + np.linalg.norm(qsol-qsolrelax,2)**2 + (csol-csolrelax)**2)
        assert(np.sum(distance_term.level())>=S-1E-9)
        for i in range(p):
            w = wlist[i]
            test+= (z[i]-0.5*w.dot(Qsol).dot(w) - w.dot(qsol) - csol)**2

        sol_time =  M.getSolverDoubleInfo("optimizerTime")
        return Qsol,qsol,csol,Qsolrelax,qsolrelax,csolrelax,test,objrelax.level()[0]**2, S
    
    
def solve_subproblem_App1(n,Q,q,c,tl):
    m = gp.Model("LL problem")
    m.Params.LogToConsole = 0
    y = m.addMVar(n, lb = 0.0, ub = 1.0, name="y")
    Qtimeshalf = 0.5*Q
    m.setObjective(y@Qtimeshalf@y+  q@y +c, GRB.MINIMIZE)
    m.setParam('NonConvex', 2)
    m.setParam('TimeLimit', tl)
    m.optimize()
    return y.X, m.objVal
