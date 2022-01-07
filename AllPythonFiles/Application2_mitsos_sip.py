from DimacsReader import DimacsReader
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import pandas as pd


def save(name,finished,value,ub,soltime,iteration, xsol):
    f = open("../output/Application2/"+name+"/mitsos_sip.txt","w+")
    if finished==True:
        f.write("Finished before time limit.\n")
    else:
        f.write("Time limit reached.\n")
    f.write("Obj: "+str(value)+"\n")
    if finished==False:
        f.write("Upper bound: "+str(ub)+"\n")
    f.write("SolTime: "+str(soltime)+"\n")
    f.write("It. number: "+str(iteration)+"\n")
    f.write("\nUpper level solution: "+str(xsol)+"\n")
    f.close()

def main_app2(name_dimacs,name,r=10,timelimit=18000):
    #Logs
    UpperBoundsLogs,LowerBoundsLogs, EpsLogs, MasterTimeLogs, LLTimeLogs = [],[],[],[],[]
    yubd = []
    ub = np.inf
    eps_r = 0.1
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
    t0 = time.time()
    relax = gp.Model("relax problem")
    xvar = relax.addMVar(n,lb=0,ub=1,name='x')
    zvar = relax.addMVar(1,name='z',lb=-GRB.INFINITY,ub=GRB.INFINITY)
    relax.addConstr(np.ones(n)@xvar == 1)
    relax.setObjective(zvar+xvar@(0.5*Q1)@xvar + q1@xvar, GRB.MINIMIZE)
    #First cut for boundedness
    y = np.ones(n)*(1/n)
    Y = (y.reshape(n,1).dot(y.reshape(1,n)))
    coeffC = np.array([0.5*Y[i,i] * diagonalQ2x[i] for i in range(n)])
    relax.addConstr(zvar+coeffC@xvar+ (y@M)@xvar + q2@y + y@(0.5*Q2)@y >=0)
    running = True
    iteration,ub = 0,np.inf
    while running and (time.time()-t0<timelimit):
        #Solve LBD
        t1 = time.time()
        relax.optimize()
        lb = relax.objVal
        relaxtime = time.time() - t1
        x,z = xvar.X, zvar.X
        Q = Q2+np.diag(diagonalQ2x*x)
        b = q2 + (M.T)@x
        #Solve LLP
        t1 = time.time()
        tl = 10+max(0,timelimit-(t1-t0))
        y,val = solve_subproblem_App2(n,Q,b,z,tl)
        LLtime1 = time.time() - t1
        iteration+=1
        if val>-1E-6:
            ub = lb
            running=False
        else:
            Y = (y.reshape(n,1).dot(y.reshape(1,n)))
            coeffC = np.array([0.5*Y[i,i] * diagonalQ2x[i] for i in range(n)])
            relax.addConstr(zvar+coeffC@xvar+ (y@M)@xvar + q2@y + y@(0.5*Q2)@y >=0)
            
        if running:
            t1 = time.time()
            feasible,xres,zres = ubd_problem(n,Q1,q1,Q2,q2,M,diagonalQ2x,yubd,eps_r)
            restime = time.time() - t1
            if feasible:
                t1 = time.time()
                tl = 10+max(0,timelimit-(t1-t0))
                Qres = Q2+np.diag(diagonalQ2x*x)
                bres = q2 + (M.T)@x
                y,valres = solve_subproblem_App2(n,Qres,bres,zres,tl)
                LLtime2 =time.time() - t1
               
                if valres>-1E-6:
                    eps_r = eps_r/r
                    ub = min(ub,zres+xres@(0.5*Q1)@xres + q1@xres)
                else:
                    yubd.append(y)
            else:
                eps_r = eps_r/r
        
        
        running = ub-lb>1E-6      
        
        
        
        #Log
        UpperBoundsLogs.append(ub)
        LowerBoundsLogs.append(lb)
        EpsLogs.append(val)
        MasterTimeLogs.append(relaxtime+restime)
        LLTimeLogs.append(LLtime1+LLtime2)
        
       
    soltime = time.time() - t0
    save(name,not(running),relax.objVal,ub,soltime,iteration,x)
    df = pd.DataFrame()
    df['UB'],df['LB'],df["Epsilon"],df["MasterTime"],df['LLTime'] = UpperBoundsLogs, LowerBoundsLogs, EpsLogs, MasterTimeLogs, LLTimeLogs
    df.to_csv("../output/Application2/"+name+"/mitsos_sip.csv")

def solve_subproblem_App2(n,Q,b,z,tl):
    m = gp.Model("LL problem")
    y = m.addMVar(n, lb = 0.0, ub = 1.0, name="y")
    m.addConstr(np.ones(n)@y==1)
    m.setObjective(y@(0.5*Q)@y+  b@y +z, GRB.MINIMIZE)
    m.setParam('NonConvex', 2)
    m.setParam('TimeLimit', tl)
    m.optimize()
    return y.X, m.objVal

def ubd_problem(n,Q1,q1,Q2,q2,M,diagonalQ2x,yubd,eps_r):
    model = gp.Model("relax problem")
    xvar = model.addMVar(n,lb=0,ub=1,name='x')
    zvar = model.addMVar(1,name='z',lb=-GRB.INFINITY,ub=GRB.INFINITY)
    model.addConstr(np.ones(n)@xvar == 1)
    model.setObjective(zvar+xvar@(0.5*Q1)@xvar + q1@xvar, GRB.MINIMIZE)
    #First cut for boundedness
    y = np.ones(n)*(1/n)
    Y = (y.reshape(n,1).dot(y.reshape(1,n)))
    coeffC = np.array([0.5*Y[i,i] * diagonalQ2x[i] for i in range(n)])
    model.addConstr(zvar+coeffC@xvar+ (y@M)@xvar + q2@y + y@(0.5*Q2)@y >=eps_r)
    for y in yubd:
        Y = (y.reshape(n,1).dot(y.reshape(1,n)))
        coeffC = np.array([0.5*Y[i,i] * diagonalQ2x[i] for i in range(n)])
        model.addConstr(zvar+coeffC@xvar+ (y@M)@xvar + q2@y + y@(0.5*Q2)@y >=eps_r)
    model.optimize()

    
    if model.status in [3,4]:
        return False, np.zeros(0),np.zeros(0)
    else:
        return True, xvar.X, zvar.X

 
