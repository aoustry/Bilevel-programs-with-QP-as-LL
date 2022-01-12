from DimacsReader import DimacsReader
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import pandas as pd


def save(name,finished,value,ub,soltime,iteration,inner, xsol):
    f = open("../output/Application2/"+name+"/cutting_planes.txt","w+")
    if finished==True:
        f.write("Finished before time limit.\n")
    else:
        f.write("Time limit reached.\n")    
    f.write("Obj: "+str(value)+"\n")
    if finished==False:
        f.write("Upper bound: "+str(ub)+"\n")
    f.write("SolTime: "+str(soltime)+"\n")
    f.write("It. number: "+str(iteration)+"\n")
    f.write("Percent. inner: "+str(inner)+"\n")  
    f.write("\nUpper level solution: "+str(xsol)+"\n")
    f.close()

def main_app2(name_dimacs,name,timelimit=18000):
    #Logs
    UpperBoundsLogs,LowerBoundsLogs, EpsLogs, MasterTimeLogs, LLTimeLogs = [],[],[],[],[]
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
    mastertime_tot = 0
    LLtime_tot = 0
    t0 = time.time()
    master = gp.Model("Master problem")
    xvar = master.addMVar(n,lb=0,ub=1,name='x')
    zvar = master.addMVar(1,name='z',lb=-GRB.INFINITY,ub=GRB.INFINITY)
    master.addConstr(np.ones(n)@xvar == 1)
    master.setObjective(zvar+xvar@(0.5*Q1)@xvar + q1@xvar, GRB.MINIMIZE)
    #First cut for boundedness
    y = np.ones(n)*(1/n)
    Y = (y.reshape(n,1).dot(y.reshape(1,n)))
    coeffC = np.array([0.5*Y[i,i] * diagonalQ2x[i] for i in range(n)])
    master.addConstr(zvar+coeffC@xvar+ (y@M)@xvar + q2@y + y@(0.5*Q2)@y >=0)
    running = True
    iteration = 0
    while running and (time.time()-t0<timelimit):
        iteration+=1
        t1 = time.time()
        print("----------------Master problem--------------")
        master.optimize()
        mastertime = time.time() - t1
        mastertime_tot = mastertime_tot + mastertime
        x,z = xvar.X, zvar.X
        Q = Q2+np.diag(diagonalQ2x*x)
        b = q2 + (M.T)@x
        
        t1 = time.time()
        tl = 10+max(0,timelimit-(t1-t0))
        print("----------------Inner problem--------------", name)
        y,val = solve_subproblem_App2(n,Q,b,z,tl)
        LLtime = time.time() - t1
        LLtime_tot = LLtime_tot + LLtime
        
        
        #Log
        UpperBoundsLogs.append(master.objVal+max(0,-val))
        LowerBoundsLogs.append(master.objVal)
        EpsLogs.append(val)
        MasterTimeLogs.append(mastertime)
        LLTimeLogs.append(LLtime)
        
        if val>-1E-6:
            running=False
        else:
            Y = (y.reshape(n,1).dot(y.reshape(1,n)))
            coeffC = np.array([0.5*Y[i,i] * diagonalQ2x[i] for i in range(n)])
            master.addConstr(zvar+coeffC@xvar+ (y@M)@xvar + q2@y + y@(0.5*Q2)@y >=0)
    soltime = time.time() - t0
    percentLL = LLtime_tot/(LLtime_tot+mastertime_tot)
    
    save(name,not(running), master.objVal,master.objVal+max(0,-val),soltime,iteration,percentLL, x)
    df = pd.DataFrame()
    df['UB'],df['LB'],df["Epsilon"],df["MasterTime"],df['LLTime'] = UpperBoundsLogs, LowerBoundsLogs, EpsLogs, MasterTimeLogs, LLTimeLogs
    df.to_csv("../output/Application2/"+name+"/cutting_plane.csv")

def solve_subproblem_App2(n,Q,b,z,tl):
    m = gp.Model("LL problem")
    y = m.addMVar(n, lb = 0.0, ub = 1.0, name="y")
    m.addConstr(np.ones(n)@y==1)
    m.setObjective(y@(0.5*Q)@y+  b@y +z, GRB.MINIMIZE)
    m.setParam('NonConvex', 2)
    m.setParam('TimeLimit', tl)
    m.optimize()
    return y.X, m.objVal

