import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from itertools import combinations
import pandas as pd


def save(name,finished,p,value,soltime,iteration, bigQ,q,c):
    f = open("../output/Application1/"+name+"/mitsos_sip.txt","w+")
    if finished==True:
        f.write("Finished before time limit.\n")
    else:
        f.write("Time limit reached.\n")
    f.write("Obj value returned by the CP solver: "+str(value)+"\n")
    f.write("Average LSE: {0}\n".format(value/p))
    f.write("SolTime: "+str(soltime)+"\n")
    f.write("It. number: "+str(iteration)+"\n")
    f.write("\nQ matrix recovered: " +str(bigQ) +"\n")
    f.write("q vector recovered: " +str(q) +"\n")
    f.write("Scalar c recovered: " +str(c) +"\n")
    f.close()

def main_app1(name,r=10,timelimit=18000):
    #Logs
    UpperBoundsLogs,LowerBoundsLogs, EpsLogs, MasterTimeLogs, LLTimeLogs = [], [],[],[],[]
    yubd = []
    ub = np.inf
    eps_r = 0.1
    
    #Loading data
    wlist = np.load("../Application1_data/"+name+"/w.npy")
    p,n = wlist.shape
    z = np.load("../Application1_data/"+name+"/z.npy")
    wlist_square_flattened = np.array([(w.reshape(n,1).dot(w.reshape(1,n))).reshape(n**2) for w in wlist])
    
    t0 = time.time()
    relaxation = gp.Model("relax")
    flattenedQvar = relaxation.addMVar(n**2,lb=-GRB.INFINITY,ub=GRB.INFINITY,name='flattenedQmatrix')
    spread = relaxation.addMVar(p,lb=-GRB.INFINITY,ub=GRB.INFINITY)
    qvar = relaxation.addMVar(n,name='qvector',lb=-GRB.INFINITY,ub=GRB.INFINITY)
    cvar = relaxation.addMVar(1,name='c',lb=-GRB.INFINITY,ub=GRB.INFINITY)
    relaxation.addConstr(spread== z-0.5*wlist_square_flattened@flattenedQvar - wlist@qvar - np.ones((p,1))@cvar)
    for (i,j) in combinations(range(n),2):
          relaxation.addConstr(flattenedQvar[i*n+j]==flattenedQvar[j*n+i])
    relaxation.setObjective(spread@spread, GRB.MINIMIZE)
    running, iteration = True, 0
    
    while running and (time.time()-t0<timelimit):
        #Solve relaxation (LBD)
        t1 = time.time()
        relaxation.optimize()
        relaxtime = time.time() - t1
        lb = relaxation.objVal
        flattenedQ,q,c = flattenedQvar.X, qvar.X, cvar.X
        Q = flattenedQ.reshape((n,n))
        iteration+=1
        
        #Solve (LLP)
        t1 = time.time()
        tl = 10+max(0,timelimit-(t1-t0))
        y,val = solve_subproblem_App1(n,Q,q,c,tl)
        LLtime1 = time.time() - t1
        restime, LLtime2 = 0,0
        if val>-1E-6:
            ub = lb
            running = False
        else:
            Y = (y.reshape(n,1).dot(y.reshape(1,n))).flatten()
            relaxation.addConstr(0.5*Y@flattenedQvar + y@qvar + cvar >=0)
        
        #Solve (UBD)
        if running:
            t1 = time.time()
            feasible, fQres,qres,cres = ubd_problem(n,p,wlist,wlist_square_flattened,z,yubd,eps_r)
            restime = time.time() - t1
            if feasible:
                #Solve (LLP)
                t1 = time.time()
                tl = 10+max(0,timelimit-(t1-t0))
                Qres = fQres.reshape((n,n))
                y,valres = solve_subproblem_App1(n,Qres,qres,cres,tl)
                LLtime2 =time.time() - t1
               
                if valres>-1E-6:
                    eps_r = eps_r/r
                    ub = min(ub,objective_value(n,Qres,qres,cres,wlist,z))
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
    save(name,not(running),p,relaxation.objVal,soltime,iteration, flattenedQ,q,c)
    df = pd.DataFrame()
    df['UB'],df['LB'],df["Epsilon"],df["MasterTime"],df['LLTime'] = UpperBoundsLogs, LowerBoundsLogs, EpsLogs, MasterTimeLogs, LLTimeLogs
    df.to_csv("../output/Application1/"+name+"/mitsos_sip.csv")

def solve_subproblem_App1(n,Q,q,c,tl):
    m = gp.Model("LL problem")
    y = m.addMVar(n, lb = 0.0, ub = 1.0, name="y")
    Qtimeshalf = 0.5*Q
    m.setObjective(y@Qtimeshalf@y+  q@y +c, GRB.MINIMIZE)
    m.setParam('NonConvex', 2)
    m.setParam('TimeLimit', tl)
    m.optimize()
    return y.X, m.objVal


def objective_value(n,Q,q,c,wlist,z):
    val,i = 0,0
    for w in wlist:
        val+=(z[i]-0.5*w.dot(Q.dot(w))-q.dot(w)-c)**2
        i+=1
    return val

def ubd_problem(n,p,wlist,wlist_square_flattened,z,yubd,eps_r):
    model = gp.Model("relaxation problem")
    fQ_ubd = model.addMVar(n**2,lb=-GRB.INFINITY,ub=GRB.INFINITY,name='flattenedQmatrix')
    spread = model.addMVar(p,lb=-GRB.INFINITY,ub=GRB.INFINITY)
    q_ubd = model.addMVar(n,name='qvector',lb=-GRB.INFINITY,ub=GRB.INFINITY)
    c_ubd = model.addMVar(1,name='c',lb=-GRB.INFINITY,ub=GRB.INFINITY)
    model.addConstr(spread== z-0.5*wlist_square_flattened@fQ_ubd - wlist@q_ubd - np.ones((p,1))@c_ubd)
    for (i,j) in combinations(range(n),2):
          model.addConstr(fQ_ubd[i*n+j]==fQ_ubd[j*n+i])
    for y in yubd:
        Y = (y.reshape(n,1).dot(y.reshape(1,n))).flatten()
        model.addConstr(0.5*Y@fQ_ubd + y@q_ubd + c_ubd >=eps_r)
    model.setObjective(spread@spread, GRB.MINIMIZE)
    model.optimize()

    
    if model.status in [3,4]:
        return False, np.zeros(0),np.zeros(0),np.zeros(0)
    else:
        return True, fQ_ubd.X, q_ubd.X, c_ubd.X
        
