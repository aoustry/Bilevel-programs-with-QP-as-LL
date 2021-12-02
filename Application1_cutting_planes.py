# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:01:47 2021

@author: aoust
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from itertools import combinations

def save(name, p,value, soltime, bigQ, q,c):
    f = open("Application1_data/"+name+"/cutting_plane.txt","w+")
    f.write("Obj value returned by the CP solver: "+str(value)+"\n")
    f.write("Average LSE = {0}".format(value/p))
    f.write("SolTime: "+str(soltime)+"\n")
    f.write("Q matrix recovered: " +str(bigQ) +"\n")
    f.write("q vector recovered: " +str(q) +"\n")
    f.write("Scalar c recovered: " +str(c) +"\n")
    f.close()

def main(name):
    #Loading data
    wlist = np.load("Application1_data/"+name+"/w.npy")
    p,n = wlist.shape
    z = np.load("Application1_data/"+name+"/z.npy")
    wlist_square_flattened = np.array([(w.reshape(n,1).dot(w.reshape(1,n))).reshape(n**2) for w in wlist]) 
    t0 = time.time()
    master = gp.Model("Master problem")
    flattenedQvar = master.addMVar(n**2,lb=-GRB.INFINITY,ub=GRB.INFINITY,name='flattenedQmatrix')
    spread = master.addMVar(p,lb=-GRB.INFINITY,ub=GRB.INFINITY)
    qvar = master.addMVar(n,name='qvector',lb=-GRB.INFINITY,ub=GRB.INFINITY)
    cvar = master.addMVar(1,name='c',lb=-GRB.INFINITY,ub=GRB.INFINITY)
    master.addConstr(spread== z-0.5*wlist_square_flattened@flattenedQvar - wlist@qvar - np.ones((p,1))@cvar)
    for (i,j) in combinations(range(n),2):
          master.addConstr(flattenedQvar[i*n+j]==flattenedQvar[j*n+i])
    master.setObjective(spread@spread, GRB.MINIMIZE)
    running = True
    while running:
        master.optimize()
        flattenedQ,q,c = flattenedQvar.X, qvar.X, cvar.X
        Q = flattenedQ.reshape((n,n))
        y,val = solve_subproblem_App1(n,Q,q,c)
        if val>-1E-6:
            running=False
        else:
            Y = (y.reshape(n,1).dot(y.reshape(1,n))).flatten()
            master.addConstr(0.5*Y@flattenedQvar + y@qvar + cvar >=0)
    soltime = time.time() - t0
    save(name, p,master.objVal, soltime, flattenedQ, q,c)

def solve_subproblem_App1(n,Q,q,c):
    m = gp.Model("LL problem")
    y = m.addMVar(n, lb = 0.0, ub = 1.0, name="y")
    Qtimeshalf = 0.5*Q
    m.setObjective(y@Qtimeshalf@y+  q@y +c, GRB.MINIMIZE)
    m.setParam('NonConvex', 2)
    m.optimize()
    return y.X, m.objVal

main('notPSD_random8')
# if __name__ == "__main__":
#     main(sys.argv[1])
   