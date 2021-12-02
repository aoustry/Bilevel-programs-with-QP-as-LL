# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:01:47 2021

@author: aoust
"""
from DimacsReader import DimacsReader
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time


def save(name, value,soltime, itnumber, xsol):
    f = open("Application2_data/"+name+"/cutting_planes.txt","w+")
    f.write("Obj: "+str(value)+"\n")
    f.write("SolTime: "+str(soltime)+"\n")
    f.write("It. number: "+str(itnumber)+"\n")
    f.write("Upper level solution: "+str(xsol)+"\n")
    f.close()

def main_app2(name_dimacs,name):
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
    t0 = time.time()
       
    
    master = gp.Model("Master problem")
    xvar = master.addMVar(n,lb=0,ub=1,name='x')
    vvar = master.addMVar(1,name='v',lb=-GRB.INFINITY,ub=GRB.INFINITY)
    master.addConstr(np.ones(n)@xvar == 1)
    master.setObjective(vvar+xvar@(0.5*Q1)@xvar + q1@xvar, GRB.MINIMIZE)
    #First cut for boundedness
    y = np.ones(n)*(1/n)
    Y = (y.reshape(n,1).dot(y.reshape(1,n)))
    coeffC = np.array([0.5*Y[i,i] * diagonalQ2x[i] for i in range(n)])
    master.addConstr(vvar+coeffC@xvar+ (y@M)@xvar + q2@y + y@(0.5*Q2)@y >=0)
    running = True
    itnumber = 0
    while running:
        master.optimize()
        x,v = xvar.X, vvar.X
        Q = Q2+np.diag(diagonalQ2x*x)
        b = q2 + (M.T)@x
        y,val = solve_subproblem_App2(n,Q,b,v)
        itnumber+=1
        
        if val>-1E-6:
            running=False
        else:
            Y = (y.reshape(n,1).dot(y.reshape(1,n)))
            coeffC = np.array([0.5*Y[i,i] * diagonalQ2x[i] for i in range(n)])
            master.addConstr(vvar+coeffC@xvar+ (y@M)@xvar + q2@y + y@(0.5*Q2)@y >=0)
    soltime = time.time() - t0
    save(name, master.objVal,soltime,itnumber, x)

def solve_subproblem_App2(n,Q,b,v):
    m = gp.Model("LL problem")
    y = m.addMVar(n, lb = 0.0, ub = 1.0, name="y")
    m.addConstr(np.ones(n)@y==1)
    m.setObjective(y@(0.5*Q)@y+  b@y +v, GRB.MINIMIZE)
    m.setParam('NonConvex', 2)
    m.optimize()
    return y.X, m.objVal


   