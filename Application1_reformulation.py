# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:22:49 2020

@author: aoust
"""

from mosek.fusion import *
import sys
import numpy as np
import time

def save(name, value,value_tested, soltime, bigQ, q,c):
    f = open("Application1_data/"+name+"/reformulation_obj_value.txt","w+")
    f.write("Obj value returned by the solver: "+str(value)+"\n")
    f.write("Obj value computed with solution variables: "+str(f.write("Obj: "+str(value)+"\n"))+"\n")
    f.write("SolTime: "+str(soltime)+"\n")
    f.write("Q matrix recovered: " +str(bigQ) +"\n")
    f.write("q vector recovered: " +str(q) +"\n")
    f.write("Scalar c recovered: " +str(c) +"\n")
    f.close()

def main(name):
    #Loading data
    Qref = np.load("Application1_data/"+name+"/bigQref.npy")
    qref = np.load("Application1_data/"+name+"/qref.npy")
    n = len(qref)
    cref = float(np.load("Application1_data/"+name+"/cref.npy"))
    wlist = np.load("Application1_data/"+name+"/w.npy")
    noise = np.load("Application1_data/"+name+"/noise.npy")
    z = np.load("Application1_data/"+name+"/z.npy")
    noiseless_z=0.5 *np.array([ w.dot(Qref).dot(w) for w in wlist]) + wlist.dot(qref) + cref
    print(np.linalg.norm(z-noiseless_z-noise))
    p = len(z)
    
    wlist_square_flattened = np.array([(w.reshape(n,1).dot(w.reshape(1,n))).reshape(n**2) for w in wlist]) 
    # #for each vector w, we create a matrix n x n (given by w^T * w) and then a vector n**2
    
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
        #M.constraint( Expr.sub(Expr.add(Expr.mul(0.5,q), Expr.mul(lam,0.5*A)), PSDVar_vec),  Domain.equalsTo(0,n) )
        M.constraint( Expr.sub(Expr.add(beta, alpha), PSDVar_offset),  Domain.equalsTo(0) )
    
    
        # Solve
        M.setLogHandler(sys.stdout)            # Add logging
        M.writeTask("App1.ptf")                # Save problem in readable format
        M.solve()
    
        #Get results
        print("Objective value ={0}".format(obj.level()**2))
        print(PSDVar.level().reshape(n+1,n+1))
        test = 0
        Qsol = Q.level().reshape(n,n)
        qsol = q.level()
        csol = c.level()
        for i in range(p):
            w = wlist[i]
            test+= (z[i]-0.5*w.dot(Qsol).dot(w) - w.dot(qsol) - csol)**2
        print(test,obj.level()**2)
        sol_time =  M.getSolverDoubleInfo("optimizerTime")
        save(name,obj.level()[0]**2,test,sol_time,Qsol,qsol,csol)
        print("Matrix Q")
        print(Qsol)
        print("Matrix Q_ref")
        print(Qref)
        print("Vector q")
        print(qsol)
        print("Vector q_ref")
        print(qref)
        print("Average square error reconstruction ={0}".format(obj.level()**2/p))
        print("Average square error data ={0}".format(np.array([n**2 for n in noise]).sum()/p))
        print("TODO : also print error between ref and reconstructed output")
    

if __name__ == "__main__":
    main(sys.argv[1])
    # # Sample input data
    # n = 5
    # #asym = np.array([[3.7,-0.2,0,1,3],[2,2,1,0,0],[0,0,3,1,2],[1,0,0.4,5,2],[1.2,0,0.4,-1,3]])
    # asym = 5*np.random.rand(n,n)
    # Qref = 0.5*(asym + asym.transpose())
    # #qref = np.array([-2.3,-2,1,2,1])
    # qref = 3*np.random.rand(n)
    # #cref = 2
    # cref= 2*np.random.rand()
    # sigma = 0.3
    # p = 5000
    # wlist = np.random.rand(p,n)
    # noise = np.random.normal(loc = 0, scale = sigma, size = p)
    # noiseless_z=0.5 *np.array([ w.dot(Qref).dot(w) for w in wlist]) + wlist.dot(qref) + cref
    # print(noiseless_z.min())
    # z = noiseless_z + noise
    # wlist_square_flattened = np.array([(w.reshape(n,1).dot(w.reshape(1,n))).reshape(n**2) for w in wlist]) 
    # #for each vector w, we create a matrix n x n (given by w^T * w) and then a vector n**2
    
    # #Definition of the LL polytope : [0,1]^n box
    # A = np.concatenate([np.eye(n),-np.eye(n)])
    # b = np.concatenate([np.ones(n),np.zeros(n)])
    # rho = np.sqrt(n)
    
    # #Write the .dat file
    # moment=time.strftime("%Y-%b-%d__%H_%M_%S",time.localtime()) #to have different .dat files for each python run
    # f = open("./instance"+moment+".dat","w")
    # f.write("param n := %d;\n"%n)
    # f.write("param p_max := %d;\n"%p)
    # f.write("param r_dim := 0;\n")
    # f.write("\nparam Q_ref :")
    # for i in range(n):
    #     f.write(" %d "%(i+1))
    # f.write(":=\n")
    # for i in range(n):
    #     for j in range(n):
    #         if j==0:
    #             f.write("   %d "%(i+1))
    #         f.write("%f "%Qref[i,j])
    #         if j==(n-1):
    #             if i==(n-1):
    #                 f.write(";")
    #             f.write("\n")
    
    # f.write("\nparam q_ref := ")
    # for i in range(n):
    #     f.write("%d "%(i+1))
    #     f.write("%f "%qref[i])
    # f.write(";\n\nparam c_ref := %f ;\n\n"%cref)
    # f.write("\nparam w:")
    # for i in range(n):
    #     f.write(" %d "%(i+1))
    # f.write(":=\n")
    # for i in range(p):
    #     for j in range(n):
    #         if j==0:
    #             f.write("   %d "%(i+1))
    #         f.write("%f "%wlist[i,j])
    #         if j==(n-1):
    #             if i==(p-1):
    #                 f.write(";")
    #             f.write("\n")
    # f.write("\nparam epsilon := ")
    # for i in range(p):
    #     f.write("%d "%(i+1))
    #     f.write("%f "%noise[i])
    # f.write(";\n")
    # f.write("\nparam z := ")
    # for i in range(p):
    #     f.write("%d "%(i+1))
    #     f.write("%f "%z[i])
    # f.write(";")
    # f.close()