# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 10:38:36 2020

@author: aoust
"""
import numpy as np
from DimacsReader import *
import time
import os


def create_files(name,n,Q1,Q2,q1,q2,M):
   #Reading graph file
    
    
    #Write the .dat file
    os.mkdir("Application2_data/"+name)
    f = open("Application2_data/"+name+"/instance.dat","w")
    f.write("param n := %d;\n"%n)
    f.write("\nparam M :")
    for i in range(n):
        f.write(" %d "%(i+1))
    f.write(":=\n")
    for i in range(n):
        for j in range(n):
            if j==0:
                f.write("   %d "%(i+1))
            f.write("%f "%M[i,j])
            if j==(n-1):
                if i==(n-1):
                    f.write(";")
                f.write("\n")
    f.write("\nparam Q1 :")
    for i in range(n):
        f.write(" %d "%(i+1))
    f.write(":=\n")
    for i in range(n):
        for j in range(n):
            if j==0:
                f.write("   %d "%(i+1))
            f.write("%f "%Q1[i,j])
            if j==(n-1):
                if i==(n-1):
                    f.write(";")
                f.write("\n")
                
    f.write("\nparam Q2_fix :")
    for i in range(n):
        f.write(" %d "%(i+1))
    f.write(":=\n")
    for i in range(n):
        for j in range(n):
            if j==0:
                f.write("   %d "%(i+1))
            f.write("%f "%Q2[i,j])
            if j==(n-1):
                if i==(n-1):
                    f.write(";")
                f.write("\n")
    
    f.write("\nparam q1 := ")
    for i in range(n):
        f.write("%d "%(i+1))
        f.write("%f "%q1[i])
    f.write(";")
        
    f.write("\nparam q2_fix := ")
    for i in range(n):
        f.write("%d "%(i+1))
        f.write("%f "%q2[i])
    f.write(";")
    f.close()
    
    #Write the numpy files 
    np.save("Application2_data/"+name+"/bigQ1",Q1)
    np.save("Application2_data/"+name+"/bigQ2_fix",Q2)
    np.save("Application2_data/"+name+"/q1",q1)
    np.save("Application2_data/"+name+"/q2_fix",q2)
    np.save("Application2_data/"+name+"/M",M)
    
    

def create_files_deterministic_type1(name):
    """Create the files for an instance linked to a DIMACS graph called "name"
    The construction of this instance is deterministic"""
    f = DimacsReader("DIMACS/{0}.col".format(name))
    M = f.M
    n = f.n
    
    #Cost param 
    quadcostlevel = 0.01
    linear_cost1 = 0.1
    linear_cost2 = 0.1
    
    # Input data
    Q1 = quadcostlevel*(4*np.eye(n,k=0) - np.eye(n,k=1)-np.eye(n,k=-1))
    Q2 = quadcostlevel*(2*np.eye(n,k=0) -np.eye(n,k=1)-np.eye(n,k=-1))
    
    q1 = linear_cost1*np.ones(n)
    q2 = linear_cost2*np.ones(n)
    create_files(name+"_det1",n,Q1,Q2,q1,q2,M)
    
def create_files_random(name,nb):
    """Create the files for an instance linked to a DIMACS graph called "name"
    The construction of this instance is random"""
    f = DimacsReader("DIMACS/{0}.col".format(name))
    M = f.M
    n = f.n
    
    #Cost param 
    quadcostlevel = 0.3
    linear_cost1 = 0.2
    linear_cost2 = 0.2
    
    np.random.seed(nb)
    # Input data
    Q1 = quadcostlevel*np.random.rand(n,n)
    Q2 = quadcostlevel*np.random.rand(n,n)
    
    q1 = linear_cost1*np.random.rand(n)
    q2 = linear_cost2*np.random.rand(n)
    create_files(name+"_random"+str(nb),n,Q1,Q2,q1,q2,M)