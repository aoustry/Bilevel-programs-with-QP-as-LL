# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 10:38:36 2020

This file is needed to generate the instances used to test Application 2.
"""
import numpy as np
from DimacsReader import *
import time
import os


def create_files(name,n,Q1,Q2,q1,q2,M,diagonalQ2x):
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
    
    f.write("\nparam diagQ2 := ")
    for i in range(n):
        f.write("%d "%(i+1))
        f.write("%f "%diagonalQ2x[i])
    f.write(";")
    
    f.close()
    
    #Write the numpy files 
    np.save("Application2_data/"+name+"/bigQ1",Q1)
    np.save("Application2_data/"+name+"/bigQ2_fix",Q2)
    np.save("Application2_data/"+name+"/q1",q1)
    np.save("Application2_data/"+name+"/q2_fix",q2)
    np.save("Application2_data/"+name+"/M",M)
    np.save("Application2_data/"+name+"/diagQ2x",diagonalQ2x)
    
    

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
    h = (np.linalg.eigvals)(Q1)
    print(h)
    assert(h.min()>=0)
    h = (np.linalg.eigvals)(Q2)
    print(h)
    assert(h.min()>=0)
    
    q1 = linear_cost1*np.ones(n)
    q2 = linear_cost2*np.ones(n)
    diagonalQ2x = quadcostlevel*3* np.random.random(n)
    create_files(name+"_det1",n,Q1,Q2,q1,q2,M,diagonalQ2x)
    
def create_files_random(name,nb):
    """Create the files for an instance linked to a DIMACS graph called "name"
    The construction of this instance is random"""
    f = DimacsReader("DIMACS/{0}.col".format(name))
    M = f.M
    n = f.n
    
    #Cost param 
    quadcostlevel = 0.3
    linear_cost1 = 0.5
    linear_cost2 = 0.2
    
    np.random.seed(nb)
    # Input data
    asym1 = np.random.rand(n,n)
    asym2 = np.random.rand(n,n)
    Q1 = quadcostlevel*(asym1.dot(asym1.T)) #Create PSD matrix
    Q2 = quadcostlevel*(asym2+asym2.T)
    
    q1 = linear_cost1*np.random.rand(n)
    q2 = linear_cost2*np.random.rand(n)
    diagonalQ2x = quadcostlevel*3*(2* np.random.random(n)-1)
    create_files(name+"_random"+str(nb),n,Q1,Q2,q1,q2,M,diagonalQ2x)
    
    
new_instances = ["jean","myciel4","myciel5","myciel6","myciel7","queen5_5","queen6_6","queen7_7","queen8_8","queen8_12","queen9_9"]

for name in new_instances:
    create_files_deterministic_type1(name)
    create_files_random(name, 1)
