# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 18:02:36 2020

@author: aoust
"""
import numpy as np

class DimacsReader():
    
    
    def __init__(self,path):
        
        self.path = path
        self.file = open(path)
        
        self.read_file()
        self.file.close()
    
    def read_file(self):
        
        line = self.file.readline()
        
        while len(line)>0:
            
            if line[0]=="p":
                array = line.split(" ")
                self.n = int(array[2])
                self.e = int(array[3])
                self.M = np.eye(self.n)
                
            if line[0]=="e":
                array = line.split(" ")
                i,j = int(array[1]),int(array[2])
                assert(i!=j)
                self.M[i-1,j-1] = 1
                self.M[j-1,i-1] = 1
            
            line = self.file.readline()
        

