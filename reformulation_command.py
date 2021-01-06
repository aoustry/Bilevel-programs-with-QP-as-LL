# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:24:39 2021

@author: aoust
"""
import os

#Call application 1

for i in range(1,7):
    name = "PSD_random"+str(i)
    os.system("python3 Application1_reformulation.py "+name)
    name = "nonPSD_random"+str(i)
    os.system("python3 Application1_reformulation.py "+name)

#Call application 2
list_graphs = ["myciel4","myciel7","jean"]
for graph in list_graphs:
    nameDimacs = graph+".col"
    name = graph+"_det1"
    os.system("python3 Application1_reformulation.py "+nameDimacs+" "+name)
    name = graph+"_random1"
    os.system("python3 Application1_reformulation.py "+nameDimacs+" "+name)