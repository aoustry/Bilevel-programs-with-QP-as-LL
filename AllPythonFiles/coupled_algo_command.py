# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:24:43 2021

@author: aoust
"""


from Application1_coupled_algo import main_app1
from Application2_coupled_algo import main_app2

#Call application 1

for i in range(1,10):
    name = "PSD_random"+str(i)
    main_app1(name,0.1)
    # name = "notPSD_random"+str(i)
    # main_app1(name,0.1)

#Call application 2
list_graphs = ["jean","myciel4","myciel5","myciel6","myciel7","queen5_5","queen6_6","queen7_7","queen8_8","queen8_12","queen9_9"]

for graph in list_graphs:
    nameDimacs = graph+".col"
    name = graph+"_det1"
    main_app2(nameDimacs,name,0.1)
    name = graph+"_random1"
    main_app2(nameDimacs,name,0.1)
