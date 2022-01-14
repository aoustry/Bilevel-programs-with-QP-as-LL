# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:07:40 2021

@author: aoust
"""
import matplotlib.pyplot as plt
import os
import numpy as np
from csv import DictReader

    
def plot_convergence_app(app_idx,instance_liste,target):
    m = 0
    for name in instance_liste:
        if app_idx == 1:
            file = open("../output/Application1/"+name+"/cutting_plane.csv", 'r')
        else:
            file = open("../output/Application2/"+name+"/cutting_plane.csv", 'r')
        csv_dict_reader = DictReader(file)
        array = []
        for row in csv_dict_reader:
            array.append(float(row['Epsilon']))
        m = max(m, len(array))
        plt.plot(np.abs(np.array(array[app_idx-1:])), color ='grey') #remove grey for colored plots
    plt.plot(range(m), np.ones(m)*target,color ='black',linestyle = ":")
    plt.ylim([1E-8,10])
    plt.yscale('log')
    plt.xlabel("Iterations")
    plt.ylabel(r'Feasibility error $\epsilon_k$')
    plt.savefig("app_{0}.png".format(app_idx))
    plt.close()
    


app1liste = ['notPSD_random1',
 'notPSD_random10',
 'notPSD_random2',
 'notPSD_random3',
 'notPSD_random4',
 'notPSD_random5',
 'notPSD_random6',
 'notPSD_random7',
 'notPSD_random8',
 'notPSD_random9',
 'PSD_random1',
 'PSD_random2',
 'PSD_random3',
 'PSD_random4',
 'PSD_random6',
 'PSD_random7',
 'PSD_random8',
 'PSD_random9']

app2liste = ['jean_random1',
 'jean_det1',
 'myciel4_random1',
 'myciel4_det1',
 'myciel5_random1',
 'myciel5_det1',
 'myciel6_random1',
 'myciel6_det1',
 'myciel7_random1',
 'myciel7_det1',
 'queen5_5_random1',
 'queen5_5_det1',
 'queen6_6_random1',
 'queen6_6_det1',
 'queen7_7_random1',
 'queen7_7_det1',
 'queen8_8_random1',
 'queen8_8_det1',
 'queen8_12_random1',
 'queen8_12_det1',
 'queen9_9_random1',
 'queen9_9_det1']

plot_convergence_app(1,app1liste,1E-6)
plot_convergence_app(2,app2liste,1E-6)
