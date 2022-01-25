# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:07:40 2021

@author: aoust
"""
import matplotlib.pyplot as plt
import pandas as pd
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
    
def clean(string):
    if type(string)==str:
        string=string.replace("[","")
        string=string.replace("]","")
        return float(string)
    return string

def plot_convergence_case(app_idx,name_instance):

    df_cp = pd.read_csv("../output/Application{0}/{1}/cutting_plane.csv".format(app_idx, name_instance))
    df_IOA = pd.read_csv("../output/Application{0}/{1}/InnerOuterApproxAlgo.csv".format(app_idx, name_instance))
    df_mitsos = pd.read_csv("../output/Application{0}/{1}/mitsos_sip.csv".format(app_idx, name_instance))
    
    ub_cp, ub_ioa, ub_mitsos = [clean(a) for a in df_cp['UB']], [clean(a) for a in df_IOA['MasterObjRes']], [clean(a) for a in df_mitsos['UB']]
    
    ref = 0.5*(ub_cp[-1]+ub_ioa[-1])
    plt.plot(np.array(ub_cp)-ref,label = 'CP')
    plt.plot(np.array(ub_ioa)-ref,label = 'IOA')
    plt.plot(np.array(ub_mitsos)-ref,label = 'Mitsos')
    plt.yscale('log')
    plt.ylim([1E-7,1000])
    plt.legend()
    plt.savefig("ub/app_{0}_{1}.png".format(app_idx,name_instance))
    plt.close()
    
    return ub_cp, ub_ioa, ub_mitsos
    


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

# plot_convergence_app(1,app1liste,1E-6)
# plot_convergence_app(2,app2liste,1E-6)
for name in app1liste:
    plot_convergence_case(1,name)
for name in app2liste:
    plot_convergence_case(2,name)