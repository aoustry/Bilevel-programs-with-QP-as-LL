# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:07:40 2021

@author: aoust
"""
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_convergence_single(name):
    
    file = open("../epsilon/"+name+".txt")
    array = []
    for line in file.readlines():
        array.append(float(line))
        
    plt.plot(np.abs(np.array(array)),color='grey')
    plt.yscale('log')
    plt.title(name +", convergence of the feasibility error.")
    plt.xlabel("Iterations")
    plt.ylabel(r'Feasibility error $\epsilon_k$')
    plt.savefig(name+".png")
    plt.close()
    
def plot_convergence_app(app_idx,instance_liste,target):
    m = 0
    for name in instance_liste:
        file = open("../epsilon/"+name+".txt")
        array = []
        for line in file.readlines():
            array.append(float(line))
        m = max(m, len(array))
        plt.plot(np.abs(np.array(array[app_idx-1:])), color ='grey')
    plt.plot(range(m), np.ones(m)*target,color ='black',linestyle = ":")
    #plt.xscale('log')
    plt.ylim([1E-8,10])
    plt.yscale('log')
    #plt.title("Application {0}, convergence of the feasibility error.".format(app_idx))
    plt.xlabel("Iterations")
    plt.ylabel(r'Feasibility error $\epsilon_k$')
    plt.savefig("plots/app_{0}.png".format(app_idx))
    plt.close()
    
# for name in os.listdir('epsilon'):
#     plot_convergence_single(name[:len(name)-4])

app1liste = ['nonpsd1',
 'nonpsd10',
 'nonpsd2',
 'nonpsd3',
 'nonpsd4',
 'nonpsd5',
 'nonpsd6',
 'nonpsd7',
 'nonpsd8',
 'nonpsd9',
 'psd1',
 'psd2',
 'psd3',
 'psd4',
 'psd6',
 'psd7',
 'psd8',
 'psd9']

app2liste = ['jeannotpsd',
 'jeanpsd',
 'myciel4notpsd',
 'myciel4psd',
 'myciel5notpsd',
 'myciel5psd',
 'myciel6notpsd',
 'myciel6psd',
 'myciel7notpsd',
 'myciel7psd',
 'queen5notpsd',
 'queen5psd',
 'queen6notpsd',
 'queen6psd',
 'queen7notpsd',
 'queen7psd',
 'queen8notpsd',
 'queen8psd',
 'queen8_12notpsd',
 'queen8_12psd',
 'queen9notpsd',
 'queen9psd']

plot_convergence_app(1,app1liste,1E-6)
plot_convergence_app(2,app2liste,1E-6)
