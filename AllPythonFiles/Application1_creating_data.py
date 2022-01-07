import numpy as np
import time
import os

def create_files(name,n,p,Qref,qref,cref,wlist, noise,z):
    #Write the .dat file
    os.mkdir("Application1_data/"+name)
    f = open("Application1_data/"+name+"/instance.dat","w")
    f.write("param n := %d;\n"%n)
    f.write("param p_max := %d;\n"%p)
    f.write("param r_dim := 0;\n")
    f.write("\nparam Q_ref :")
    for i in range(n):
        f.write(" %d "%(i+1))
    f.write(":=\n")
    for i in range(n):
        for j in range(n):
            if j==0:
                f.write("   %d "%(i+1))
            f.write("%f "%Qref[i,j])
            if j==(n-1):
                if i==(n-1):
                    f.write(";")
                f.write("\n")
    
    f.write("\nparam q_ref := ")
    for i in range(n):
        f.write("%d "%(i+1))
        f.write("%f "%qref[i])
    f.write(";\n\nparam c_ref := %f ;\n\n"%cref)
    f.write("\nparam w:")
    for i in range(n):
        f.write(" %d "%(i+1))
    f.write(":=\n")
    for i in range(p):
        for j in range(n):
            if j==0:
                f.write("   %d "%(i+1))
            f.write("%f "%wlist[i,j])
            if j==(n-1):
                if i==(p-1):
                    f.write(";")
                f.write("\n")
    f.write("\nparam epsilon := ")
    for i in range(p):
        f.write("%d "%(i+1))
        f.write("%f "%noise[i])
    f.write(";\n")
    f.write("\nparam z := ")
    for i in range(p):
        f.write("%d "%(i+1))
        f.write("%f "%z[i])
    f.write(";")
    f.close()
    
    #Write the numpy files 
    np.save("Application1_data/"+name+"/bigQref",Qref)
    np.save("Application1_data/"+name+"/qref",qref)
    np.save("Application1_data/"+name+"/cref",cref)
    np.save("Application1_data/"+name+"/w",wlist)
    np.save("Application1_data/"+name+"/noise",noise)
    np.save("Application1_data/"+name+"/z",z)
 
def create_files_PSD_instances(n,nb):
    name = "PSD_random"+str(nb)
    seed = nb
    np.random.seed(seed)
    asym = 2*np.random.rand(n,n) - 1
    Qref = asym.dot(asym.transpose())
    qref = 2*(2*np.random.rand(n)-1)
    sigma = 0.3
    p = 4000
    wlist = np.random.rand(p,n)
    noise = np.random.normal(loc = 0, scale = sigma, size = p)
    aux=0.5 *np.array([ w.dot(Qref).dot(w) for w in wlist]) + wlist.dot(qref) 
    cref = -aux.min()
    noiseless_z = aux +cref
    print(noiseless_z.min())
    z = noiseless_z + noise
    create_files(name,n,p,Qref,qref,cref,wlist, noise,z)
    
    
def create_files_nonPSD_instances(n,nb):
    name = "nonPSD_random"+str(nb)
    seed = 10+nb
    asym = 2*np.random.rand(n,n) - 1
    Qref = asym + asym.transpose()
    qref = 2*(2*np.random.rand(n)-1)
    sigma = 0.3
    p = 4000
    wlist = np.random.rand(p,n)
    noise = np.random.normal(loc = 0, scale = sigma, size = p)
    aux=0.5 *np.array([ w.dot(Qref).dot(w) for w in wlist]) + wlist.dot(qref)
    cref = -aux.min()
    noiseless_z = aux +cref
    print(noiseless_z.min())
    z = noiseless_z + noise
    create_files(name, n, p, Qref, qref,cref, wlist, noise, z)
    
moment=time.strftime("%Y-%b-%d__%H_%M_%S",time.localtime()) #to have different .dat files for each python run
create_files_PSD_instances(5, 1)
create_files_PSD_instances(5, 2)
create_files_PSD_instances(10, 3)
create_files_PSD_instances(10, 4)
create_files_PSD_instances(15, 5)
create_files_PSD_instances(15, 6)

create_files_nonPSD_instances(5,1)
create_files_nonPSD_instances(5, 2)
create_files_nonPSD_instances(10, 3)
create_files_nonPSD_instances(10, 4)
create_files_nonPSD_instances(15, 5)
create_files_nonPSD_instances(15, 6)
