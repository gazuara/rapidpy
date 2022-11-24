##################################################
####                                          ####
####    RAPID algorithm in python .mat inputs ####
####                                          ####
##################################################


import numpy as np
from matplotlib import pyplot as plt
import matplotlib

print('Numpy version: ' + str(np.__version__))
print('Matplotlib version: ' + str(matplotlib.__version__))

from tkinter.filedialog import askdirectory, askopenfile
import os
import struct
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

# folder_result = askdirectory(initialdir='.')
# Lf = len(folder_result)

def eu2dist(p1,p2):
    dist = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    return dist

transducers  = np.array([[130.0,	40.0],
                [220.0,	150.5],
                [220.0,	261.0],
                [220.0,	371.5],
                [130.0,	482.0],
                [40.0,  371.5],
                [40.0, 	261.0],
                [40.0, 	150.5]], dtype=float)/1000 # in meters

N = len(transducers)
Fs = 40e6

xmin = 0; # (m), ver plano (origen de coordenadas situado en PZT#1)
xmax = +0.26; # (m), ver plano
ymin = 0; # (m), ver plano
ymax = +0.522; # (m), ver plano

Nx = 100;
Ny = 100;

x_malla=np.linspace(xmin,xmax,Nx);
y_malla=np.linspace(ymin,ymax,Ny);

high = askopenfile(initialdir='../data/')
print(high.name)
low = askopenfile(initialdir='../data/')
print(low.name)

with open(high.name, mode='rb') as file:
    fileContentH = file.read()
with open(low.name, mode='rb') as file:
    fileContentL = file.read()

dataHighT = struct.unpack('>'+'f'*int((len(fileContentH))/4), fileContentH)
dataLowT = struct.unpack('>'+'f'*int((len(fileContentL))/4), fileContentL)
samples=int(len(dataHighT)/(N*N+1))
dH = np.array(dataHighT[samples:],dtype=np.float32)
dL = np.array(dataLowT[samples:],dtype=np.float32)
dataH = np.zeros([samples,N,N])
dataL = np.zeros([samples,N,N])
iterator = 0;
for i in range(N):
    for j in range(N):
        dataH[:,i,j] = dH[iterator*samples:(iterator+1)*samples]
        dataL[:,i,j] = dL[iterator*samples:(iterator+1)*samples]
        iterator=iterator+1

DI = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if (i!=j):
            sL = dataL[:,i,j]
            sH = dataH[:,i,j]
            DI[i,j] = np.corrcoef(sL,sH)[0][1]

t = np.arange(0,samples/Fs,1/Fs)
tr = 1
rc = 5
# plt.figure(1)
plt.plot(t,dataL[:,tr,rc],color='r',label='High')
plt.plot(t,dataH[:,tr,rc],color='b',label='Low')
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.title("Lamb wave signals")
plt.legend()
# plt.show(block=False)
plt.show()

x_grid=np.linspace(xmin,xmax,Nx);
y_grid=np.linspace(ymin,ymax,Ny);
beta = 1.015
result = np.zeros((Nx,Ny))
activation = [1,1,1,1,1,1,1,1]

print('Processing RAPID...')
for i in range(Nx):
    for j in range(Ny):
        pxy=0;
        for tr in range (N):
            for rc in range(N):
                if (activation[tr]==1)&(activation[rc]==1)&(tr!=rc):
                
                    p = [x_grid[i],y_grid[j]]
                    
                    ptp = [transducers[tr,0],transducers[tr,1]]
                    prp = [transducers[rc,0],transducers[rc,1]]
                    
                    dtp=eu2dist(p,ptp);                         
                    dpr=eu2dist(p,prp);
                    dtr=eu2dist(ptp,prp);
                    
                    Etr = (dtp+dpr)/dtr;
                    
                    if Etr>=beta:
                        Rtr=beta;
                    else:
                        Rtr=Etr;
                           
                    pxy = pxy+DI[tr,rc]*((beta-Rtr)/(beta-1));    #Actualización del valor de Pxy
    
        result[i,j] = pxy
print('Plotting...')

plt.figure(2)
plt.imshow(result.transpose())
plt.show()
