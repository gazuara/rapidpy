##################################################
####                                          ####
####    RAPID algorithm in python .mat inputs ####
####                                          ####
##################################################


import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import scipy.io as sio
from scipy import signal
import scipy
print('Numpy version: ' + str(np.__version__))
print('Matplotlib version: ' + str(matplotlib.__version__))
print('Numpy version: ' + str(scipy.__version__))

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

plate = input('Input name plate (plate tp fidamc - plate tp isati): ')

if plate == 'plate tp isati':
    transducers = np.array([[130.0,	40.0],
                    [220.0,	150.5],
                    [220.0,	261.0],
                    [220.0,	371.5],
                    [130.0,	482.0],
                    [40.0,  371.5],
                    [40.0, 	261.0],
                    [40.0, 	150.5]], dtype=float)/1000 # in meters

elif plate == 'plate tp fidamc':
    transducers = np.array([[0.00,  0.00],
                    [73.33,     0.00],
                    [146.66,    0.00],
                    [220.00,    0.00],
                    [220.00,    73.33],
                    [220.00,    146.66],
                    [220.00,    220.00],
                    [146.66,    220.00],
                    [73.333,    220.00],
                    [0.00,      220.00],
                    [0.00,      146.66],
                    [0.0,       73.33]],dtype=float)/1000 # in meters

N = len(transducers)
Fs = float(input('Input Sampling Frequency: '))

xmin = float(input('X min: ')); # (m), ver plano (origen de coordenadas situado en PZT#1)
xmax = float(input('X max: ')); # (m), ver plano
ymin = float(input('Y min: ')); # (m), ver plano
ymax = float(input('Y max: ')); # (m), ver plano

Nx = 100;
Ny = 100;

x_malla=np.linspace(xmin,xmax,Nx);
y_malla=np.linspace(ymin,ymax,Ny);

high = askopenfile(initialdir='../data/')
print(high.name)
low = askopenfile(initialdir='../data/')
print(low.name)

typefile = input('Input "mat" or "bin" to select type of file to open (matlab or binary-labview): ')

if typefile == 'bin':

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

elif typefile == 'mat':
    
    dataset = sio.loadmat(high.name)
    dataH = dataset['data']
    dataset = sio.loadmat(low.name)
    dataL = dataset['data']
    samples = int(len(dataL))

t0 = 0e-6
anaw = np.arange(round(t0*Fs),samples)

norder = 4;
flow = 200e3
fhigh = 1500e3

b, a = signal.butter(norder, [flow,fhigh], btype='band', analog=False, output='ba', fs=Fs)

def butter_bandpass(lowcut, highcut, fs, order):
    return signal.butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=norder):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = signal.lfilter(b, a, data)
    return y

DI1 = np.zeros((N,N))
DI2 = np.zeros((N,N))
DI3 = np.zeros((N,N))

filtering = True
if filtering == True:
    dataLf = np.zeros(dataL.shape)
    dataHf = np.zeros(dataH.shape)

for i in range(N):
    for j in range(N):
        if (i!=j):
            
            if filtering == False:
                sL = dataL[:,i,j]
                sH = dataH[:,i,j]
            else:
                dataLf[:,i,j] = butter_bandpass_filter(dataL[:,i,j],flow,fhigh,Fs,norder)
                dataHf[:,i,j] = butter_bandpass_filter(dataH[:,i,j],flow,fhigh,Fs,norder)
                sL = dataLf[:,i,j]
                sH = dataHf[:,i,j]

            DI1[i,j] = 1-np.corrcoef(sL[anaw],sH[anaw])[0][1]
            DI2[i,j] = sum(np.power((sL[anaw]-sH[anaw]),2))

DI = DI2

t = np.arange(0,samples/Fs,1/Fs)
tr = 1
rc = 7
# plt.figure(1)
plt.plot(t[anaw],dataL[anaw,tr,rc],color='r',label='High')
plt.plot(t[anaw],dataH[anaw,tr,rc],color='b',label='Low')
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.title("Lamb wave signals")
plt.legend()
# plt.show(block=False)
plt.show()

if filtering == True:
    plt.plot(t[anaw],dataLf[anaw,tr,rc],color='r',label='High')
    plt.plot(t[anaw],dataHf[anaw,tr,rc],color='b',label='Low')
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.title("FILTERED Lamb wave signals")
    plt.legend()
    plt.show()

x_grid=np.linspace(xmin,xmax,Nx);
y_grid=np.linspace(ymin,ymax,Ny);
beta = 1.025
result = np.zeros((Nx,Ny))
activation = [1,1,1,1,1,1,1,1,1,1,1,1]
#reference = [1,2,3,4,5,6,7,8,9,0,1,2]

print('Processing RAPID...')
for i in range(Nx):
    for j in range(Ny):
        pxy = 0;
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
                        Rtr=beta
                    else:
                        Rtr=Etr
                           
                    pxy = pxy + DI[tr,rc]*((beta-Rtr)/(beta-1));    #Actualización del valor de Pxy
    
        result[i,j] = pxy
        
print('Plotting...')

result_oriented =np.flip(result.transpose(),axis=0)
plt.figure(2)
plt.imshow(result_oriented)
plt.show()
