# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:39:29 2021

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from numpy.fft import fft, ifft

from IPython.display import Audio

##1.1
mysterysig = loadmat('mysterysignal')

signal = (mysterysig['x'])
signal = signal[0][:] #isolates the signal from the matlab file

time = len(signal[:]) #time points from 0-2seconds multiplied by the sample freq

samplefreq = 22050 #sample frequency in Hz

t = np.arange(0, 2, 1/samplefreq)  #time points from 0-2seconds divided by the sample freq


fig,ax = plt.subplots(figsize=(8,6))
ax.plot(t,signal, color = 'tab:blue',linewidth=2,label='Voltage (V)')
ax.set_xlabel('Time(s)',fontweight='bold',fontsize=14)
ax.set_ylabel('Voltage',fontweight='bold',fontsize=14)


Xf = np.fft.fft(signal) # np.fft.fft(signal) is what he did in lecture
f = np.fft.fftfreq(Xf.shape[0]) * samplefreq #

fig,ax = plt.subplots(figsize=(8,6))
ax.plot(f,abs(Xf), label='Fourier of X')
ax.set(xlim=(0,1000), ylim=(0,6000))
ax.set_xlabel('Frequency',fontweight='bold',fontsize=14)
ax.set_ylabel('Voltage?',fontweight='bold',fontsize=14)

DPeak = [] # Freq where peaks occur
PeakAmp = [] # amps of peaks
count = 0

for i in np.arange(0,len(Xf)):
    if (Xf[i] > 1000 and f[i] < 1000):
        DPeak.append(f[i])
        PeakAmp.append(Xf[i])
    
#print('Peak max frequencies are',DPeak,'Hz')

NormalPeakAmp = (PeakAmp / PeakAmp[4].real).real

#print('Peak amplitudes are', NormalPeakAmp)

#1.1.d
fig,ax = plt.subplots(figsize=(8,6))
ax.plot(f,np.angle(Xf), color = 'tab:blue',linewidth=2,label='Voltage (V)')
ax.set_xlabel('Time(s)',fontweight='bold',fontsize=14)
ax.set_ylabel('Voltage squared',fontweight='bold',fontsize=14)


threshold = PeakAmp[4]
phi = np.angle(Xf)
phi[np.abs(Xf) < threshold] = 0

fig,ax = plt.subplots(figsize=(8,6))
ax.scatter(f,phi, color = 'tab:blue',facecolors='none', edgecolors = 'b')
ax.set(xlim=(0,1000), ylim=(-1,1))
ax.set_xlabel('Freq)',fontweight='bold',fontsize=14)
ax.set_ylabel('Voltage squared',fontweight='bold',fontsize=14)

phizero = []

for i in np.arange(0,len(Xf)):
    if (phi[i] <0):
        phizero.append(phi[i])
        
#print(phizero)


#2
#t = np.arange(0,2, 1/samplefreq)
#need first 10% of 

x200 = signal[:4410]

Xf2 = np.fft.fft(x200, len(Xf))
fig,ax = plt.subplots(figsize=(8,6))
ax.plot(f,Xf2, color = 'tab:blue',linewidth=2,label='Voltage (V)')
ax.set(xlim=(0,900), ylim=(-10,1000))
ax.set_xlabel('Freq',fontweight='bold',fontsize=14)
ax.set_ylabel('Voltage squared',fontweight='bold',fontsize=14)


#2e explaiend in 9/2 lecture (multiply by rect in time in time is convol in freq. multipling xf y a sinc funtion causing ripples)

#3
stalbans = loadmat('stalbans') #audio data

h = (stalbans['h']) # impulse response
fs = (stalbans['fs']) #sampling rate
x = (stalbans['x']) # random sentence

fs = fs[0] #formatting the sampling rate into a vector
fs = fs[0]


Audio(data=x, rate =fs) #plays audio

zerosneeded = round((len(h[0])-len(x[0])) / 2)

NewX = np.pad(x[0][:],zerosneeded, 'constant')
NewX = np.append(0,NewX)


hF = np.fft.fft(h) 
xFF = np.fft.fft(NewX)

hF = hF[0][:]
#xFF = xFF[0][:]
#convulving is a way to combine the two signals over time, so now in the frequency domain they just multiply
ConvX = hF * xFF #multiplies in freq domain


Xconvolved = abs(np.fft.ifft(ConvX)) #slight delay compared to the np.conv funciton
fx = np.fft.fftfreq(Xconvolved.shape[0]) * fs

h = h[0]
x = x[0]
Convolutionx = np.convolve(h, x)

fig,ax = plt.subplots(figsize=(8,6))
ax.plot(fx,ConvX, color = 'tab:blue',linewidth=2,label='Voltage (V)')
#ax.plot(fx,Convolutionx, color = 'tab:red',linewidth=2,label=' Voltage (V)')
ax.set(xlim=(0,900), ylim=(-10,1000))
ax.set_xlabel('Freq',fontweight='bold',fontsize=14)
ax.set_ylabel('Voltage squared',fontweight='bold',fontsize=14)

#4
#All Answers Confirmed via graphing on Desmos
#A = Q Because A views as an aboslute value of a sine wave with other functions added
#B = P As it is a couple frequencies close to each other. closer than E or S
#C = T Because One period of a sine wave must have had destructive interference with other frequency signals. It appears to be a sine wave multiplied by a small box, making it a sinc function in the frequencey domain
#D = R Because although it is similar to B/P, the more waves make it closer to dirac delta function. it appears to be a sine wave multiplied by a large box, making it more like a delta function in the frequency domain. 
#E = S Because it is a few frequencies similarly close together but more spread out than B since the waving of the change in amplitude is slower and diffferent than the sine frequency


#5
variance = 100 #mm noise
stdev = 10
TruDepth = 47 #mm true depth of target

#5a
NumMeasures = 4
Measurement = np.arange(0,NumMeasures)

for i in range(0,NumMeasures):
    Measurement[i] = np.random.normal(TruDepth,stdev,1) #Generates 4 numbers with normal dist and given stdev and average
    
#5b
depth = np.mean(Measurement) #average measurement with given noise

#5c
realdepth = np.arange(0,100)
for i in range(0,100):
    realdepth[i] = np.mean(np.random.normal(TruDepth,stdev,4))

measuredvar = np.var(realdepth) #variance of the 100 days of 4 measurements
print(measuredvar)

#6
import math
#5chose5 * 0.5^5
#6a
Probabilitysuccess = np.power(.5,5) # 3.125% to get 5 heads therefore not a fair coin
#if less than 5% it is not considered fair
#null hypthesis is that the coin is fair, reject the null hypothesis and say that the coin is in fact unfair
#6b
#20trials(5flips), all with 3.125% of 5 heads
#6b
successes_k = 1
unsuccess_k = 0
Trials = 20
Repeatedsuccesschance = 1 - (math.comb(Trials,unsuccess_k)*np.power(Probabilitysuccess,0) * np.power(1-Probabilitysuccess,20)) 
# 1 minus because we need the opposite percentage of 20 trials with all giving no 5 heads in a row.  

print(Repeatedsuccesschance*100,'% that one person will get all heads out of 20') #probability of one person getting all 5 heads

