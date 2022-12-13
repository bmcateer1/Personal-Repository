# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 21:37:46 2021

@author: Ben
"""
import numpy as np
import matplotlib.pyplot as plt

#2.1
t = np.arange(0, 1, 0.001) 
freq = 10 #Hz 
x = np.sin(freq * 2* np.pi * t)
y = np.sin(freq * 2* np.pi * t + np.pi/2)




fig1 = plt.figure()
plt.plot(t,x, t, y)
plt.title('voltage and current')
plt.xlabel('Time (S)')
plt.ylabel('Voltage or Current')
plt.legend(['Voltage (V)', 'Current(A)'], loc = 'upper right')

"""
ANSWER KEY FOR PLOTTING 
fig,ax = plt.subplots(figsize=(8,6))
ax.plot(t,x, color = 'tab:blue',linewidth=2,label='Voltage (V)')
ax.plot(t,y,color='red',linewidth=2, label = 'Current (A)')
ax.set(xlim=(0,1), ylim=(-1.1,1.1))
ax.set_xlabel('Time(s)',fontweight='bold',fontsize=14)
ax.set_ylabel('Voltage or Current signal',fontweight='bold',fontsize=14)
plt.xticks(ticks = [0, 0.25, 0.5, 0.75, 1], fontsize=12,fontweight='bold')
plt.yticks(ticks = [-1, -0.5, 0, 0.5, 1], fontsize=12, fontweight='bold')
ax.legend(loc='upper left',fontsize=12,prop={'weight':'bold'}, bbox_to_anchor=(1.05,1.0))
fig.show()
"""


##2.2

Power = pow(x,2)
E = np.cumsum(Power)


fig2, (plt1,plt2) = plt.subplots(2)
plt1.plot(t,Power)
plt1.set_ylabel("Instantaeous Power")
plt2.plot(t,E)
plt2.set_ylabel("running integral")


"""
ANSWER KEY FOR PLOTTING
fig,ax = plt.subplots(2,1,figsize=(8,10))
ax[0].plot(t,P)
ax[0].set(xlim=[0,1])
ax[0].set_ylabel('Instantaneous Power ($Volts^2$)', fontsize=14)
ax[0].tick_params(axis='both', which='major', labelsize=12)

ax[1].plot(t,E)
ax[1].set(xlim=[0,1])
ax[1].set_ylabel('Running Integral ($Volts^2 x seconds$)', fontsize=14)
ax[1].set_xlabel('Time (s)',fontsize=14)
ax[1].tick_params(axis='both', which='major', labelsize=12)
fig.show()
"""


##2.3
from scipy.io import loadmat
moneydata = loadmat('marketdata.mat')
euro = moneydata['eurusd']
gold = moneydata['xauusd']

days = list(range (1,len(gold)))

eurodif = np.diff(euro, axis = 0)
golddif = np.diff(gold, axis = 0)

fig3, (plt1,plt2) = plt.subplots(2)
plt1.plot(days,golddif)
plt1.set_ylabel('gold to USD price per day')
plt2.plot(days,eurodif)
plt2.set_ylabel('Euro to USD price per day')


EUR2012 = np.diff(euro[-200:], axis = 0)
XAU2012 = np.diff(gold[-200:], axis = 0)
two_hunda_days = list(range (1,len(EUR2012)+1))

EURyearsago = np.diff(euro[:-200], axis = 0)
XAUyearsago = np.diff(gold[:-200], axis = 0)

fig4, (plt1,plt2) = plt.subplots(2)
plt1.scatter(EUR2012, XAU2012, s = 10, facecolors='none', edgecolors = 'b')
plt1.set_ylabel('Euro to USD price per day (last 200')
plt1.set_xlim([-0.02, 0.02])
plt1.set_ylim([-25,25])
plt2.scatter(EURyearsago,XAUyearsago,s = 10, facecolors='none', edgecolors = 'b')
plt2.set_ylabel('prevous euro and gold prices')
plt2.set_xlim([-0.02, 0.02])
plt2.set_ylim([-25,25])


"""
ANSWER KEY FOR 2.3
import scipy.io as sio
from google.colab import drive
drive.mount('/content/gdrive/') #gain acess to my google drive where I put marketdata.mat
pathToFile = '/content/gdrive/MyDrive/Purdue/BME511/ProblemSets/PS0/marketdata.mat' 

#Load Data
marketdata = sio.loadmat(pathToFile)
eurusd = marketdata['eurusd']
xauusd = marketdata['xauusd']

#Compute daily change in price
eur_change = np.diff(eurusd,axis=0)
gld_change = np.diff(xauusd,axis=0)

#Plot daily change in price
fig,ax = plt.subplots(2,1,figsize=(8,10))
ax[0].plot(eur_change)
ax[0].set_ylabel('Daily price CHANGE of EUR/USD',fontsize=14)
ax[0].set_yticks([-.02,0,.02,.04])
ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[0].set_xlim([0,eur_change.size])


ax[1].plot(gld_change)
ax[1].set_ylabel('Daily price CHANGE of GOLD/USD',fontsize=14)
ax[1].set_xlabel('Time (Trading Days)',fontsize=14)
ax[1].set_yticks([-50,0,50])
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[1].set_xlim([0,gld_change.size])
fig.show()

Questions:
1) On how many days has the price of GOLD gone up and how many days has it 
gone down compared to the previous day?
2) On how many days has the price of GOLD gone up by more than 15 USD?
'''
#1
gold_up = np.sum(gld_change>0)
gold_down = np.sum(gld_change < 0)
print(' \n 1) Gold has gone up for ' + str(gold_up) +' days and down for ' 
      + str(gold_down) + ' days compared to the previous day \n')

#2
gold_up15 = np.sum(gld_change>15)
print(' \n 2) Gold has gone up by more than 15 USD in a day on ' + 
      str(gold_up15) +' days \n')

fig,ax = plt.subplots(2,1,figsize=(8,14))

ax[0].scatter(np.diff(EUR2012,axis=0), np.diff(XAU2012,axis=0),facecolors='none',edgecolors='b')
ax[0].set(xlim=[-.02,.02],ylim=[-25,25])
ax[0].set_xticks([-.02,-.01, 0, .01,0.02])
ax[0].set_ylabel('Last year\'s daily changes in GOLD prices',fontsize=14)
ax[0].tick_params(axis='both', which='major', labelsize=14)

ax[1].scatter(np.diff(EURyearsago,axis=0),np.diff(XAUyearsago,axis=0),facecolors='none',edgecolor='b')
ax[1].set(xlim=[-.02,.02],ylim=[-25,25])
ax[1].set_xticks([-.02,-.01, 0, .01,0.02])
ax[1].set_ylabel('Previous years\' daily changes in GOLD prices',fontsize=14)
ax[1].tick_params(axis='both', which='major', labelsize=14)

fig.show()
"""

#2.4 Answer Key
#Generate mixtureOfSines
frequencies = np.arange(10,110,10)
mixtureOfSines = np.zeros([frequencies.size,t.size])
for f in range(frequencies.size):
  mixtureOfSines[f,:] = np.sin(2*np.pi*frequencies[f]*t)

#Plot data
fig,ax = plt.subplots(figsize=(8,8))
ax.plot(t,np.sum(mixtureOfSines,axis=0))
ax.set_ylabel('Mixture of Sine Waves',fontsize=14)
ax.set_xlabel('Time (s)',fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
# MixtureOfSines = np.zeros(10)
# count = 1
# for i in range(0,len(t)):
    
#     MixtureOfSines(count) = np.array([10,20,30,40,50,60,70,80,90,100])
#     Sinwaves = np.sin(MixtureOfSines * np.pi *2* t)
 

