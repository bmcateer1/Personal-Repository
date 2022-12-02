# 

#__/\\\\\\\\\\\\\____/\\\\____________/\\\\__/\\\\\\\\\\\\\\\_______________/\\\\\\\\\\______/\\\\\\\_________/\\\_        
# _\/\\\/////////\\\_\/\\\\\\________/\\\\\\_\/\\\///////////______________/\\\///////\\\___/\\\/////\\\___/\\\\\\\_       
#  _\/\\\_______\/\\\_\/\\\//\\\____/\\\//\\\_\/\\\________________________\///______/\\\___/\\\____\//\\\_\/////\\\_      
#   _\/\\\\\\\\\\\\\\__\/\\\\///\\\/\\\/_\/\\\_\/\\\\\\\\\\\_______________________/\\\//___\/\\\_____\/\\\_____\/\\\_     
#    _\/\\\/////////\\\_\/\\\__\///\\\/___\/\\\_\/\\\///////_______________________\////\\\__\/\\\_____\/\\\_____\/\\\_    
#     _\/\\\_______\/\\\_\/\\\____\///_____\/\\\_\/\\\_________________________________\//\\\_\/\\\_____\/\\\_____\/\\\_   
#      _\/\\\_______\/\\\_\/\\\_____________\/\\\_\/\\\________________________/\\\______/\\\__\//\\\____/\\\______\/\\\_  
#       _\/\\\\\\\\\\\\\/__\/\\\_____________\/\\\_\/\\\\\\\\\\\\\\\___________\///\\\\\\\\\/____\///\\\\\\\/_______\/\\\_ 
#        _\/////////////____\///______________\///__\///////////////______________\/////////________\///////_________\///_ 


import unittest
import inspect                      # For checking correct types of variables
import numpy as np
import matplotlib.pyplot as plt 


# Import the file specified at the user prompt
submission = input('Please enter the rk function file name you wish to test.\n>>')
# Strip the '.py' if it exists
if submission[-3:] == '.py':
    submission = submission[0:(len(submission)-3)]
# Import the homework submission
soln = __import__(submission)
submission2 = input('Please enter the HH function file name you wish to test.\n>>')
# Strip the '.py' if it exists
if submission2[-3:] == '.py':
    submission2 = submission[0:(len(submission)-3)]
# Import the homework submission
hh = __import__(submission2)


Cm = 1
Vna = -115
Vk = 12
Vl = -10.613
gna_bar = 120
gk_bar = 36
gl_bar = 0.3
    
start = 0                               # Start of range
stop = 20                          # Stop of range                                # Number of steps
dt = 0.01                # Step size
# Instantiate the independent variable
x_soln = [i for i in soln.irange(start, stop, dt)]

class TestHH(unittest.TestCase):   
    Vnmh = soln.rk([hh.vmp_eq_26,hh.np_eq_7, hh.mp_eq_15,hh.hp_eq_16], x_soln, step_size = dt, initial_values = [0, 0, 0, 0])
    
    t = x_soln
    vm = np.asarray(Vnmh[0])
    n = np.asarray(Vnmh[1])
    m = np.asarray(Vnmh[2])
    h = np.asarray(Vnmh[3])
    i_na = gna_bar*m**3*h*(vm-Vna)
    i_k = gk_bar*n**4*(vm-Vk)
    g_na = gna_bar*m**3*h
    g_k = gk_bar*n**4
    fig = plt.figure(figsize = [8, 10])
    plt.subplot(3,1,1)
    plt.plot(t, vm)
    plt.xticks([])
    plt.ylabel('Vm (mV)')
    plt.subplot(3,1,2)
    plt.plot(t, i_na)
    plt.plot(t, i_k)
    plt.legend(['ina','ik'])
    plt.xticks([])
    plt.ylabel('i (uA/cm^2)')
    plt.subplot(3,1,3)
    plt.plot(t, g_na)
    plt.plot(t, g_k)
    plt.legend(['g_na', 'g_k'])
    plt.xlabel('t (ms)')
    plt.ylabel('g (mS/cm^2)')
    plt.show()
    
if __name__ == '__main__':
    unittest.main()