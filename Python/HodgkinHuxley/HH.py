#! /usr/local/bin/python3.4


# firstLast_HH.py
# Create functions that will solve a second order ODE using various numerical methods
# Use only standard Python, no NumPy

#__/\\\\\\\\\\\\\____/\\\\____________/\\\\__/\\\\\\\\\\\\\\\_______________/\\\\\\\\\\______/\\\\\\\_________/\\\_        
# _\/\\\/////////\\\_\/\\\\\\________/\\\\\\_\/\\\///////////______________/\\\///////\\\___/\\\/////\\\___/\\\\\\\_       
#  _\/\\\_______\/\\\_\/\\\//\\\____/\\\//\\\_\/\\\________________________\///______/\\\___/\\\____\//\\\_\/////\\\_      
#   _\/\\\\\\\\\\\\\\__\/\\\\///\\\/\\\/_\/\\\_\/\\\\\\\\\\\_______________________/\\\//___\/\\\_____\/\\\_____\/\\\_     
#    _\/\\\/////////\\\_\/\\\__\///\\\/___\/\\\_\/\\\///////_______________________\////\\\__\/\\\_____\/\\\_____\/\\\_    
#     _\/\\\_______\/\\\_\/\\\____\///_____\/\\\_\/\\\_________________________________\//\\\_\/\\\_____\/\\\_____\/\\\_   
#      _\/\\\_______\/\\\_\/\\\_____________\/\\\_\/\\\________________________/\\\______/\\\__\//\\\____/\\\______\/\\\_  
#       _\/\\\\\\\\\\\\\/__\/\\\_____________\/\\\_\/\\\\\\\\\\\\\\\___________\///\\\\\\\\\/____\///\\\\\\\/_______\/\\\_ 
#        _\/////////////____\///______________\///__\///////////////______________\/////////________\///////_________\///_ 
'worked with Chris Kannmacher and Safa Chowdhury'

import numpy as np

def vmp_eq_26(step, Vnmh):
    """
    f(step, Vnmh) = dVm/dt

    Parameters
    ----------
    step : scalar
        current time point
    Vnmh : list [vm, n, m, h]
        list of variables in the f function
    Returns
    -------
    f = dvm/dt

    """
    Cm = 1
    Vna = -115
    Vk = 12
    Vl = -10.613
    gna_bar = 120
    gk_bar = 36
    gl_bar = 0.3
    
    """
    enter your code here
    """
    f = (gk_bar * Vnmh[1]**4 * (Vk - Vnmh[0]) + gna_bar * Vnmh[2]**3 * Vnmh[3] * (Vna - Vnmh[0]) + gl_bar * (Vl - Vnmh[0])) / Cm #Calculated dVm/dt given I = 0
   
    return f

def np_eq_7(step, Vnmh):
    """

    f(step, Vnmh) = dn/dt

    Parameters
    ----------
    step : scalar
        current time point
    Vnmh : list
        [vm, n, m, h]

    Returns
    -------
    f = dn/dt

    
    enter your code here
    """

    alpha_n = 0.01* (10 + Vnmh[0]) / (np.exp((Vnmh[0]+10)/10)-1) #Calculates Alpha depending on instantaneous value of the membrane potential
    beta_n = 0.125 * np.exp(Vnmh[0] / 80) #Calculates Beta depending on instantaneous value of the membrane potential
    
    f = alpha_n * (1-Vnmh[1]) - beta_n * Vnmh[1] #Calculates dn/dt
    return f

def mp_eq_15(step, Vnmh):
    """
    f(step, Vnmh) = dm/dt

    Parameters
    ----------
    step : scalar
        current time point
    values : list
        [vm, n, m, h]

    Returns
    -------
    f = dm/dt

    enter your code here
    """
    alpha_m = 0.1 * (Vnmh[0] + 25) / (np.exp((Vnmh[0]+25)/10)-1) #calculates alpha of m
    beta_m = 4 * np.exp(Vnmh[0] / 18)#calculates beta of m
    
    f = alpha_m * (1 - Vnmh[2]) - beta_m * Vnmh[2] #calculates dm/dt 
    
    return f
    
def hp_eq_16(step, Vnmh):
    """
    f(step, Vnmh) = dh/dt

    Parameters
    ----------
    step : scalar
        current time point
    values : list
        [vm, n, m, h]

    Returns
    -------
    f = dh/dt
    
    enter your code here
    """
    
    alpha_h = 0.07 * np.exp(Vnmh[0] / 20) #calculates alpha of h
    beta_h = 1 / (np.exp((Vnmh[0]+30)/10) +1) #calculates beta of h
    
    f = alpha_h * (1 - Vnmh[3]) - beta_h * Vnmh[3] #calculates dh/dt

    return f
