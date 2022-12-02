#! /usr/local/bin/python3.4


# 1st_order_ODE.py
# Create functions that will solve a first order ODE using various numerical methods
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


def irange(start, stop, step):
    """ Create a generator to iterate over the interval we wish to solve the ODE on
    """
    while start < stop:
        yield (start)
        start += step
        

def euler(ode, interval, step_size, initial_value):
    """ Solve for y(x), given ODE dy/dx = f(x) and boundary conditions, over the values in interval
    y_n+1 = y_n + (F(x)*h)
    """
    
    y = [initial_value] #initial value of y
    ct = 1 #iteration variable
    
    for i in interval:  #create a for loop in size of iteration
        
        if ct < (len(interval)): #keeps lenght of y the same length of for loop keeping dimensions same
            y.append(y[ct-1] + (ode(i) * step_size)) #changes y value every iteration
            ct = ct + 1 #updates iteration variable
   
    return y


def rk(ode, interval, step_size, initial_value):
    """ 
    solve for y(x), given ODE dy/dx = f(x) and boundary conditions, over the values in interval 
    k1 = f(x[n], y[n])
    k2 = f(x[n] + h/2, y[n]+k1*h/2)
    k3 = f(x[n] + h/2, y[n]+k2*h/2)
    k4 = f(x[n] + h, y[n]+k3*h)
    k_bar = (k1 + 2*k2 + 2*k3+k4) / 6
    y[n+1] = y[n] + (k_bar*h)
    """
    y = [initial_value] #initial value of y
    ct = 1 #iteration variable
    
    for i in interval: #create a for loop in size of iteration
        
        if ct < (len(interval)): #keeps length of y the same length of for loop, keeping dimensions same
            k1 = ode(i)          #calculates k1
            k2 = ode(i + step_size/2)                 #calculates k2
            k3 = ode(i + step_size/2)                 #calculates k3
            k4 = ode(i + step_size)                   #calculates k4
            k_bar = (k1 + 2*k2 + 2*k3+k4) / 6         #calulated k bar
            y.append(y[ct-1] + (k_bar * step_size))   #updates y each iteration based on kbar and step size and previous y
            ct = ct + 1                               #updates iteration variable
    
    return y
