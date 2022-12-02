#! /usr/local/bin/python3.4


# firstLast_nth_order_ODE.py
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

'Worked on code with or received help from Damen Wilson, Chris Kannmacher, and Safa Chowdhury'
'Ben McAteer, 0029592670'

def irange(start, stop, step):
    """ Create a generator to iterate over the interval we wish to solve the ODE on
    """
    while start < stop:
        yield start
        start += step


def rk(odes, interval, step_size, initial_values):
    """  

    Parameters
    ----------
    odes: a list of python functions
    interval: iterate returned from irange
    step_size: numerical number of step size h
    initial_values: a list of initial values from u1 to uN

    Returns
    -------
    u :  a list of discrete values with u[:][0] = y
    
    enter your code here

    """
   
    
    u = [initial_values] # initial value for US
    ct = 1 #initialized count variable
    n = range(len(odes)) #creates a count variable that changes per number of Odes "N"
    
    for i in interval: #create a for loop in size of iteration
    
    #Initialized empty arrays
        k1 = [] 
        k1_alt = []
        k2 = []
        k2_alt = []
        k3 = []
        k3_alt = []
        k4 = []
        k_bar = []
        u_alt = [] 
        
        if ct <= (len(interval)-1): #keeps length of y the same length of for loop, keeping dimensions same
            
            for d in (n): #iterates for N number of odes 
               
               k1.append(odes[d](i,u[ct-1])) #calculates k1 for each ode
               k1_alt.append(u[ct-1][d] + k1[d] * step_size * 0.5) #calculates the k1 that will be used to calculate k2
              
            
            for d in (n): #iterates for N number of odes 
               
               k2.append(odes[d](i + step_size * 0.5,k1_alt)) #calcualtes k2 for each ode
               k2_alt.append(u[ct-1][d] + k2[d] * step_size * 0.5) #calculates the k2 that will be used to calculate k3
               
           
            for d in (n): #iterates for N number of odes 
               
               k3.append(odes[d](i + step_size * 0.5,k2_alt)) #calculates k3 for each ode
               k3_alt.append(u[ct-1][d] + k3[d] * step_size) #calculates the k3 that will be used to calculate k4
               
             
            for d in (n): #iterates for N number of odes 
               
               k4.append(odes[d](i + step_size,k3_alt)) #calculates k4 for each ode  
            
            for d in (n): #iterates for N number of odes 
               
               k_bar.append((k1[d]+ 2*k2[d] + 2*k3[d] + k4[d]) / 6) #calculates kbar using each k value
             
            for d in (n): #iterates for N number of odes
            
               u_alt.append(u[ct-1][d] + k_bar[d] * step_size) #appends a temporary u value given the k bar and the previous u value
               
            u.append(u_alt) # adds the temporary u value into u
            
        ct = ct + 1 #iterates the coutn variable
        
    # Compute the output u, so that u is a list with length equals to length of intervals (T), and each element in u will be a list 
    # with length equal to number of order (N)
    # u = [u_0, u_1, ... u_T-1], with u_t = [u1_t, u2_t, ... uN_t] for t in [0, T-1]
    u_transpose = []
    for i in range(len(odes)):
        u_transpose.append([x[i] for x in u])  #transpose
    return u_transpose
