# Common imports
import os
import pandas as pd
import numpy as np

# Where to save the figures and data files
DATA_ID = "../Outputs/"


from numpy import log2, zeros, mean, var, sum, loadtxt, arange, array, cumsum, dot, transpose, diagonal, sqrt
from numpy.linalg import inv
import numpy as np

def block(x):
    # preliminaries
    n = len(x)
    d = int(log2(n))
    s, gamma = zeros(d), zeros(d)
    mu = np.mean(x)

    # estimate the auto-covariance and variances 
    # for each blocking transformation
    for i in arange(0,d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*sum( (x[0:(n-1)]-mu)*(x[1:n]-mu) )
        # estimate variance of x
        s[i] = np.var(x)
        # perform blocking transformation
        x = 0.5*(x[0::2] + x[1::2])
   
    # generate the test observator M_k from the theorem
    M = (cumsum( ((gamma/s)**2*2**arange(1,d+1)[::-1])[::-1] )  )[::-1]

    # we need a list of magic numbers
    q =array([6.634897,9.210340, 11.344867, 13.276704, 15.086272, 16.811894, 18.475307, 20.090235, 21.665994, 23.209251, 24.724970, 26.216967, 27.688250, 29.141238, 30.577914, 31.999927, 33.408664, 34.805306, 36.190869, 37.566235, 38.932173, 40.289360, 41.638398, 42.979820, 44.314105, 45.641683, 46.962942, 48.278236, 49.587884, 50.892181])

    # use magic to determine when we should have stopped blocking
    for k in arange(0,d):
        if(M[k] < q[k]):
            break
    if (k >= d-1):
        print("Warning: Use more data")
    return mu, s[k]/2**(d-k)


if __name__ == '__main__':
    data = np.loadtxt('../Outputs/Energies.dat')
    step = int(2**18)

    print(len(data), step)

    for i in range(int(len(data)/step)):
        #x = data[-1:-step-1:-1]
        x = data[i*step:(i+1)*step]
        j = i+1
        #print(data[i*step:j*step])
        #print(np.sum(data[i*step:(i+1)*step]))
        #print(mean(data[i*step:(i+1)*step]))
        (mean, var) = block(np.array(x)) 
        std = sqrt(var)
        values ={'Mean':[mean], 'STDev':[std]}
        frame = pd.DataFrame(values,index=['Values'])
        print(frame)