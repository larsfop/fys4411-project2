
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from blocking import block

if len(sys.argv) > 1:
    showplot=bool(sys.argv[1])
else:
    showplot = False

def analytical_energy():
    alpha = np.linspace(0.1,2.08,1000)
    return alpha, (4*alpha**2 + 1)/(8*alpha)


def compare_interaction(solver):
    title = {'SM' : 'Standard Metropolis', 'MH' : 'Metropolis Hastings'}
    data = pd.read_csv(f'../Outputs/2N_{solver}_NS_NI.dat', sep='\s+')
    plt.figure()
    plt.plot(data['alpha'], data['Energy'], label='No Coulomb interaction')
    data = pd.read_csv(f'../Outputs/2N_{solver}_NS_YI.dat', sep='\s+')
    plt.plot(data['alpha'], data['Energy'], label='Coulomb interaction')
    plt.vlines(1, 20, 0, colors='gray', linestyles='--')
    plt.ylim(0,12)
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'E [$\hbar\omega$]')
    plt.title(title[solver])
    plt.grid()
    plt.savefig(f'plot_compare_{solver}.pdf')



solvers = ['SM', 'MH']
for solver in solvers:
    compare_interaction(solver)

data = pd.read_csv(f'../Outputs/2N_OP_MH_NS_NI.dat', sep='\s+')
data = data[data.Thread==0]
plt.figure()
plt.plot(data.alpha, data.Energy)



if showplot:
    plt.show()