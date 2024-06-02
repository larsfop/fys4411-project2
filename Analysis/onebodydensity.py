
import numpy as np
import matplotlib.pyplot as plt

hist = np.loadtxt('../Outputs/histogram_IW.dat',dtype=np.float64)
hist_max = np.max(hist)
hist = hist/hist_max
plt.hist(hist,label='W/ Jastrow', bins=100, histtype='step', density=True)
hist = np.loadtxt('../Outputs/histogram_SG.dat',dtype=np.float64)
hist_max = np.max(hist)
hist = hist/hist_max
plt.hist(hist,label='W/o Jastrow', bins=100, histtype='step', density=True)

plt.legend()
plt.xlabel('Distance from particle')
plt.ylabel('Number of particles')
plt.savefig('plot_onebodydensity_6N.pdf')

plt.show()