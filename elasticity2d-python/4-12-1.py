from Elasticity2D import FERun
from Exact import Exact_stress_4_12, Exact_deflection_4_12, sigmaxx_4_12, deflection_4_12

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

files_1 = ("4-12_1_1x5.json", "4-12_2_1x5.json")

files_2 = ("4-12_1_2x10.json", "4-12_2_2x10.json")

# Run FE analysis for all files using 2L element
nplot=21

n1 = len(files_1)
n2 = len(files_2)

xplot1 = np.zeros((5*nplot, n1))
xplot2 = np.zeros((10*nplot, n2))

xplot1_osp = np.zeros((5, n1))
xplot2_osp = np.zeros((10, n2))

deflection_1 = np.zeros((5*nplot, n1))
deflection_2 = np.zeros((10*nplot, n2))

deflection_1_osp = np.zeros((5, n1))
deflection_2_osp = np.zeros((10, n2))

sigma_xx_1 = np.zeros((5*nplot, n1))
sigma_xx_2 = np.zeros((10*nplot, n1))

sigma_xx_1_osp = np.zeros((5, n1))
sigma_xx_2_osp = np.zeros((10, n1))

for i in range(n1):
    FERun("4-12/"+files_1[i])

    xplot1[:,i], deflection_1[:,i], xplot1_osp[:,i], deflection_1_osp[:,i] = deflection_4_12(1)
    xplot1[:,i], sigma_xx_1[:,i], xplot1_osp[:,i], sigma_xx_1_osp[:,i] = sigmaxx_4_12(1)


for i in range(n2):
    FERun("4-12/"+files_2[i])

    xplot2[:,i], deflection_2[:,i], xplot2_osp[:,i], deflection_2_osp[:,i] = deflection_4_12(2)
    xplot2[:,i], sigma_xx_2[:,i], xplot2_osp[:,i], sigma_xx_2_osp[:,i] = sigmaxx_4_12(2)

fig, (ax0) = plt.subplots(1,1)
ax0.set_title('T4-12-1(deflection)')
ax0.set_ylabel('v')
# ax0.set_ylim(-0.1,0.1)

line1, = ax0.plot(xplot1[:,0], deflection_1[:,0],'--',label='1x5(1)')
line2, = ax0.plot(xplot1[:,1], deflection_1[:,1],'-',label='2x10(1)')
line3, = ax0.plot(xplot2[:,0], deflection_2[:,0],'--',label='1x5(2)')
line4, = ax0.plot(xplot2[:,1], deflection_2[:,1],'-',label='2x10(2)')

#Optimal deflection point
ax0.scatter(xplot1_osp,deflection_1_osp,edgecolors='k')
ax0.scatter(xplot2_osp,deflection_2_osp,edgecolors='k')

Exact_deflection_4_12(ax0)

ax0.legend()

tikzplotlib.save("T4-12-1(deflection).tex")

plt.savefig("T4-12-1(deflection).pdf")
plt.show()



fig, (ax1) = plt.subplots(1,1)
ax1.set_title('T4-12-1(stress)')
ax1.set_ylabel('sigma_xx')
# ax1.set_ylim(-1.0,0.4)

line1, = ax1.plot(xplot1[:,0], sigma_xx_1[:,0],'--',label='1x5(1)')
line2, = ax1.plot(xplot1[:,1], sigma_xx_1[:,1],'-',label='2x10(1)')
line3, = ax1.plot(xplot2[:,0], sigma_xx_2[:,0],'--',label='1x5(2)')
line4, = ax1.plot(xplot2[:,1], sigma_xx_2[:,1],'-',label='2x10(2)')

#Optimal stress point
ax1.scatter(xplot1_osp,sigma_xx_1_osp,edgecolors='k')
ax1.scatter(xplot2_osp,sigma_xx_2_osp,edgecolors='k')

Exact_stress_4_12(ax1)

ax1.legend()

tikzplotlib.save("T4-12-1(stress).tex")

plt.savefig("T4-12-1(stress).pdf")
plt.show()