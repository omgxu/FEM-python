from Elasticity2D import FERun
from Exact import Exact_stress_4_12, Exact_deflection_4_12, sigmaxx_4_12, deflection_4_12, ErrorNorm_4_12

import math
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

h_all = []
L2Norm_all = []
EnNorm_all = []

for i in range(n1):
    FERun("4-12/"+files_1[i])

    xplot1[:,i], deflection_1[:,i], xplot1_osp[:,i], deflection_1_osp[:,i] = deflection_4_12(1)
    xplot1[:,i], sigma_xx_1[:,i], xplot1_osp[:,i], sigma_xx_1_osp[:,i] = sigmaxx_4_12(1)

    if i == 1:
        h, L2Norm, EnNorm = ErrorNorm_4_12()
        h_all.append(h)
        L2Norm_all.append(L2Norm)
        EnNorm_all.append(EnNorm)

for i in range(n2):
    FERun("4-12/"+files_2[i])

    xplot2[:,i], deflection_2[:,i], xplot2_osp[:,i], deflection_2_osp[:,i] = deflection_4_12(2)
    xplot2[:,i], sigma_xx_2[:,i], xplot2_osp[:,i], sigma_xx_2_osp[:,i] = sigmaxx_4_12(2)

    if i == 1:
        h, L2Norm, EnNorm = ErrorNorm_4_12()
        h_all.append(h)
        L2Norm_all.append(L2Norm)
        EnNorm_all.append(EnNorm)

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

# 绘制对数坐标图

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制曲线
plt.plot(h_all, L2Norm_all, marker='o', label='L2Norm')

# 设置对数刻度
plt.xscale('log')  # 横轴为对数刻度
plt.yscale('log')  # 纵轴为对数刻度

# 添加标题和标签
plt.title('L2Norm vs h')
plt.xlabel('h (log scale)')
plt.ylabel('L2Norm (log scale)')

# 添加网格
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

# 添加图例
plt.legend()

# 显示图形
plt.show()

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制曲线
plt.plot(h_all, EnNorm_all, marker='o', label='EnNorm')

# 设置对数刻度
plt.xscale('log')  # 横轴为对数刻度
plt.yscale('log')  # 纵轴为对数刻度

# 添加标题和标签
plt.title('EnNorm vs h')
plt.xlabel('h (log scale)')
plt.ylabel('EnNorm (log scale)')

# 添加网格
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

# 添加图例
plt.legend()

# 显示图形
plt.show()

print('L2范数收敛率: ', (math.log10(L2Norm_all[1])-math.log10(L2Norm_all[0]))/(math.log10(h_all[1])-math.log10(h_all[0])))
print('能量范数收敛率: ', (math.log10(EnNorm_all[1])-math.log10(EnNorm_all[0]))/(math.log10(h_all[1])-math.log10(h_all[0])))