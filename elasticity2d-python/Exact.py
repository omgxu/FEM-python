#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot exact solutions for some problems
	ExactSolution_Cantilever: Plot the exact solution of the cantilever given
	in Example 10.1 in Fish's textbook

Created on Aug. 15 2022

@author: jsli@163.com, xzhang@tsinghua.edu.cn
"""

import numpy as np
import math 
import matplotlib.pyplot as plt
import FEData as model
from utitls import gauss

from Elast2DElem import NmatElast2D, BmatElast2D


def Exact(ax1):
	"""
	Plot the exact deflection and moment Mx along the centerline
	of the plate(Example 6-3) in ax1 and ax2, respectively.

	Args:
		ax1 : axis to draw deflection distribution
		ax2 : axis to draw moment Mx distribution
	"""
	a=1
	b=4
	pa=1
	pb=0
	E=13
	
	dx = 0.1
	nx  = math.ceil((b-a) / dx)
	x = np.arange(a, b, dx)
	sigma_rr = np.zeros(nx, np.float)

	for index, xi in enumerate(x):
		sigma_rr[index] = (pa*a**2-pb*b**2)/(b**2-a**2) - a**2*b**2/(b**2-a**2)/xi**2*(pa-pb)

	xplot = np.arange(a, b, dx)
	line5, = ax1.plot(xplot, sigma_rr, 'r', label='Exact')

def Exact_stress_4_12(ax):
	a = 0.0
	b = 5.0
	dx = 0.1
	nx  = math.ceil((b-a) / dx)
	sigma_xx = np.zeros(nx, np.float)
	x = np.arange(a, b, dx)
	for index, xi in enumerate(x):
		sigma_xx[index] = 480 * math.sqrt(3) / 12
	ax.plot(x, sigma_xx, 'r', label='Exact')

def Exact_deflection_4_12(ax):
	a = 0.0
	b = 5.0
	dx = 0.1
	nx  = math.ceil((b-a) / dx)
	v = np.zeros(nx, np.float)
	x = np.arange(a, b, dx)
	for index, xi in enumerate(x):
		v[index] = -24 * xi**2 / 1000
	ax.plot(x, v, 'r', label='Exact')

def sigmarr():
	"""
	Plot deflection and moment Mx distributions along the radius
	
	"""
	nplot=21
	
	xplot = np.zeros(4*nplot)
	sigma_rr = np.zeros(4*nplot)
	
	xplot_osp = np.zeros(4)
	sigma_osp = np.zeros(4)
	
	e_all = np.array([7, 15, 23, 31])
	
	for index, e in enumerate(e_all):
		
		# get coordinate and deflection of element nodes
		je = model.IEN[:, e] - 1
		C = np.array([model.x[je], model.y[je]]).T
		de = model.d[model.LM[:,e]-1]
		
		# equally distributed coordinates on the psi = 1 line of an element
		
		xplot_e = np.linspace(C[1,0], C[2,0], nplot)
		etaplot = 0.0
		psiplot = (2*xplot_e - C[1,0] - C[2,0])/(C[2,0] - C[1,0])
		
		sigma_rr_e = np.zeros(nplot)
		sigma_all = np.zeros(3)
		n = np.array([0.980785280403230, 0.195090322016128])
		
		for i in range(nplot):
			psi = psiplot[i]
			B, detJ = BmatElast2D(etaplot, psi, C)
			sigma_all = model.D@B@de
			sigma_rr_e[i] = (sigma_all[0]+sigma_all[1])/2.0 + (sigma_all[0]-sigma_all[1])/2.0*n[0] - sigma_all[2]*n[1]
		
		xplot[index*nplot:(index+1)*nplot] = xplot_e[:]
		sigma_rr[index*nplot:(index+1)*nplot] = sigma_rr_e[:]
		
		xplot_osp[index] = xplot_e[10]
		sigma_osp[index] = sigma_rr_e[10]
		
	return xplot, sigma_rr, xplot_osp, sigma_osp

def sigmaxx_4_12(N):
	nplot=21
	
	xplot = np.zeros(5*N*nplot)
	sigma_xx = np.zeros(5*N*nplot)
	
	xplot_osp = np.zeros(5*N)
	sigma_osp = np.zeros(5*N)
	
	if N == 1:
		e_all = np.array([0, 1, 2, 3, 4])
		psiplot = -0.57735
	elif N == 2:
		e_all = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
		psiplot = -0.1547

	for index, e in enumerate(e_all):

		# get coordinate and deflection of element nodes
		je = model.IEN[:, e] - 1
		C = np.array([model.x[je], model.y[je]]).T
		de = model.d[model.LM[:,e]-1]

		# equally distributed coordinates on the psi = 1 line of an element
		xplot_e = np.linspace(C[1,0], C[2,0], nplot)
			
		etaplot = (2*xplot_e - C[1,0] - C[2,0])/(C[2,0] - C[1,0])

		sigma_xx_e = np.zeros(nplot)
		sigma_all = np.zeros(3)
		
		for i in range(nplot):
			eta = etaplot[i]
			B, detJ = BmatElast2D(eta, psiplot, C)
			sigma_all = model.D@B@de
			sigma_xx_e[i] = sigma_all[0]

		xplot[index*nplot:(index+1)*nplot] = xplot_e[:]
		sigma_xx[index*nplot:(index+1)*nplot] = sigma_xx_e[:]
		
		xplot_osp[index] = xplot_e[10]
		sigma_osp[index] = sigma_xx_e[10]

	return xplot, sigma_xx, xplot_osp, sigma_osp

def deflection_4_12(N):

	nplot=21
	
	xplot = np.zeros(5*N*nplot)
	deflection = np.zeros(5*N*nplot)
	
	xplot_osp = np.zeros(5*N)
	deflection_osp = np.zeros(5*N)
	
	if N == 1:
		e_all = np.array([0, 1, 2, 3, 4])
		psiplot = 0.0
	elif N == 2:
		e_all = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
		psiplot = 1.0

	for index, e in enumerate(e_all):

		# get coordinate and deflection of element nodes
		je = model.IEN[:, e] - 1
		C = np.array([model.x[je], model.y[je]]).T
		de = model.d[model.LM[:,e]-1]

		# equally distributed coordinates on the psi = 1 line of an element
		xplot_e = np.linspace(C[1,0], C[2,0], nplot)
			
		etaplot = (2*xplot_e - C[1,0] - C[2,0])/(C[2,0] - C[1,0])

		deflection_e = np.zeros(nplot)
		
		for i in range(nplot):
			eta = etaplot[i]
			N = NmatElast2D(eta, psiplot)
			u_h = N@de
			deflection_e[i] = u_h[1]

		xplot[index*nplot:(index+1)*nplot] = xplot_e[:]
		deflection[index*nplot:(index+1)*nplot] = deflection_e[:]
		
		xplot_osp[index] = xplot_e[10]
		deflection_osp[index] = deflection_e[10]

	return xplot, deflection, xplot_osp, deflection_osp


def ErrorNorm_4_12():
	"""
	Calculate and print the error norm (L2 and energy norm) of displacement in T4-12
	"""

	w, gp = gauss(model.ngp)

	L2Norm = 0.0
	EnNorm = 0.0

	for e in range(model.nel):

		# get coordinate and displacement of element nodes
		je = model.IEN[:, e] - 1
		C = np.array([model.x[je], model.y[je]]).T
		de = model.d[model.LM[:,e]-1]

		for i in range(model.ngp):
			for j in range(model.ngp):
				eta = gp[i]
				psi = gp[j]

				xt = C[0,0] + 0.5*(C[1,0] - C[0,0])*(eta + 1.0)
				yt = C[0,1] + 0.5*(C[2,1] - C[0,1])*(psi + 1.0)

				N = NmatElast2D(eta, psi)
				B, detJ = BmatElast2D(eta, psi, C)

				u_h = N@de
				u_ex = 48*xt*yt/1000  # Exact displacement for the problem T4-12
				L2Norm += (u_h[0] - u_ex)**2 * detJ * w[i] * w[j]

				s_h = B@de
				s_ex = 480*yt  # Exact strain for the problem T4-12
				EnNorm += (s_h[0] - s_ex)**2 * detJ * w[i] * w[j]
		
		L2Norm = math.sqrt(L2Norm)
		EnNorm = math.sqrt(EnNorm)

		length = 5.0

		N_h = math.sqrt(model.nel/5)

		h = length/(5*N_h)

		print("h = %13.6E, L2Norm = %13.6E, EnNorm = %13.6E" %(h, L2Norm, EnNorm))

		return h, L2Norm, EnNorm
