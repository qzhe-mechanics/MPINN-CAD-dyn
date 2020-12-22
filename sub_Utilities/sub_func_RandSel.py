"""
@author     : Qizhi He @ PNNL (qizhi.he@pnnl.gov)
Decription  : Subfunctions for select measurement
Version: 1.1 @ 2020.02.04
Version: 1.2 @ 2020.03.04 [use lb and ub]
"""
import numpy as np
from pyDOE import lhs   # The experimental design package for python; Latin Hypercube Sampling (LHS)
# import sobol_seq  # require https://pypi.org/project/sobol_seq/


def sel_measurement(seed_num,flag,N,N_k):
	if flag == 1:
		np.random.seed(seed_num)
		idx_k = np.random.choice(N, N_k, replace=False) # Randomly select N_k points from N
	else:
		raise NotImplementedError
	return idx_k


def sel_colpoint(seed_num,flag,N,N_f,N_max,dim=None,lb=None,ub=None):
	if flag == 1:
		np.random.seed(seed_num)
		idx_f = np.random.choice(N, N_f, replace=False) 	# Randomly select N_k points from N
		x_nor = np.empty((0,2),dtype=float) 
	elif flag == 21:
		np.random.seed(seed_num)
		idx_f = np.empty((0,1),dtype=int) 
		x_nor = np.random.rand(N_f, dim) 			        # Use Rand for collocation point
	elif flag == 22:
		np.random.seed(seed_num)
		idx_f = np.empty((0,1),dtype=int) 
		x_nor = lhs(dim,N_max)[:N_f,:]
	else:
		raise NotImplementedError
	if lb is not None:
		x_nor = lb  + (ub-lb) * x_nor
	return x_nor, idx_f