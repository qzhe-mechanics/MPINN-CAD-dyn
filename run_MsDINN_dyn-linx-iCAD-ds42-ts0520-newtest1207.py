import os
import numpy as np
from pyDOE import lhs   # The experimental design package for python; Latin Hypercube Sampling (LHS)

## Randomize Initialization
N_i = 1
if N_i > 1:
	np.random.seed(1234)
	rand_seed = np.random.randint(0,2000,N_i)
else:
	rand_seed = np.array([111])

N_s = 1 
rand_seed_mea = 111
print(rand_seed,rand_seed_mea)

# # *********************************************************
# Use MsDINN_dyn version 3
# # *********************************************************

# Note: default (quick check)
array_k  = np.array([40])
array_h  = np.array([40])
array_c  = np.array([100])
array_f  = np.array([2000])
num_data = 1
array_nn = np.array([56])
Nt       = 39       # Evey 0.5 min

## Note: test ts-1-5; sel22; bf ---
for i_nn in array_nn:
	directory = '../MsDINN-CAD-output/dyn_v3_iCAD/test_ts42_new/Linx_ts0520_op2m400i30w_bf_lr2e4_nc56_nt39fc20k_Kh40C100_rep/'
	src_dir   = '../MsDINN-CAD-dyn/main_MsDINN_dyn_bwADE.py'

	if not os.path.exists(directory):
		os.makedirs(directory)

	for ii in range(0,num_data):
		num_k = array_k[ii]
		num_h = array_h[ii]
		num_f = array_f[ii]
		num_c = array_c[ii]

		for i_s in rand_seed:
			os.system("python {0} {1} {2} {3} \
				{4} {5} {6} {7} 20000 {10}\
				1.0 1.0 1.0 1.0 1.0\
				42 'i-CAD' 1  \
				{9} {9} 56\
				2 400 300000 0.0002 'BFGS' 0 0 1.e-8 \
				2 0 30000 0.001 'BFGS' 0.001 \
				1 22 {8} 0 1 'ts-05-20'".format(src_dir,directory,i_s,N_s,
					num_k,num_h,num_f,num_c,
					rand_seed_mea,i_nn,Nt))

