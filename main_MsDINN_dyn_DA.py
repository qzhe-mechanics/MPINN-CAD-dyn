"""
-------------------------------------------------------------------------------
=============================== MsDINN-CAD Library ================================
-------------------------------------------------------------------------------
=========================================
Author: Qizhi He
Computational Mathematics Group
Advanced Computing, Mathematics and Data Division
Pacific Northwest National Laboratory 
Email: qizhi.he@pnnl.gov

Copyright (c) 2020 Qizhi He - licensed under the MIT License
For a full copyright statement see the accompanying LICENSE.md file.
=========================================

The code is fully functional with the following module versions:
	- tensorflow: 1.13.1
	- numpy: 1.16.2
	- scipy: XXX
	- matplotlib: XXX
	- Python: 3.6
-------------------------------------------------------------------------------
Codes description: Time-dependent 2D Darcy-Advection-Dispersion Equation (2D) by MPINN
Codes: Reference: CAD-v6s5 (pro5), CADn_dyn, MDINN_Hanford
* flag_pro: {i-CAD: inverse learning about all K, h & C} 
* flag_data: {12: sin_smooth; 41: k02_normal; 42: k05_normal}
* flag_t_test: {ts-1-20'}
* if_BC_h: {0: Not include any data from h on boundaries; 1: Consider Dirichlet and Neumann; 2: Assume all Dirichlet on all boundary}
---Note---
* CAD Applications
* k and h are steady variables, c is time-dependent variable
* Time-independent Boundary conditions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### History ###
<2020.12.22> [beta v0] [Clean version i-CAD][Done]
"""
'''Public Library'''
import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from pyDOE import lhs                       # The experimental design package for python; Latin Hypercube Sampling (LHS)
from scipy.interpolate import griddata
import time               
import tensorflow as tf
'''Customized Library'''
sys.path.insert(0, 'subcode_MsDINN_dyn/')
sys.path.insert(0, 'sub_Utilities/')
from class_MsDINN_probl import *
from sub_func_analysis import *
from sub_func_plot import *
from sub_func_RandSel import *             
from sub_MsDINN_dyn import *

sys_len = len(sys.argv)
print(sys_len)

'''System Input'''
path_f   = sys.argv[1]      if len(sys.argv) > 1 else '../MsDINN-CAD-output/dyn_v3_iCAD/temp_test/test_dt42/temp_ts010_opt2m100_nopre_rep3/'
seed_num = int(sys.argv[2]) if len(sys.argv) > 2 else 111
N_s      = int(sys.argv[3]) if len(sys.argv) > 3 else 1    # Initialization of NNs
N_k      = int(sys.argv[4]) if len(sys.argv) > 4 else 200   # Measurement of Conductivity
N_h      = int(sys.argv[5]) if len(sys.argv) > 5 else 200   # Measurement of Pressure head
N_f      = int(sys.argv[6]) if len(sys.argv) > 6 else 200  # Collocations
N_c      = int(sys.argv[7]) if len(sys.argv) > 7 else 40   # Measurement of Concentration (at t_init)
N_fc     = int(sys.argv[8]) if len(sys.argv) > 8 else 200  # Collocations for time-CAD (40K)
Nt       = int(sys.argv[9]) if len(sys.argv) > 9 else 20   # Approx. time step (117)
# Parameter to control the loss function
sys_id = 10
para_k   = float(sys.argv[sys_id])   if len(sys.argv) > sys_id   else 1.0
para_h   = float(sys.argv[sys_id+1]) if len(sys.argv) > sys_id+1 else 1.0
para_kh  = float(sys.argv[sys_id+2]) if len(sys.argv) > sys_id+2 else 1.0
para_c   = float(sys.argv[sys_id+3]) if len(sys.argv) > sys_id+3 else 1.0
para_khc = float(sys.argv[sys_id+4]) if len(sys.argv) > sys_id+4 else 1.0
# System control
sys_id += 5  
flag_data = int(sys.argv[sys_id])  if len(sys.argv) > sys_id else 42
sys_id += 1
flag_pro   = sys.argv[sys_id]      if len(sys.argv) > sys_id else 'i-CAD'
sys_id += 1
IF_plot   = int(sys.argv[sys_id])  if len(sys.argv) > sys_id else 1
sys_id += 1
# NN Architech
type_NNk  = int(sys.argv[sys_id])  if len(sys.argv) > sys_id  else 36
sys_id += 1
type_NNh  = int(sys.argv[sys_id])  if len(sys.argv) > sys_id  else 36
sys_id += 1
type_NNc  = int(sys.argv[sys_id])  if len(sys.argv) > sys_id  else 36
sys_id += 1
# Optimizer
# type_op: 1: L-BFGS ; 2. Adam; 3. BFGS - Adam 4: Other
type_op    = int(sys.argv[sys_id])    if len(sys.argv) > sys_id      else 2
batchs     = int(sys.argv[sys_id+1])  if len(sys.argv) > sys_id+1    else 100            # number of batchs
num_epochs = int(sys.argv[sys_id+2])  if len(sys.argv) > sys_id+2    else 1000         # number of itrations
learn_rate = float(sys.argv[sys_id+3])  if len(sys.argv) > sys_id+3  else 0.001         # learning rate (default for Reg:0.00001;PINN 0.0001)
# 'standard', 'BFGS', 'BFGS-f'
flag_solv  = sys.argv[sys_id+4]  if len(sys.argv) > sys_id+4         else 'standard'	
sys_id += 5
conv_opt   = float(sys.argv[sys_id])   if len(sys.argv) > sys_id     else 0        # The conv coefficient for opt7; 0: not set a tolerance
sys_id += 1
# Regularization
type_reg    = int(sys.argv[sys_id])   if len(sys.argv) > sys_id       else 0            # 1: L1 Reg (default)
coe_reg     = float(sys.argv[sys_id+1]) if len(sys.argv) > sys_id+1   else 1e-8         # default (1e-8)
sys_id += 2
# Optimizer for pretraining
type_op_pre    = int(sys.argv[sys_id])    if len(sys.argv) > sys_id      else 2 
batchs_pre     = int(sys.argv[sys_id+1])  if len(sys.argv) > sys_id+1    else 0            # number of batchs
num_epochs_pre = int(sys.argv[sys_id+2])  if len(sys.argv) > sys_id+2    else 2000         # number of itrations
learn_rate_pre = float(sys.argv[sys_id+3])  if len(sys.argv) > sys_id+3  else 0.001         # learning rate (default for Reg:0.00001;PINN 0.0001)
# 'standard', 'BFGS', 'BFGS-f'
flag_solv_pre  = sys.argv[sys_id+4]  if len(sys.argv) > sys_id+4         else 'standard'   		
conv_opt_pre   = float(sys.argv[sys_id+5])   if len(sys.argv) > sys_id+5 else 0        # The conv coefficient for opt7
sys_id += 6
# Other control
# type_mea=22 in v6s5 become type_mea=1 & flag_sel_col= 1
# 1: using fixed grids; 2*: random select from domain; 3*: manually controlled
type_mea    = int(sys.argv[sys_id]) if len(sys.argv) > sys_id else 1  
sys_id += 1
flag_sel_col = int(sys.argv[sys_id]) if len(sys.argv) > sys_id else 22
sys_id += 1
seed_mea = int(sys.argv[sys_id])    if len(sys.argv) > sys_id else 111
sys_id += 1
flag_lsty = int(sys.argv[sys_id])   if len(sys.argv) > sys_id else 0
sys_id += 1
flag_lss  = int(sys.argv[sys_id])   if len(sys.argv) > sys_id else 1
sys_id += 1
flag_t_test = sys.argv[sys_id]  if len(sys.argv) > sys_id else 'ts-05-5' # 
sys_id += 1
t_test  = float(sys.argv[sys_id]) if len(sys.argv) > sys_id else 5/60 # [hour]

'''Jason Problem Input'''
param_file = './para_MsDINN_dyn.json'
with open(param_file) as json_params:
	pro_params = json.load(json_params)

if flag_pro == 'i-CAD':
	if_BC_h       = 1

Nc_mea  	  = pro_params['Nc_mea']

# ====================================================================
## Print
# ====================================================================
if not os.path.exists(path_f):
	os.makedirs(path_f)

path_fig = path_f+'figures'+'/'
if not os.path.exists(path_fig):
	os.makedirs(path_fig)

f_rec     = open(path_f+'record.out', 'a+')
f1        = open(path_f+'record_data.out', 'a+')
f1_loss   = open(path_f+'record_loss.out', 'a+')
f1_loss2  = open(path_f+'record_loss2.out', 'a+')
f2_weight = open(path_f+'record2_weight.out', 'a+')
f2_bias   = open(path_f+'record2_bias.out', 'a+')

print("Problem & data set: {} {}".format(flag_pro,flag_data),file=f_rec)
print("test: Nk: {}, Nh: {}, Nf: {}, N_c: {}, N_fc: {}"\
	.format(N_k, N_h, N_f, N_c, N_fc), file=f_rec)
print("seed: {}, # Initial: {}".format(seed_num,N_s),file=f_rec)
print("Measurement Seed: {}, type: {}".format(seed_mea,type_mea),file=f_rec)
print("BCs for h: {}".format(if_BC_h),file=f_rec)
print("para: para_k: {}, para_h: {}, para_kh: {}, para_c: {}, para_khc: {}"\
	.format(para_k, para_h, para_kh, para_c, para_khc), file=f_rec)
print("NN architecture: {} {} {}".format(type_NNk, type_NNh, type_NNc), file=f_rec)
# print("NN activation: {} {} {}".format(type_actk, type_acth, type_actc), file=f_rec)   
print("Optimier: {} {} {} {} {} {}".format(type_op,batchs,num_epochs,learn_rate,flag_solv,conv_opt),file=f_rec)
print("Regularization & coefficient: {} {} | loss lsty = {}".format(type_reg,coe_reg,flag_lsty),file=f_rec)
print("Assimilate C data: {} {}".format(pro_params['IF_DataAssi'],Nc_mea),file=f_rec)

'''Solution Analysis'''
pro_dim = 3 # The dimensionality of problem 
Nf_max = 100000
# if flag_t_test == 'ts-1':
# 	mul_t_test = np.array([1/60, 2/60, 5/60, 10/60, 15/60, 20/60])
# 	t_min, t_max =  1/60, 30/60 # hr
# 	if_ref_exist = 1
# elif flag_t_test == 'ts-2':
# 	mul_t_test = np.array([1/60, 1.5/60, 2/60, 2.5/60, 3/60, 3.5/60, 4/60, 4.5/60, 5/60])
# 	t_min, t_max =  1/60, 10/60 # hr
# 	if_ref_exist = 0
if flag_t_test == 'ts-1-2':
	mul_t_test = np.array([1/60, 1.5/60, 2/60])
	t_min, t_max =  1/60, 2/60 # hr
	if_ref_existM = np.array([1,1,1])
	t_ref_lstr = ['1','1.5','2']
# elif flag_t_test == 'ts-1-5':
# 	mul_t_test = np.array([1/60, 1.5/60, 2/60, 2.5/60, 3/60, 3.5/60, 4/60, 4.5/60, 5/60])
# 	t_min, t_max =  1/60, 5/60 # hr
# 	if_ref_existM = np.array([1,1,1,1,1,1,1,1,1])
# 	t_ref_lstr = ['1','1.5','2','2.5','3','3.5','4','4.5','5']
elif flag_t_test == 'ts-1-20': # @2020.09.30
	mul_t_test_min = np.linspace(1,20,39)
	mul_t_test = mul_t_test_min/60  # Hour
	t_min, t_max =  1.0/60, 20/60 # hr
	t_ref_lstr = [str(t_i).rstrip('0').rstrip('.') for t_i in mul_t_test_min]
	if_ref_existM = [1] * mul_t_test.shape[0]
elif flag_t_test == 'ts-0-10': # @2020.10.27
	mul_t_test_min = np.linspace(0,10,21)
	mul_t_test = mul_t_test_min/60  # Hour
	t_min, t_max =  0.0/60, 10/60 # hr
	t_ref_lstr = [str(t_i).rstrip('0').rstrip('.') for t_i in mul_t_test_min]
	if_ref_existM = [1] * mul_t_test.shape[0]
elif flag_t_test == 'ts-1-10': # @2020.10.27
	mul_t_test_min = np.linspace(1,10,19)
	mul_t_test = mul_t_test_min/60  # Hour
	t_min, t_max =  1.0/60, 10/60 # hr
	t_ref_lstr = [str(t_i).rstrip('0').rstrip('.') for t_i in mul_t_test_min]
	if_ref_existM = [1] * mul_t_test.shape[0]
elif flag_t_test == 'ts-2-15': # @2020.10.26
	mul_t_test_min = np.linspace(2,15,27)
	mul_t_test = mul_t_test_min/60  # Hour
	t_min, t_max =  2.0/60, 15/60 # hr
	t_ref_lstr = [str(t_i).rstrip('0').rstrip('.') for t_i in mul_t_test_min]
	if_ref_existM = [1] * mul_t_test.shape[0]
elif flag_t_test == 'ts-2-5': # @2020.10.27
	mul_t_test_min = np.linspace(2,5,7)
	mul_t_test = mul_t_test_min/60  # Hour
	t_min, t_max =  2.0/60, 5/60 # hr
	t_ref_lstr = [str(t_i).rstrip('0').rstrip('.') for t_i in mul_t_test_min]
	if_ref_existM = [1] * mul_t_test.shape[0]
elif flag_t_test == 'ts-05-5':  # With all reference
	mul_t_test_min = np.linspace(0.5,5,10)
	mul_t_test = mul_t_test_min/60  # Hour
	t_min, t_max =  0.5/60, 5/60 # hr
	t_ref_lstr = [str(t_i).rstrip('0').rstrip('.') for t_i in mul_t_test_min]
	if_ref_existM = [1] * mul_t_test.shape[0]
elif flag_t_test == 'ts-05-20':  # With all reference
	mul_t_test_min = np.linspace(0.5,20,40)
	mul_t_test = mul_t_test_min/60  # Hour
	t_min, t_max =  0.5/60, 20/60 # hr
	t_ref_lstr = [str(t_i).rstrip('0').rstrip('.') for t_i in mul_t_test_min]
	if_ref_existM = [1] * mul_t_test.shape[0]
else:
	t_min, t_max = 1/60, 30/60 # hr
	if_ref_exist = 1


# test: make the output string consistent
# t_lstr_file = '{0:.1f}'.format(18.5)
# t_lstr_file = t_lstr_file.rjust(4,'0')
# t_test = 20/60 # [hour]
# trailing_removed = [s.rstrip("0") for s in listOfNum]
# leading_removed = [s.lstrip("0") for s in listOfNum]
# both_removed = [s.strip("0") for s in listOfNum]

if flag_t_test == 'None':
	print("test time: {0: 3e}".format(t_test),file=f_rec)
else:
	print("test type: {}, test time: {}".format(flag_t_test,mul_t_test),file=f_rec)

if flag_pro == 'i-CAD':
	type_op_pre     = 0
	learn_rate_pre  = 0
	conv_opt_pre    = 0
	flag_solv_pre   = 'standard'
	print("Optimier for Darcy: Empty",file=f_rec)

## Data
if flag_data == 12:
	path_data = './Data_dyn/Data_dyn_sin_smooth/'
	lstr_pth_data = path_data+'raw_data/plot_matlab_sin_'
elif flag_data == 42:
	path_data = './Data_dyn/Data_dyn_k05_normal/' # Correlation lenght = 0.2
	lstr_pth_data = path_data+'raw_data/plot_matlab_smooth_field_05_normal'

## Times
t_nod = np.linspace(t_min,t_max,Nt)
print("Analysis: t_min = {}, t_max = {}, Nt = {}".format(t_min,t_max, Nt),file=f_rec)

## Geometry
lb = np.array([0.0, 0.0])
ub = np.array([1.0, 0.5])

## Define Parameters
Nx = pro_params['data_Nx']
Ny = pro_params['data_Ny']
N_test  = Nx * Ny    
coe_L1, coe_L2  = pro_params['coe_L1'], pro_params['coe_L2'] # (m)
coe_q    = pro_params['coe_q'] # (m/hr)
coe_H2   = pro_params['coe_H2'] # (m)
coe_C0   = pro_params['coe_C0']
coe_phi  = pro_params['coe_phi']
coe_Dw   = pro_params['coe_Dw'] # m^2/hr
coe_aL   = pro_params['coe_aL'] # (m)
coe_aT   = pro_params['coe_aT'] # (m)
coe_tau  = coe_phi**(1/3) # Dw * tau

'''Plot setting'''
if_plot_C_sol = pro_params['if_plot_C_sol']
if_plot_pts2D = pro_params['if_plot_pts2D']
if_vis        = pro_params['if_vis']
if_plot_rl2   = pro_params['if_plot_rl2'] 	# Plot rl2 loss, require to know the exact solutions

'''Record setting'''
if_rec_predic = 1
if_rec_loss   = 1

# ====================================================================
## Start Analyse
# ====================================================================  
tf.reset_default_graph()
if __name__ == "__main__": 
	# ====================================================================
	## PREPROCESS
	# ====================================================================  
	'''Loading information for k & h'''	
	if flag_data == 12:
		# Load conductive coefficient, k(x,z)
			# if flag_data == 11:
			# 	k_star    = np.ones((h_star.shape[0],1))
		dataset_k = np.loadtxt(path_data+'ksx_ijk.dat', dtype=float)
		k_star    = dataset_k[:,np.newaxis]

		# Load hydraulic head, h(x,z)
		dataset_h = np.loadtxt(path_data+'plot_head.dat', dtype=float) # Load hydraulic head
		h_star    = dataset_h[:,2:3]

		Yl_h, Yr_h = h_star.min(0), h_star.max(0)
		h_star = h_star - Yl_h

		# Total Coordinates
		X_star         = dataset_h[:,0:2]   # Coordinates of all nodes (x,z)
	elif flag_data == 41:
		dataset_all = np.loadtxt(path_data+'plot_matlab_smooth_field_02_normal50d.dat', dtype=float)
		# Load hydraulic head, h(x,z)
		dataset_k = np.loadtxt(path_data+'smooth_field_02_normal.txt', dtype=float)
		k_star    = dataset_k[:][:,np.newaxis]      

		# Load hydraulic head, h(x,z)
		h_star    = dataset_all[:,4:5]
		Yl_h, Yr_h = h_star.min(0), h_star.max(0)
		h_star = h_star - Yl_h
		# Total Coordinates
		X_star         = dataset_all[:,[0,2]]  # Coordinates of all nodes (x,z)

	elif flag_data == 42:
		dataset_all = np.loadtxt(path_data+'plot_matlab_smooth_field_05_normal50d.dat', dtype=float)
		# Load hydraulic head, h(x,z)
		dataset_k = np.loadtxt(path_data+'smooth_field_05_normal.txt', dtype=float)
		k_star    = dataset_k[:][:,np.newaxis]      

		# Load hydraulic head, h(x,z)
		h_star    = dataset_all[:,4:5]
		Yl_h, Yr_h = h_star.min(0), h_star.max(0)
		h_star = h_star - Yl_h  # Be careful in CAD use Yl_h if needed
		# Total Coordinates
		X_star         = dataset_all[:,[0,2]]  # Coordinates of all nodes (x,z)

	N = X_star.shape[0]  # the column size of u_star, total nodes

	if Nx * Ny != N:
		print('Input Error: N')
		input()

	# if if_plot_C_sol == 1:
	# 	tM_test = np.array([t_min,20/60])
	# 	for ti in tM_test:
	# 		t_lstr = str(int(ti * 60))
	# 		dataset_c = np.loadtxt(path_data+'raw_data/plot_matlab_sin_'+t_lstr+'min.dat', dtype=float) # Load solute concentration
	# 		Cmt   = dataset_c[:,4:5]   # Concentration
	# 		sub_plt_surf2D(lb,ub,X_star,Cmt,ti*60,savefig=path_fig,visual=None,plt_eps=None)

	# ====================================================================
	## Training data 
	# ====================================================================
	# Note {Z: a new variable concatenate both x and t, z=(x,y,t)}
	
	'''Time-dependent data for C(x) (Initial condition)'''
	'''If for i-CAD, no initial condition'''

	if flag_pro == 'i-CAD':
		# ** None initial condition in C(x) but measurements
		Zm_train = np.empty((0,3),dtype=float)  
		Um_train = np.empty((0,1),dtype=float)
		
		N_c_tref    = N_c          # At each ref time instance, select N_c measurements
		if N_c_tref != 0:
			idx_c = sel_measurement(seed_num,type_mea,N,N_c_tref)
			X_c = X_star[idx_c,:]
		else:
			X_c = np.empty((0,2),dtype=float) 
		
		for II_t in range(mul_t_test.shape[0]):
			# Currently only works for cosine function
			t_ref_ls = t_ref_lstr[II_t]
			t_init = mul_t_test[II_t]

			dataset_c = np.loadtxt(lstr_pth_data+t_ref_ls+'min.dat', dtype=float) # Load solute concentration	
			if flag_data == 12:
				c_ref    = dataset_c[:,4:5]
			elif flag_data == 41 or flag_data == 42:
				c_ref    = dataset_c[:,6:7]   # Concentration
			else:
				raise NotImplementedError

			if N_c_tref != 0:
				Y_c = c_ref[idx_c,:]
			else:
				Y_c = np.empty((0,1),dtype=float)
					
			T_i = np.tile(t_init,N_c_tref).T[:,np.newaxis]
			zz1 = np.hstack((X_c[:], T_i))

			Zm_train = np.vstack([Zm_train,zz1])
			Um_train = np.vstack([Um_train,Y_c])

	'''Boundary Condtions for C(x)'''
	# Note: {The B.C. of C(x) is related to the simulation time scales. Not the same as measurements}
	if flag_pro == 'i-CAD':
		# Time-independent B.C. (Use the soluton at t_min to impose the boundary)
		t_init = t_min
		# t_min_lstr = str(int(t_init * 60))
		# t_min_lstr = str(round(t_init * 60,1))   # Keep one decimal if it is not a integer

		t_min_lstr = str(round(t_init * 60,1)).rstrip('0').rstrip('.') 

		if flag_data == 12:
			dataset_c = np.loadtxt(path_data+'raw_data/plot_matlab_sin_'+t_min_lstr+'min.dat', dtype=float) # Load solute concentration
			C_star    = dataset_c[:,4:5]        # Concentration at initial time
		elif flag_data == 41:
			dataset_c = np.loadtxt(path_data+'raw_data/plot_matlab_smooth_field_02_normal'+t_min_lstr+'min.dat', dtype=float) # Load solute concentration
			C_star    = dataset_c[:,6:7]   # Concentration
		elif flag_data == 42:
			dataset_c = np.loadtxt(path_data+'raw_data/plot_matlab_smooth_field_05_normal'+t_min_lstr+'min.dat', dtype=float) # Load solute concentration
			C_star    = dataset_c[:,6:7]   # Concentration	
		else:
			raise NotImplementedError
		
		# Dirichlet boundaries (edge 0 left)
		xb0, cb0 = X_star[::Nx,:], C_star[::Nx,:]           # approximating x1 = 0
		X_cbD = xb0
		Y_cbD = cb0

		# Neumann boundaries normal to x (edge 2)
		xb2 = np.zeros((Ny,2))
		xb2[:,0:1] = ub[0]
		xb2[:,1:2] = lb[1] + (ub[1]-lb[1])*lhs(1, Ny) 		# Exactly on x1 = L1
		
		X_cbN1 = xb2
		Y_cbN1 = np.zeros((X_cbN1.shape[0], 1))  
		
		# Neumann boundaries normal to y (edge 1,3)
		xb1 = np.zeros((Nx,2))
		xb1[:,0:1] = lb[0] + (ub[0]-lb[0])*lhs(1, Nx)
		xb3 = np.zeros((Nx,2))
		xb3[:,1:2] = ub[1]      
		xb3[:,0:1] = xb1[:,0:1]

		X_cbN2 = np.concatenate([xb1, xb3], axis = 0)   # approximating value
		Y_cbN2 = np.zeros((X_cbN2.shape[0], 1))
	
	## Add EBC into training data
	if len(X_cbD)!=0:  # If the EBC not empty
		Zm_b, Um_b = fun_BC2dyn(X_cbD,Y_cbD,t_nod,pro_dim)
		Zm_train = np.vstack([Zm_train,Zm_b])
		Um_train = np.vstack([Um_train,Um_b])

	## Add NBC into training data
	if len(X_cbN1)!=0:  # If the NBC1 not empty
		Zm_b, Um_b = fun_BC2dyn(X_cbN1,Y_cbN1,t_nod,pro_dim)
	else:
		Zm_b = np.empty((0,pro_dim),dtype=float) 
		Um_b = np.empty((0,1),dtype=float)
	Zm_cbN1_train = np.vstack((Zm_b))
	Um_cbN1_train = np.vstack((Um_b))

	if len(X_cbN2)!=0:  # If the NBC1 not empty
		Zm_b, Um_b = fun_BC2dyn(X_cbN2,Y_cbN2,t_nod,pro_dim)
	else:
		Zm_b = np.empty((0,pro_dim),dtype=float) 
		Um_b = np.empty((0,1),dtype=float)
	Zm_cbN2_train = np.vstack((Zm_b))
	Um_cbN2_train = np.vstack((Um_b))

	'''Collocation points for CAD (Used for imposing physics)'''
	# 1: using fixed grids; 2*: random select from domain; 3*: manually controlled
	if flag_sel_col == 31:
		# z_nor, _ = sel_colpoint(seed_num,flag_sel_col,N,N_fc,Nf_max,pro_dim,lb=None,ub=None)
		Nx_fc = round(N_fc/Nt)
		alpha_local = 0.4
		Nx_fc_global = round((1-alpha_local)* Nx_fc)
		Nx_fc_local  = round(alpha_local * Nx_fc)

		# x_nor_global = np.random.uniform(0.,1.,(Nx_fc_global,2))
		# np.random.seed(seed_num)
		# x_nor_local  = np.random.uniform(low=[0,0.35],high=[0.3,0.65],size=(Nx_fc_local,2))
		x_nor_global, _ = sel_colpoint(seed_num,22,N,Nx_fc_global,Nf_max,dim=2,lb=None,ub=None)
		lb_temp = np.array([0.0, 0.35])
		ub_temp = np.array([0.3, 0.65])
		x_nor_local, _ = sel_colpoint(seed_num,22,N,Nx_fc_local,Nf_max,dim=2,lb=lb_temp,ub=ub_temp)

		x_nor  = np.vstack([x_nor_global,x_nor_local])
		xm = lb  + (ub-lb)*x_nor
		
		# --- Test ---
		# sub_plt_pts2D(xm,savefig=path_fig, visual=if_vis)
		# print("something")
		# wait = input("PRESS ENTER TO CONTINUE.")		
		
		Ym  = np.zeros((xm.shape[0], 1))
		Zm_f_train, _ = fun_BC2dyn(xm,Ym,t_nod,pro_dim)
	
	elif flag_sel_col == 32:
		alpha_local = 0.4
		Nx_fc_global = round((1-alpha_local)* N_fc)
		Nx_fc_local  = round(alpha_local * N_fc)		
		
		z_nor_global, _ = sel_colpoint(seed_num,22,N,Nx_fc_global,Nf_max,dim=pro_dim,lb=None,ub=None)
		lb_temp = np.array([0.0, 0.35, 0.0])
		ub_temp = np.array([0.3, 0.65, 1.0])
		z_nor_local, _ = sel_colpoint(seed_num,22,N,Nx_fc_local,Nf_max,dim=pro_dim,lb=lb_temp,ub=ub_temp)

		z_nor  = np.vstack([z_nor_global,z_nor_local])
		
		Zm_f_train = np.zeros((N_fc,pro_dim))
		Zm_f_train = np.zeros((N_fc,pro_dim))
		Zm_f_train[:,0:1] = lb[0]  + (ub[0]-lb[0])*z_nor[:,0:1]
		Zm_f_train[:,1:2] = lb[1]  + (ub[1]-lb[1])*z_nor[:,1:2]
		Zm_f_train[:,2:3] = t_init + (t_max-t_init)*z_nor[:,2:3]

		# --- Test ---
		# sub_plt_pts3D(Zm_f_train,savefig=path_fig, visual=if_vis)
		# print("something")
		# wait = input("PRESS ENTER TO CONTINUE.")

		# --- Test ---
		# sub_plt_pts2D(Zm_f_train[:,0:2],savefig=path_fig, visual=if_vis)
		# print("something")
		# wait = input("PRESS ENTER TO CONTINUE.")	
	else:
		z_nor, _ = sel_colpoint(seed_num,flag_sel_col,N,N_fc,Nf_max,pro_dim,lb=None,ub=None)
		Zm_f_train = np.zeros((N_fc,pro_dim))
		Zm_f_train[:,0:1] = lb[0]  + (ub[0]-lb[0])*z_nor[:,0:1]
		Zm_f_train[:,1:2] = lb[1]  + (ub[1]-lb[1])*z_nor[:,1:2]
		Zm_f_train[:,2:3] = t_init + (t_max-t_init)*z_nor[:,2:3]

	Zm_f_train = np.vstack((Zm_f_train,Zm_train)) # Add all measurement into collocations
	
	# ====================================================================
	## Training data for k and h
	# ====================================================================	
	'''Measurement interior domain for k and h'''
	if N_k != 0:
		idx_k = sel_measurement(seed_num,type_mea,N,N_k)
		X_k = X_star[idx_k,:]               		# Coordinates
		Y_k = k_star[idx_k]				    # Selected values of K function
	else:
		X_k = np.empty((0,2),dtype=float) # Predefine the the proper dimensionality.
		Y_k = np.empty((0,1),dtype=float)    

	if_BC_K = pro_params['f_CAD_BC_K']
	
	if if_BC_K == 1:
		# Dirichlet boundaries (edge 0 left) # approximating x1 = 0 （every two lines）
		xb0, kb0 = X_star[::2*Nx,:], k_star[::2*Nx,:]          

		# Dirichlet boundaries (edge 1 bottom) # approximating x2 = 0 （every two pts)
		xb1, kb1 = X_star[0:Nx:2,:], k_star[0:Nx:2,:]  			

		# Dirichlet boundaries (edge 2 right) （every two lines）
		xb2, kb2 = X_star[Nx-1:N:2*Nx,:], k_star[Nx-1:N:2*Nx,:]

		# Dirichlet boundaries (edge 3 top) # approximating x2 = L2 （every two pts)
		xb3, kb3 = X_star[-Nx:N:2,:] , k_star[-Nx:N:2,:]  

		X_kbD = np.concatenate([xb0, xb1, xb2, xb3], axis = 0)
		Y_kbD = np.concatenate([kb0, kb1, kb2, kb3], axis = 0)
	elif if_BC_K == 0:
		X_kbD = np.empty((0,2),dtype=float) 
		Y_kbD = np.empty((0,1),dtype=float) 	

	# Training data ensemble
	Xk_train = np.concatenate([X_k, X_kbD], axis = 0) 
	Yk_train = np.concatenate([Y_k, Y_kbD], axis = 0)  

	if N_h != 0:
		idx_h = sel_measurement(seed_num,type_mea,N,N_h)
		X_h = X_star[idx_h,:]               
		Y_h = h_star[idx_h]           
	else:
		X_h = np.empty((0,2),dtype=float)
		Y_h = np.empty((0,1),dtype=float)
	Xh_train = X_h
	Yh_train = Y_h

	if if_plot_pts2D == 1:
		path_fig_plt = path_fig + 'Xk_'
		title_lstr  = 'Measurement K'
		sub_plt_pts2D(X_k,savefig=path_fig_plt, visual=if_vis, title = title_lstr)
		print("something")
		# wait = input("PRESS ENTER TO CONTINUE.")

		path_fig_plt = path_fig + 'Xh_'
		title_lstr  = 'Measurement h'
		sub_plt_pts2D(X_h,savefig=path_fig_plt, visual=if_vis, title = title_lstr)
		print("something")
		# wait = input("PRESS ENTER TO CONTINUE.")

		path_fig_plt = path_fig + 'Xc_'
		title_lstr  = 'Measurement C'
		sub_plt_pts2D(X_c,savefig=path_fig_plt, visual=if_vis, title = title_lstr)
		print("something")
		# wait = input("PRESS ENTER TO CONTINUE.")
	
	'''
	Boundary condtions for Darcy
	# Added for i-CAD at 4/14/2020
	''' 
	if if_BC_h == 0:
		X_hbD  = np.empty((0,2),dtype=float) 
		Y_hbD  = np.empty((0,1),dtype=float) 
		X_hbN1 = np.empty((0,2),dtype=float) 
		Y_hbN1 = np.empty((0,1),dtype=float) 		
		X_hbN2 = np.empty((0,2),dtype=float) 
		Y_hbN2 = np.empty((0,1),dtype=float)
	elif if_BC_h == 1: # Use Dirichlet and Neuman
		# Dirichlet boundaries
		xb2, hb2 = X_star[Nx-1:N:Nx,:], h_star[Nx-1:N:Nx,:]
		X_hbD = xb2
		Y_hbD = hb2
		
		# Neumann boundaries
		np.random.seed(seed_mea)
		xb1 = np.zeros((Nx,2))
		xb1[:,0:1] = lb[0] + (ub[0]-lb[0])*lhs(1, Nx)
		
		xb3 = np.zeros((Nx,2))
		xb3[:,1:2] = ub[1]  # y-coordinate
		xb3[:,0:1] = xb1[:,0:1]  # x-coordinate

		X_hbN2 = np.concatenate([xb1, xb3], axis = 0)   # approximating value
		Y_hbN2 = np.zeros((X_hbN2.shape[0], 1))   
			
		# Inhomogeneous Neumann boundaries 
		xb0 = np.zeros((Ny,2))
		xb0[:,1:2] = lb[1] + (ub[1]-lb[1])*lhs(1, Ny)
		
		X_hbN1 = xb0
		Y_hbN1 = np.ones((X_hbN1.shape[0],1))*coe_q


	'''Collocation points for Darcy'''
	if flag_sel_col == 31 or flag_sel_col == 32:
		x_nor, _ = sel_colpoint(seed_num,22,N,N_f,N,2,lb=None,ub=None)
	else:
		x_nor, _ = sel_colpoint(seed_num,flag_sel_col,N,N_f,N,2,lb=None,ub=None)
	Xf_col = lb + (ub-lb) * x_nor
	Yf_col = np.zeros((N_f, 1))
	
	# ====================================================================
	## Analysis
	# ====================================================================
	'''
	Randomize neural networks
	'''
	# Randomize neural-nets
	if N_s > 1:
		np.random.seed(seed_num) # reset        
		tf_seed_set = np.random.randint(0,2000,N_s)
		print('rand NNs seeds {}'.format(tf_seed_set),file=f_rec)

	for i_loop in range(0,N_s):
		if N_s == 1:
			rand_seed_tf_i = seed_num
		elif N_s > 1:
			rand_seed_tf_i = tf_seed_set[i_loop]

		path_tf = path_f+'tf_model'+'_r'+str(seed_num)+'_s'+str(i_loop)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_c'+str(N_c)+'_fc'+str(N_fc)+'/'
		if not os.path.exists(path_tf):
			os.makedirs(path_tf)

		print("\n",file=f_rec)
		print("seed_tf: {}".format(rand_seed_tf_i),file=f_rec)

		'''
		Multiphysics Model-Data Integration Neural Netowrk (MsDINN)
		'''
		layers_k = sub_NN_type(type_NNk,pro_dim-1)
		layers_h = sub_NN_type(type_NNh,pro_dim-1)
		layers_c = sub_NN_type(type_NNc,pro_dim)

		model = MDINN_CAD(Xk_train, Yk_train, Xh_train, Yh_train, Xf_col, Yf_col,
							Zm_train, Um_train, Zm_f_train, 
							Zm_cbN1_train, Um_cbN1_train, Zm_cbN2_train, Um_cbN2_train,
							X_hbD, Y_hbD, X_hbN1, Y_hbN1, X_hbN2, Y_hbN2, if_BC_h,
							layers_h, layers_k, layers_c, lb, ub, t_min, t_max,
							coe_phi,coe_Dw,coe_tau,coe_aL,coe_aT,
							f_rec,path_tf,f1_loss,f1_loss2,f2_weight,f2_bias,rand_seed_tf_i,
							para_k,para_h,para_kh,para_c,para_khc,
							type_op,learn_rate,flag_solv,conv_opt,type_reg,coe_reg,
							flag_pro,flag_lsty,flag_lss,
							type_op_pre,learn_rate_pre,flag_solv_pre,conv_opt_pre,pro_params,
							if_plot_rl2,ref_xm = X_star,ref_k = k_star, ref_h = h_star, ref_c = None)
				
		'''Train Neural Network'''
		num_print_opt = 200
		[if_output_k, if_output_h, if_output_c, if_output_f, if_output_fc] = [0, 0, 0, 0, 0] # Pre-defined
		start_time = time.time()
		
		if flag_pro == 'i-CAD':
			if pro_params['model_load'] == 'True':
				model.restore()
				if type_op_pre == 0:
					# ----------
					if_output_k, if_output_h, if_output_f = 1, 1, 1
					if_output_c, if_output_fc = 1, 1
					if_output_v = 1
					# ----------				
			else:
				if type_op_pre == 0:
					print ("Start Training MsDINN-CAD") # Conservation-Advection-Dispesion Equations
					start_time_AD = time.time()
					model.train_CAD(type_op,batchs,num_epochs,flag_solv,num_print_opt,f_rec)
					print ("End of learning process of CAD net")
					elapsed = time.time() - start_time_AD
					print('Training time for CAD: %.4f' % (elapsed))
					print('Training time for CAD: {0: .4f}'.format(elapsed),file=f_rec)
					print('----------------------------------',file=f_rec)
					# ----------
					if_output_k, if_output_h, if_output_f = 1, 1, 1
					if_output_c, if_output_fc = 1, 1
					if_output_v = 1			
					# ----------
				else:
					print ("Start Pretraining...")
					print ("End of learning process of Darcy Net")
					raise NotImplementedError				

			'''Output loss and errors'''
			elapsed = time.time() - start_time
			print('Total Training time: %.4f' % (elapsed))
			print('Total Training time: {0: .4f}'.format(elapsed),file=f_rec)
			model.out_loss_components()

		'''Relative Error'''
		if if_output_k == 1:
			k_pred = model.predict_k(X_star)
			error_k, error_k_new, error_k_inf = error_relative(k_pred,k_star)
			print('Error k: {0: e} | Error k-mean: {1: e} | Error k-inf: {2: e}'.format(error_k,error_k_new,error_k_inf))
			print('Error k: {0: e} | Error k-mean: {1: e} | Error k-inf: {2: e}'.format(error_k,error_k_new,error_k_inf),file=f_rec)

			'''Record prediction'''
			if if_rec_predic == 1:
				f_k_pred  = open(path_tf+'record_k_pred.out', 'a+')
				mat_k = np.matrix(k_pred)
				for line in mat_k:
					np.savetxt(f_k_pred, line, fmt='%.8f')
				f_k_pred.close()		

		if if_output_h == 1:
			h_pred = model.predict_h(X_star)
			error_h, error_h_new, error_h_inf = error_relative(h_pred,h_star)
			print('Error h: {0: e} | Error h-mean: {1: e} | Error h-inf: {2: e}'.format(error_h,error_h_new,error_h_inf))
			print('Error h: {0: e} | Error h-mean: {1: e} | Error h-inf: {2: e}'.format(error_h,error_h_new,error_h_inf),file=f_rec)

			'''Record prediction'''
			if if_rec_predic == 1:
				f_h_pred  = open(path_tf+'record_h_pred.out', 'a+')
				mat_h = np.matrix(h_pred)
				for line in mat_h:
					np.savetxt(f_h_pred, line, fmt='%.8f')
				f_h_pred.close()

		######################################################################
		############################# Plotting ###############################
		######################################################################  
		'''Plot 2D'''
		if IF_plot == 1:			
			if if_output_k == 1:
				path_fig_save = path_fig+'map_k_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_pred'
				title = '$K(x_1,x_2)$'
				cmin, cmax = np.min(k_star), np.max(k_star)
				sub_plt_surf2D_wpt(lb,ub,X_star,k_pred,output=path_fig_save, visual=None,plt_eps=None, points=None, cmin=cmin, cmax=cmax, title=title)

				path_fig_save = path_fig+'map_k_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_errors'
				title = 'Absolute error: $K$'
				points = X_k
				sub_plt_surf2D_wpt(lb,ub,X_star,np.abs(k_star-k_pred),output=path_fig_save, visual=None,plt_eps=None, points=points, cmin=None, cmax=None, title=title)

			if if_output_h == 1:
				path_fig_save = path_fig+'map_h_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_pred'
				title = '$h(x_1,x_2)$'
				cmin, cmax = np.min(h_star), np.max(h_star)
				sub_plt_surf2D_wpt(lb,ub,X_star,h_pred,output=path_fig_save, visual=None,plt_eps=None, points=None, cmin=cmin, cmax=cmax, title=title)

				path_fig_save = path_fig+'map_h_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_errors'
				title = 'Absolute error: $h$'
				points = X_h
				sub_plt_surf2D_wpt(lb,ub,X_star,np.abs(h_star-h_pred),output=path_fig_save, visual=None,plt_eps=None, points=points, cmin=None, cmax=None, title=title)

			if if_output_v == 1:
				v1_pred, v2_pred = model.predict_v(X_star)
				path_fig_save = path_fig+'map_v1_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_pred'
				title = '$v_1(x_1,x_2)$'
				cmin, cmax = None, None
				sub_plt_surf2D_wpt(lb,ub,X_star,v1_pred,output=path_fig_save, visual=None,plt_eps=None, points=None, cmin=cmin, cmax=cmax, title=title)

				path_fig_save = path_fig+'map_v2_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_pred'
				title = '$v_2(x_1,x_2)$'
				cmin, cmax = None, None
				sub_plt_surf2D_wpt(lb,ub,X_star,v2_pred,output=path_fig_save, visual=None,plt_eps=None, points=None, cmin=cmin, cmax=cmax, title=title)

		'''Reference Solution, error, 2D for C(x)'''

		if flag_t_test == 'None':
			temp_tm = t_test
			nt_plot = temp_tm.shape[0]
		else:
			temp_tm = mul_t_test
			nt_plot = temp_tm.shape[0]
		
		for II_t in range(nt_plot):
			I_time = temp_tm[II_t]
			if_ref_exist = if_ref_existM[II_t]

			t_lstr = str(round(I_time * 60,1)) 

			t_lstr_file = '{0:.1f}'.format(I_time * 60)
			t_lstr_file = t_lstr_file.rjust(5,'0')
			
			print('Time: {} min'.format(t_lstr))
			print('Time: {} min'.format(t_lstr),file=f_rec)
			
			if if_ref_exist == 1: # Plot ref. C(x)
				t_ref_ls = t_ref_lstr[II_t]
				
				# dataset_c = np.loadtxt(path_data+'raw_data/plot_matlab_sin_'+t_ref_ls+'min.dat', dtype=float) # Load solute concentration
				# c_ref    = dataset_c[:,4:5]   # Concentration
				
				if flag_data == 12:
					dataset_c = np.loadtxt(path_data+'raw_data/plot_matlab_sin_'+t_ref_ls+'min.dat', dtype=float) # Load solute concentration
					c_ref    = dataset_c[:,4:5]   # Concentration at initial time
				elif flag_data == 41:
					dataset_c = np.loadtxt(path_data+'raw_data/plot_matlab_smooth_field_02_normal'+t_ref_ls+'min.dat', dtype=float) # Load solute concentration
					c_ref    = dataset_c[:,6:7]   # Concentration
				elif flag_data == 42:
					dataset_c = np.loadtxt(path_data+'raw_data/plot_matlab_smooth_field_05_normal'+t_ref_ls+'min.dat', dtype=float) # Load solute concentration
					c_ref    = dataset_c[:,6:7]   # Concentration
				else:
					raise NotImplementedError

				if if_plot_C_sol == 1:				
					# dataset_c = np.loadtxt(path_data+'raw_data/plot_matlab_sin_'+t_lstr+'min.dat', dtype=float) # Load solute concentration
					# Cmt   = dataset_c[:,4:5]   # Concentration
					sub_plt_surf2D(lb,ub,X_star,c_ref,I_time*60,savefig=path_fig,visual=None,plt_eps=None)

			if if_output_c == 1: # Plot Pred. C(x)
				c_pred = model.predict_c(X_star,I_time)
			
				if if_ref_exist == 1:
					error_c, error_c_new, error_c_inf = error_relative(c_pred,c_ref)
					print('Error c: {0: e} | Error c-mean: {1: e} | Error c-inf: {2: e}'.format(error_c,error_c_new,error_c_inf))
					print('Error c: {0: e} | Error c-mean: {1: e} | Error c-inf: {2: e}'.format(error_c,error_c_new,error_c_inf),file=f_rec)
					# print("\n",file=f_rec)
				
				'''Record prediction'''
				if if_rec_predic == 1:
					f_c_pred  = open(path_tf+'record_c_pred'+'_t_'+t_lstr_file+'min.out', 'a+')
					mat_c = np.matrix(c_pred)
					for line in mat_c:
						np.savetxt(f_c_pred, line, fmt='%.8f')
					f_c_pred.close()
			
			if pro_params['if_plot2D_C'] == 1: # Plot 2D C and err_C
				if if_ref_exist == 1:
					c_star = c_ref
					if flag_data == 12:
						cmin, cmax = 0, 1
						# cmin_err,cmax_err = 0, 0.02
						cmin_err = None
						cmax_err = None
					elif flag_data == 42:
						cmin, cmax = 0, 1
						cmin_err = 0
						cmax_err = 0.04		# forward: default 0.03; backward: 0.04
						# cmax_err = 0.1
					else:
						cmin, cmax = np.min(c_star), np.max(c_star)
						cmin_err,cmax_err = 0, 0.1*cmax
				
				elif if_ref_exist == 0:
					cmin, cmax = 0, 1

				path_fig_save = path_fig+'map_c_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_pred'+'t_'+t_lstr_file+'min'
				title = '$C:$ ' + t_lstr +'min'
				if pro_params['IF_DataAssi'] == 'True': 
					sub_plt_surf2D_wpt(lb,ub,X_star,c_pred,output=path_fig_save, visual=None,plt_eps=None, points=X_c_mea, cmin=cmin, cmax=cmax, title=title)
				else:
					sub_plt_surf2D_wpt(lb,ub,X_star,c_pred,output=path_fig_save, visual=None,plt_eps=None, points=None, cmin=cmin, cmax=cmax, title=title)

				if if_ref_exist == 1:
					path_fig_save = path_fig+'map_c_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_errors'+'t_'+t_lstr_file+'min'
					title = 'Error $C$: ' + t_lstr +'min'

					if pro_params['IF_DataAssi'] == 'True': 
						sub_plt_surf2D_wpt(lb,ub,X_star,np.abs(c_star-c_pred),output=path_fig_save, visual=None,plt_eps=None, points=X_c_mea, cmin=cmin_err, cmax=cmax_err, title=title)
					else:
						sub_plt_surf2D_wpt(lb,ub,X_star,np.abs(c_star-c_pred),output=path_fig_save, visual=None,plt_eps=None, points=None, cmin=cmin_err, cmax=cmax_err, title=title)

					path_fig_save = path_fig+'map_c_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_ref'+'t_'+t_lstr_file+'min'
					title = 'Reference $C$: ' + t_lstr +'min'
					sub_plt_surf2D_wpt(lb,ub,X_star,np.abs(c_star),output=path_fig_save, visual=None,plt_eps=None, points=None, cmin=cmin, cmax=cmax, title=title)

		f_rec.flush()

		# ====================================================================
		## Post-Analysis
		# ====================================================================
		'''Record loss'''
		if if_rec_loss == 1:
			f_rloss  = open(path_tf+'record_rloss.out', 'a+')
			mat_rloss = np.matrix(model.rloss)
			for line in mat_rloss:
				np.savetxt(f_rloss, line, fmt='%.8f')
			f_rloss.close()

			if if_output_c == 1:
				f_rloss  = open(path_tf+'record_rloss_c.out', 'a+')
				mat_rloss = np.matrix(model.rloss_c)
				for line in mat_rloss:
					np.savetxt(f_rloss, line, fmt='%.8f')
				f_rloss.close()

		'''Plot loss'''
		if flag_pro == 'i-CAD':
			'''CAD'''
			type_op_temp, batchs_temp, num_epochs_temp = type_op, batchs, num_epochs

			'''loss output every num_print_opt'''
			if if_output_k == 1:
				path_fig_save = path_tf +'loss_k_'+str(type_op_temp)+'_'+str(batchs_temp)+'_'+str(num_epochs_temp)
				lstr_ylabel = '$L_K$'
				sub_plt_cuvlog2(model.rloss_k,num_print_opt,ylabelx=lstr_ylabel,savefig=path_fig_save,visual=None,plt_eps=None)

			if if_output_h == 1:
				path_fig_save = path_tf +'loss_h_'+str(type_op_temp)+'_'+str(batchs_temp)+'_'+str(num_epochs_temp)
				lstr_ylabel = '$L_h$'
				sub_plt_cuvlog2(model.rloss_h,num_print_opt,ylabelx=lstr_ylabel,savefig=path_fig_save,visual=None,plt_eps=None)

			if if_output_f == 1:
				path_fig_save = path_tf +'loss_f_'+str(type_op_temp)+'_'+str(batchs_temp)+'_'+str(num_epochs_temp)
				lstr_ylabel = '$L_f$'
				sub_plt_cuvlog2(model.rloss_pde_f,num_print_opt,ylabelx=lstr_ylabel,savefig=path_fig_save,visual=None,plt_eps=None)


			if type_op_temp == 1: # L-BFGS
				rloss_all = model.rloss
				path_fig_save = path_tf +'loss'+str(type_op_temp)+'_'+str(batchs_temp)+'_'+str(num_epochs_temp)
				lstr_ylabel = 'Loss'
				sub_plt_cuvlog(rloss_all,ylabelx=lstr_ylabel,savefig=path_fig_save,visual=None,plt_eps=None)		

				if if_output_c == 1:
					path_fig_save = path_tf +'loss_c_'+str(type_op_temp)+'_'+str(batchs_temp)+'_'+str(num_epochs_temp)
					lstr_ylabel = '$L_C$'
					sub_plt_cuvlog(model.rloss_c,ylabelx=lstr_ylabel,savefig=path_fig_save,visual=None,plt_eps=None)
			else:
				if batchs_temp == 0:
					rloss_all = model.rloss
					path_fig_save = path_tf +'loss'+str(type_op_temp)+'_'+str(batchs_temp)+'_'+str(num_epochs_temp)
					lstr_ylabel = 'Loss'
					sub_plt_cuvlog(rloss_all,ylabelx=lstr_ylabel,savefig=path_fig_save,visual=None,plt_eps=None)
				else:
					rloss_all = model.rloss_batch
					path_fig_save = path_tf +'loss_bch_'+str(type_op_temp)+'_'+str(batchs_temp)+'_'+str(num_epochs_temp)
					lstr_ylabel = 'Batch Loss'
					sub_plt_cuvlog(rloss_all,ylabelx=lstr_ylabel,savefig=path_fig_save,visual=None,plt_eps=None)

					path_fig_save = path_tf +'loss_'+str(type_op_temp)+'_'+str(batchs_temp)+'_'+str(num_epochs_temp)
					lstr_ylabel = 'Loss'
					sub_plt_cuvlog2(model.rloss,num_print_opt,ylabelx=lstr_ylabel,savefig=path_fig_save,visual=None,plt_eps=None)
	
				if if_output_c == 1:
					path_fig_save = path_tf +'loss_c_'+str(type_op_temp)+'_'+str(batchs_temp)+'_'+str(num_epochs_temp)
					lstr_ylabel = '$L_C$'
					sub_plt_cuvlog2(model.rloss_c,num_print_opt,ylabelx=lstr_ylabel,savefig=path_fig_save,visual=None,plt_eps=None)

		plt.close('all')

		#completely reset tensorflow
		tf.reset_default_graph()
		
	print("\n",file=f_rec)
	f_rec.close()
	f1.close()
	f1_loss.close()
	f1_loss2.close()
	f2_weight.close()
	f2_bias.close()