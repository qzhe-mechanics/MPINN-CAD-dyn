"""
@author     : Qizhi He @ PNNL (qizhi.he@pnnl.gov)
Decription  : MsDINN for Advection-Dispersion Equation
---Note---
* 'i-CAD': Add functionalities of inverse CAD; Add BC conditions for Darcy
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from utilities_dnn import tf_session

# Normalization
def sub_normalization(X,Xl,Xr,scale_coe,flag):
	if flag == 'scale':
	# Mapped to [-scale_coe,scale_coe]
		len = Xr - Xl
		X_nor = 2.0 * scale_coe * (X - Xl)/len - scale_coe
	elif flag == 'shift':
		X_nor = X - 0.5
	return X_nor

def sub_batch(num_all, batch, type):
	if type == 21 or type == 22 or type == 23 or type == 24:
		if num_all < batch:
			num_batch = num_all
		else:
			num_batch = batch
	elif type == 2:
		if num_all < batch * 16:
			num_batch = 16
		else:
			num_batch = int(num_all/batchs)
	return num_batch

class model_MsDINN_CAD_dyn:
	def __init__(self, X_k, Y_k, X_h, Y_h, X_f, Y_f,
				 Zm, Um, Zm_f,
				 Z_cbN1, U_cbN1, Z_cbN2, U_cbN2,
				 X_hbD, Y_hbD, X_hbN1, Y_hbN1, X_hbN2, Y_hbN2, if_BC_h,
				 layers_h, layers_k, layers_c, lb, ub, t_min, t_max,
				 coe_phi,coe_Dw,coe_tau,coe_aL,coe_aT,
				 f,path_tf,f1_loss,f1_loss2,f2_weight,f2_bias,rand_seed_tf,
				 para_k,para_h,para_kh,para_c,para_khc,
				 type_op,learn_rate,flag_solv,conv_opt,type_reg,coe_reg,
				 flag_pro,flag_lsty,flag_lss,
				 type_op_pre,learn_rate_pre,flag_solv_pre,conv_opt_pre,pro_params,
				 if_plot_rl2,ref_xm = None,ref_k = None, ref_h = None, ref_c = None):

		'''Input Parameters'''
		self.pro_params = pro_params
		
		## Physical Parameter
		self.flag_pro  = flag_pro
		self.if_BC_h = if_BC_h

		self.phi = coe_phi
		self.Dw  = coe_Dw
		self.tau = coe_tau
		self.aL  = coe_aL
		self.aT  = coe_aT

		self.lb = lb
		self.ub = ub
		self.t_min = t_min
		self.t_max = t_max

		## DNN model
		self.conv_opt  = conv_opt
		self.flag_lsty = flag_lsty
		self.flag_lss  = flag_lss
		self.conv_opt_pre = conv_opt_pre

		self.type_reg = type_reg 

		self.para_k = para_k
		self.para_h = para_h
		self.para_kh = para_kh
		self.para_c = para_c
		self.para_khc = para_khc

		## Files
		self.f  = f  
		self.f1_loss = f1_loss
		self.f1_loss2 = f1_loss2
		self.f2_weight = f2_weight
		self.f2_bias = f2_bias
		self.path_tf = path_tf

		#########################################################################
		## Scaling & Normalization
		#########################################################################
		'''Normalization for NNs Output'''
		## Maximum values
		coe_nor_k    = max(Y_k.min(), Y_k.max(), key=abs)
		coe_nor_h    = max(Y_h.min(), Y_h.max(), key=abs)
		coe_nor_c    = max(Um.min(), Um.max(), key=abs)

		self.coe_nor_k = coe_nor_k
		self.coe_nor_h = coe_nor_h
		self.coe_nor_c = coe_nor_c

		self.coe_nor_k2 = 1/coe_nor_k**2
		self.coe_nor_h2 = 1/coe_nor_h**2
		self.coe_nor_c2 = 1/coe_nor_c**2
		
		print('k nor: {0:e}, h nor {1:e}, c nor {2:e}'.format(coe_nor_k,coe_nor_h,coe_nor_c),file=f)

		## Standard Deviation
		# Yl_k, Yr_k = Y_k.min(0), Y_k.max(0)
		Yk_std     = np.std(Y_k)
		if Yk_std <= 1e-2:
			print('Deviation of k {} is too small, reset to 1'.format(Yk_std))
			print('Deviation of k {} is too small, reset to 1'.format(Yk_std),file=f)
			Yk_std = 1.0
			
		Yh_std     = np.std(Y_h)
		if Yh_std <= 1e-2:
			print('Deviation of h {} is too small, reset to 1'.format(Yh_std))
			print('Deviation of h {} is too small, reset to 1'.format(Yh_std),file=f)
			Yh_std = 1.0      

		Yc_std     = np.std(Um)
		if Yc_std <= 1e-2:
			print('Deviation of C {} is too small, reset to 1'.format(Yc_std))
			print('Deviation of C {} is too small, reset to 1'.format(Yc_std),file=f)
			Yc_std = 1.0           
		
		self.Yk_std = Yk_std  
		self.Yh_std = Yh_std
		self.Yc_std = Yc_std

		self.coe_std_k2 = 1/Yk_std**2
		self.coe_std_h2 = 1/Yh_std**2
		self.coe_std_c2 = 1/Yc_std**2

		print('k std: {0:e}, h: std {1:e}, c: std {2:e}'.format(Yk_std,Yh_std,Yc_std),file=f)
		'''------------------------------------------------------------------'''

		'''Normalization for NNs Inputs'''
		self.scale_coe = 0.5                     # a positive number denote the distance from center to one-end
		
		## Options
		self.scale_X   = 2 * self.scale_coe / (ub-lb)
		# self.scale_X = np.array([1.0, 1.0])
		self.flag_nor = 'scale'

		# ** Coordinates of K and h
		X_k    = sub_normalization(X_k,lb,ub,self.scale_coe,self.flag_nor)
		X_h    = sub_normalization(X_h,lb,ub,self.scale_coe,self.flag_nor)   
		X_f    = sub_normalization(X_f,lb,ub,self.scale_coe,self.flag_nor)

		self.if_BC_h = if_BC_h
		if if_BC_h == 1:
			X_hbD = sub_normalization(X_hbD,lb,ub,self.scale_coe,self.flag_nor)
			X_hbN1 = sub_normalization(X_hbN1,lb,ub,self.scale_coe,self.flag_nor)
			X_hbN2 = sub_normalization(X_hbN2,lb,ub,self.scale_coe,self.flag_nor)
		elif if_BC_h == 2:
			X_hbD = sub_normalization(X_hbD,lb,ub,self.scale_coe,self.flag_nor)
		
		# ** Coordinates of C
		lbt = np.hstack((lb, t_min))
		ubt = np.hstack((ub, t_max))
		self.scale_Xt   = 2 * self.scale_coe / (ubt-lbt)
		Zm = sub_normalization(Zm,lbt,ubt,self.scale_coe,self.flag_nor)
		# Zm[:,0:2] = sub_normalization(Zm[:,0:2],lb,ub,self.scale_coe,self.flag_nor)
		# Zm[:,2:3] = sub_normalization(Zm[:,2:3],t_min,t_max,self.scale_coe,self.flag_nor)
		Zm_f = sub_normalization(Zm_f,lbt,ubt,self.scale_coe,self.flag_nor)
		Z_cbN1 = sub_normalization(Z_cbN1,lbt,ubt,self.scale_coe,self.flag_nor)
		Z_cbN2 = sub_normalization(Z_cbN2,lbt,ubt,self.scale_coe,self.flag_nor)    
		'''------------------------------------------------------------------'''

		## Input for training (normalized)
		## K(x) and h(x)
		[self.x1_k, self.x2_k, self.Y_k] = [X_k[:,0:1], X_k[:,1:2], Y_k]
		[self.x1_h, self.x2_h, self.Y_h] = [X_h[:,0:1], X_h[:,1:2], Y_h] 
		[self.x1_f, self.x2_f, self.Y_f] = [X_f[:,0:1], X_f[:,1:2], Y_f] 
		
		## Time-dependent C(x,t)
		[self.x1_c, self.x2_c, self.t_c, self.Y_c] = [Zm[:,0:1], Zm[:,1:2], Zm[:,2:3], Um]
		[self.x1_fc, self.x2_fc, self.t_fc] = [Zm_f[:,0:1], Zm_f[:,1:2], Zm_f[:,2:3]] 
				
		'''o for
		Boundary condtions for Darcy (neglected now: added) and CAD
		''' 
		[self.x1_cbN1, self.x2_cbN1, self.t_cbN1, self.Y_cbN1] = [Z_cbN1[:,0:1], Z_cbN1[:,1:2], Z_cbN1[:,2:3], U_cbN1]
		[self.x1_cbN2, self.x2_cbN2, self.t_cbN2, self.Y_cbN2] = [Z_cbN2[:,0:1], Z_cbN2[:,1:2], Z_cbN2[:,2:3], U_cbN2]

		if if_BC_h == 1:
			[self.x1_hbD, self.x2_hbD, self.Y_hbD] = [X_hbD[:,0:1], X_hbD[:,1:2], Y_hbD]
			[self.x1_hbN1, self.x2_hbN1, self.Y_hbN1] = [X_hbN1[:,0:1], X_hbN1[:,1:2], Y_hbN1] # Special Neumann
			[self.x1_hbN2, self.x2_hbN2, self.Y_hbN2] = [X_hbN2[:,0:1], X_hbN2[:,1:2], Y_hbN2]
		elif if_BC_h == 2:
			[self.x1_hbD, self.x2_hbD, self.Y_hbD] = [X_hbD[:,0:1], X_hbD[:,1:2], Y_hbD]

		#########################################################################
		### Initialize network weights and loss  
		#########################################################################
		self.layers_k = layers_k
		self.layers_h = layers_h
		self.layers_c = layers_c
		
		''' Define placeholders and computational graph'''
		## placeholders
		[self.x1_k_tf, self.x2_k_tf, self.Yk_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
		[self.x1_h_tf, self.x2_h_tf, self.Yh_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
		[self.x1_f_tf, self.x2_f_tf, self.Yf_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
		
		[self.x1_c_tf, self.x2_c_tf, self.t_c_tf, self.Yc_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
		[self.x1_fc_tf, self.x2_fc_tf, self.t_fc_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]

		[self.x1_cbN1_tf, self.x2_cbN1_tf, self.t_cbN1_tf, self.YcbN1_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
		[self.x1_cbN2_tf, self.x2_cbN2_tf, self.t_cbN2_tf, self.YcbN2_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
		
		[self.x1_v_tf, self.x2_v_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(2)]
		
		if if_BC_h == 1:
			[self.x1_hbD_tf, self.x2_hbD_tf, self.YhbD_tf]    = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
			[self.x1_hbN1_tf, self.x2_hbN1_tf, self.YhbN1_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
			[self.x1_hbN2_tf, self.x2_hbN2_tf, self.YhbN2_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
		elif if_BC_h == 2:
			[self.x1_hbD_tf, self.x2_hbD_tf, self.YhbD_tf]    = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]

		'''Xavier Initialization'''
		tf.set_random_seed(rand_seed_tf)
		self.weights_k, self.biases_k = self.initialize_NN_regu_k(layers_k)
		tf.set_random_seed(rand_seed_tf)
		self.weights_h, self.biases_h = self.initialize_NN_regu_h(layers_h)
		tf.set_random_seed(rand_seed_tf)
		self.weights_c, self.biases_c = self.initialize_NN_regu_c(layers_c)
		
		'''Regularization'''
		if type_reg == 0:
			self.reg_term = 0.0
		elif type_reg == 1:
			regularizer = tf.contrib.layers.l1_regularizer(coe_reg)
			self.reg_term = tf.contrib.layers.apply_regularization(regularizer,weights_list=None)
		elif type_reg == 2:
			regularizer = tf.contrib.layers.l2_regularizer(coe_reg)
			self.reg_term = tf.contrib.layers.apply_regularization(regularizer,weights_list=None)
		
		'''Define neural network forward pass'''
		self.k_pred = self.net_k(self.x1_k_tf, self.x2_k_tf)
		self.h_pred = self.net_h(self.x1_h_tf, self.x2_h_tf)  

		self.v1_pred = self.net_v1(self.x1_v_tf, self.x2_v_tf)
		self.v2_pred = self.net_v2(self.x1_v_tf, self.x2_v_tf)

		self.f_pred  = self.net_Darcy(self.x1_f_tf, self.x2_f_tf)

		self.c_pred = self.net_c_dyn(self.x1_c_tf, self.x2_c_tf,self.t_c_tf)
		self.cbN1_pred = self.net_cbN1(self.x1_cbN1_tf, self.x2_cbN1_tf, self.t_cbN1_tf)
		self.cbN2_pred = self.net_cbN2(self.x1_cbN2_tf, self.x2_cbN2_tf, self.t_cbN2_tf)
		self.fc_pred  = self.net_AD_dyn(self.x1_fc_tf, self.x2_fc_tf, self.t_fc_tf)

		if if_BC_h == 1:
			self.hbD_pred   = self.net_hbD(self.x1_hbD_tf, self.x2_hbD_tf) 
			self.hbN1_pred  = self.net_hbN1(self.x1_hbN1_tf, self.x2_hbN1_tf)
			self.hbN2_pred  = self.net_hbN2(self.x1_hbN2_tf, self.x2_hbN2_tf)
		elif if_BC_h == 2:
			self.hbD_pred   = self.net_hbD(self.x1_hbD_tf, self.x2_hbD_tf) 

		'''Define Loss'''
		self.loss_k, self.loss_h, self.loss_pde_f, self.loss_pde_hbD, self.loss_pde_hbN1, self.loss_pde_hbN2 = self.loss_func_ele_darcy()
		
		self.loss_c, self.loss_pde_fc, self.loss_pde_cN1, self.loss_pde_cN2 = self.loss_func_ele_AD()

		# ** Pretrain PINN-Darcy
		self.loss_khr = self.loss_function_khr()
		self.loss_hr = self.loss_function_hr()
		# ** Train PINN-CAD
		self.loss = self.loss_function() + self.reg_term
		
		'''Loss Record'''
		self.rloss = []
		self.rloss_khr = []
		self.rloss_hr = []

		self.rloss_k = []
		self.rloss_h = []
		self.rloss_c = []

		self.rloss_pde_f   = []
		self.rloss_pde_fc  = []
		self.rloss_pde_cN1 = []
		self.rloss_pde_cN2 = []
		
		self.rloss_batch   = []
		self.rloss_khr_batch = []
		self.rloss_k_batch = []
		self.rloss_h_batch = []
		self.rloss_c_batch = []
		self.rloss_pde_f_batch  = []
		self.rloss_pde_fc_batch  = []

		## Error to reference
		if if_plot_rl2 == 0:
			self.if_record_rl2 = 0
		elif if_plot_rl2 == 1:
			self.if_record_rl2 = 1
			self.ref_xm = ref_xm
			self.ref_k = ref_k
			self.ref_h = ref_h
			self.ref_c = ref_c
			
		self.rl2_k = []
		self.rl2_h = []
		self.rl2_c = []
	
		'''Define optimizer'''
		
		var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_k")
		var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_h")
		var_list3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_c")
		var_list_12 = var_list1 + var_list2
		var_opt     = var_list1 + var_list2 + var_list3

		# ====================================================================
		## Optimizer
		# ====================================================================
		'''Optimizer of Pre-trained Darcy''' # No pretraining for Data assimilation "i-CAD" Commented @ 12/22/2020
		# if self.flag_pro == 'f-CAD2' or self.flag_pro == 'b-CAD2':
		# 	if type_op_pre == 1:
		# 		raise NotImplementedError
		# 	elif type_op_pre == 2:
		# 		self.optimizer_pre_k = tf.train.AdamOptimizer(learning_rate=learn_rate_pre, beta1=0.9,
		# 						beta2=0.999, epsilon=1e-08).minimize(self.loss_k,var_list=var_list1)
		# 		self.optimizer_pre_hr = tf.train.AdamOptimizer(learning_rate=learn_rate_pre, beta1=0.9,
		# 					beta2=0.999, epsilon=1e-08).minimize(self.loss_hr,var_list=var_list2)	

				
		# 		if flag_solv_pre == 'BFGS' or flag_solv_pre == 'BFGS-f':								
		# 			self.optimizer_pre_k_BFGS = tf.contrib.opt.ScipyOptimizerInterface(self.loss_k,
		# 																	var_list=var_list1,
		# 																	method  = 'L-BFGS-B', 
		# 																	options = {'maxiter': 50000,
		# 																			'maxfun': 50000,
		# 																			'maxcor': 50,
		# 																			'maxls': 50,
		# 																			'ftol' : 1.0 * np.finfo(float).eps})
		# 			self.optimizer_pre_hr_BFGS = tf.contrib.opt.ScipyOptimizerInterface(self.loss_hr,
		# 																	var_list=var_list2,
		# 																	method  = 'L-BFGS-B', 
		# 																	options = {'maxiter': 50000,
		# 																			'maxfun': 50000,
		# 																			'maxcor': 50,
		# 																			'maxls': 50,
		# 																			'ftol' : 1.0 * np.finfo(float).eps})
		# else:
		# 	if type_op_pre == 1:
		# 		self.optimizer_pre = tf.contrib.opt.ScipyOptimizerInterface(self.loss_khr,
		# 																var_list=var_list_12,
		# 																method  = 'L-BFGS-B', 
		# 																options = {'maxiter': 50000,
		# 																		'maxfun': 50000,
		# 																		'maxcor': 50,
		# 																		'maxls': 50,
		# 																		'ftol' : 1.0 * np.finfo(float).eps})
		# 	elif type_op_pre == 2:
		# 		self.optimizer_pre = tf.train.AdamOptimizer(learning_rate=learn_rate_pre, beta1=0.9,
		# 						beta2=0.999, epsilon=1e-08).minimize(self.loss_khr,var_list=var_list_12)

		# 		if flag_solv_pre == 'BFGS' or flag_solv_pre == 'BFGS-f':
		# 			self.optimizer_pre_BFGS = tf.contrib.opt.ScipyOptimizerInterface(self.loss_khr,
		# 																	var_list=var_list_12,
		# 																	method  = 'L-BFGS-B', 
		# 																	options = {'maxiter': 50000,
		# 																			'maxfun': 50000,
		# 																			'maxcor': 50,
		# 																			'maxls': 50,
		# 																			'ftol' : 1.0 * np.finfo(float).eps})

		if self.flag_pro == 'i-CAD':
			'''Optimizer of CAD'''
			# Note: {Use same name of optimizer but different training variables}
			if type_op == 1:
				self.optimizer_CAD = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
																		var_list=var_opt,
																		method  = 'L-BFGS-B', 
																		options = {'maxiter': 50000,
																				'maxfun': 50000,
																				'maxcor': 50,
																				'maxls': 50,
																				'ftol' : 1.0 * np.finfo(float).eps})
			elif type_op == 2:
				self.optimizer_CAD = tf.train.AdamOptimizer(learning_rate=learn_rate, beta1=0.9,
							beta2=0.999, epsilon=1e-08).minimize(self.loss,var_list=var_opt)
				if flag_solv == 'BFGS' or flag_solv == 'BFGS-f':
					self.optimizer_CAD_BFGS = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
																			var_list=var_opt,
																			method  = 'L-BFGS-B', 
																			options = {'maxiter': 50000,
																					'maxfun': 50000,
																					'maxcor': 50,
																					'maxls': 50,
																					'ftol' : 1.0 * np.finfo(float).eps})			

		else:
			raise NotImplementedError
		
		'''Define Tensorflow session'''
		# self.sess = tf_session()
		config = tf.ConfigProto(allow_soft_placement=True,
								log_device_placement=False)
		config.gpu_options.force_gpu_compatible = True
		self.sess = tf.Session(config=config)
		init = tf.global_variables_initializer()

		# Define Saver
		self.save_file = path_tf + 'train_model' # Same as with .ckpt
		self.saver = tf.train.Saver()

		self.sess.run(init)

	#########################################################################
	### Model functions
	#########################################################################  
	'''Initialize network weights and biases using Xavier initialization'''
	def initialize_NN_regu_k(self, layers): 
		# Xavier initialization
		def xavier_init(size):
			in_dim = size[0]
			out_dim = size[1]
			xavier_stddev = np.sqrt(2./(in_dim + out_dim))
			return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)   
		
		with tf.variable_scope("var_k"):
			weights = []
			biases = []
			num_layers = len(layers)
			for l in range(0,num_layers-1):
				W = xavier_init(size=[layers[l], layers[l+1]])
				b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
				weights.append(W)
				biases.append(b) 
				'''add regularizer'''
				tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)  

			'''add regularizer'''
			tf.add_to_collection(tf.GraphKeys.WEIGHTS, W) 
		return weights, biases

	def initialize_NN_regu_h(self, layers): 
		# Xavier initialization
		def xavier_init(size):
			in_dim = size[0]
			out_dim = size[1]
			xavier_stddev = np.sqrt(2./(in_dim + out_dim))
			return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)   
		
		with tf.variable_scope("var_h"):
			weights = []
			biases = []
			num_layers = len(layers)
			for l in range(0,num_layers-1):
				W = xavier_init(size=[layers[l], layers[l+1]])
				b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
				weights.append(W)
				biases.append(b) 
				'''add regularizer'''
				tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)  

			'''add regularizer'''
			tf.add_to_collection(tf.GraphKeys.WEIGHTS, W) 
		return weights, biases

	def initialize_NN_regu_c(self, layers): 
		# Xavier initialization
		def xavier_init(size):
			in_dim = size[0]
			out_dim = size[1]
			xavier_stddev = np.sqrt(2./(in_dim + out_dim))
			return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)   
		
		with tf.variable_scope("var_c"):
			weights = []
			biases = []
			num_layers = len(layers)
			for l in range(0,num_layers-1):
				W = xavier_init(size=[layers[l], layers[l+1]])
				b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
				weights.append(W)
				biases.append(b) 
				'''add regularizer'''
				tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)  

			'''add regularizer'''
			tf.add_to_collection(tf.GraphKeys.WEIGHTS, W) 
		return weights, biases

	'''Forward Pass'''
	def forward_pass(self, H, layers, weights, biases):
		num_layers = len(layers)
		for l in range(0,num_layers-2):
			W = weights[l]
			b = biases[l]
			H = tf.tanh(tf.add(tf.matmul(H, W), b)) # Note: Tanh map to [-1,1], the last layer without activation..
		W = weights[-1]
		b = biases[-1]
		H = tf.add(tf.matmul(H, W), b)
		return H

	def forward_pass_other(self, H, layers, weights, biases):
		num_layers = len(layers)
		for l in range(0,num_layers-2):
			W = weights[l]
			b = biases[l]
			# H = tf.nn.elu(tf.add(tf.matmul(H, W), b))
			# H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
			H = tf.nn.swish(tf.add(tf.matmul(H, W), b))
		W = weights[-1]
		b = biases[-1]
		H = tf.add(tf.matmul(H, W), b)
		return H  

	def net_k(self, x1, x2):
		k = self.forward_pass(tf.concat([x1, x2], 1),  # tf.concat(,0 or 1) 0: row-wise; 1: column-wise
							  self.layers_k,
							  self.weights_k, 
							  self.biases_k)
		return k
	
	# Forward pass for h
	def net_h(self, x1, x2):
		h = self.forward_pass(tf.concat([x1, x2], 1),
							  self.layers_h,
							  self.weights_h, 
							  self.biases_h)
		return h

	def net_hbD(self, x1, x2):
		h = self.net_h(x1, x2)
		return h

	def net_hbN1(self, x1, x2):
		h = self.net_h(x1, x2)
		k = self.net_k(x1, x2)
		beta = self.scale_X
		g_1 = - k * tf.gradients(h, x1)[0] * beta[0]
		return g_1    

	def net_hbN2(self, x1, x2):
		h = self.net_h(x1, x2)
		k = self.net_k(x1, x2)
		beta = self.scale_X
		g_2 = - k * tf.gradients(h, x2)[0] * beta[1]
		return g_2

	# Forward pass for c
	def net_c_dyn(self, x1, x2, t):
		c = self.forward_pass(tf.concat([x1, x2, t], 1),
							  self.layers_c,
							  self.weights_c, 
							  self.biases_c)
		return c

	def net_v1(self, x1, x2):
		beta = self.scale_X
		k  = self.net_k(x1, x2)
		h  = self.net_h(x1, x2)
		v1 = -k*tf.gradients(h, x1)[0] * beta[0]
		return v1

	def net_v2(self, x1, x2):
		beta = self.scale_X
		k  = self.net_k(x1, x2)
		h  = self.net_h(x1, x2)
		v2 = -k*tf.gradients(h, x2)[0] * beta[1]
		return v2

	'''PDE-nets'''
	def net_Darcy(self, x1, x2):
		k = self.net_k(x1, x2)
		h = self.net_h(x1, x2)
		beta = self.scale_X

		h_x1 = tf.gradients(h, x1)[0] * beta[0]
		h_x2 = tf.gradients(h, x2)[0] * beta[1]
		f_1 = tf.gradients(k * h_x1, x1)[0] * beta[0]
		f_2 = tf.gradients(k * h_x2, x2)[0] * beta[1]
		f = f_1 + f_2
		return f

	def net_AD_dyn(self, x1, x2, t):
		Dw  = self.Dw
		tau = self.tau
		aL  = self.aL
		aT  = self.aT
		phi = self.phi
		# beta = self.scale_X
		beta = self.scale_Xt

		# k  = self.net_k(x1, x2)
		# h  = self.net_h(x1, x2)
		c  = self.net_c_dyn(x1, x2, t)

		v1  = self.net_v1(x1, x2)
		v2  = self.net_v2(x1, x2)

		dc_x1 = tf.gradients(c, x1)[0] * beta[0]
		dc_x2 = tf.gradients(c, x2)[0] * beta[1]
		dc_t  = tf.gradients(c, t)[0] * beta[2]
		
		## LHS (debug: phi)
		fc_L = phi * dc_t + v1*dc_x1 + v2*dc_x2
		
		## RHS
		v_l2  = tf.sqrt(v1**2+v2**2)
		coe_L = phi*Dw*tau + aL * v_l2
		coe_T = phi*Dw*tau + aT * v_l2  
		f_1 = tf.gradients(coe_L*dc_x1, x1)[0] * beta[0] 
		f_2 = tf.gradients(coe_T*dc_x2, x2)[0] * beta[1]
		fc_R = f_1 + f_2
		return fc_L - fc_R

	def net_cbN1(self, x1, x2, t):
		beta = self.scale_X
		c = self.net_c_dyn(x1, x2, t)
		g_1 = tf.gradients(c, x1)[0] * beta[0]
		return g_1
	
	def net_cbN2(self, x1, x2, t):
		beta = self.scale_X
		c = self.net_c_dyn(x1, x2, t)
		g_2 = tf.gradients(c, x2)[0] * beta[1]
		return g_2


	def restore(self):
		# Restore variables from disk.
		self.saver.restore(self.sess, self.save_file)
		print("Model restored.",file=self.f)

	# ====================================================================
	## Train CAD networks
	# ====================================================================
	def callback_CAD_fast(self,loss,loss_k,loss_h,loss_c):
		num_print_opt = self.num_print_opt
		it = self.index_opt
		
		self.rloss.append(loss)
				
		# self.rloss.append(loss)
		# self.rloss_c.append(loss_c)
		# self.rloss_pde_fc.append(loss_fc)
		# self.rloss_pde_cN1.append(loss_cN1)
		# self.rloss_pde_cN2.append(loss_cN2)

		if it % num_print_opt == 0:
			print('It={0:d} | Loss: {1:.3e} | Loss_k: {2: .3e} | Loss_h: {3:.3e} | Loss_c: {4:.3e}'.format(it,loss,loss_k,loss_h,loss_c))
			self.rloss_k.append(loss_k)
			self.rloss_h.append(loss_h)
			self.rloss_c.append(loss_c)
		self.index_opt += 1	
	
	def train_CAD(self,type_op,batchs,nIter,flag_solv,num_print_opt,f): 
		self.f = f
		self.num_print_opt = num_print_opt
		
		## CAD read measurement data
		if self.if_BC_h == 0:
			tf_dict_CAD = {
						self.x1_k_tf: self.x1_k, self.x2_k_tf: self.x2_k, self.Yk_tf: self.Y_k,
						self.x1_h_tf: self.x1_h, self.x2_h_tf: self.x2_h, self.Yh_tf: self.Y_h,
						self.x1_f_tf: self.x1_f, self.x2_f_tf: self.x2_f, self.Yf_tf: self.Y_f,		
						self.x1_c_tf: self.x1_c, self.x2_c_tf: self.x2_c, self.t_c_tf: self.t_c, self.Yc_tf: self.Y_c,
						self.x1_fc_tf: self.x1_fc, self.x2_fc_tf: self.x2_fc, self.t_fc_tf: self.t_fc,
						self.x1_cbN1_tf: self.x1_cbN1, self.x2_cbN1_tf: self.x2_cbN1, self.t_cbN1_tf: self.t_cbN1, self.YcbN1_tf: self.Y_cbN1,
						self.x1_cbN2_tf: self.x1_cbN2, self.x2_cbN2_tf: self.x2_cbN2, self.t_cbN2_tf: self.t_cbN2, self.YcbN2_tf: self.Y_cbN2}
		elif self.if_BC_h == 1:
			tf_dict_CAD = {
						self.x1_k_tf: self.x1_k, self.x2_k_tf: self.x2_k, self.Yk_tf: self.Y_k,
						self.x1_h_tf: self.x1_h, self.x2_h_tf: self.x2_h, self.Yh_tf: self.Y_h,
						self.x1_f_tf: self.x1_f, self.x2_f_tf: self.x2_f, self.Yf_tf: self.Y_f,
						self.x1_hbD_tf: self.x1_hbD, self.x2_hbD_tf: self.x2_hbD, self.YhbD_tf: self.Y_hbD,
						self.x1_hbN1_tf: self.x1_hbN1, self.x2_hbN1_tf: self.x2_hbN1, self.YhbN1_tf: self.Y_hbN1,
						self.x1_hbN2_tf: self.x1_hbN2, self.x2_hbN2_tf: self.x2_hbN2, self.YhbN2_tf: self.Y_hbN2,
						self.x1_c_tf: self.x1_c, self.x2_c_tf: self.x2_c, self.t_c_tf: self.t_c, self.Yc_tf: self.Y_c,
						self.x1_fc_tf: self.x1_fc, self.x2_fc_tf: self.x2_fc, self.t_fc_tf: self.t_fc,
						self.x1_cbN1_tf: self.x1_cbN1, self.x2_cbN1_tf: self.x2_cbN1, self.t_cbN1_tf: self.t_cbN1, self.YcbN1_tf: self.Y_cbN1,
						self.x1_cbN2_tf: self.x1_cbN2, self.x2_cbN2_tf: self.x2_cbN2, self.t_cbN2_tf: self.t_cbN2, self.YcbN2_tf: self.Y_cbN2}
		elif self.if_BC_h == 2:
			tf_dict_CAD = {
						self.x1_k_tf: self.x1_k, self.x2_k_tf: self.x2_k, self.Yk_tf: self.Y_k,
						self.x1_h_tf: self.x1_h, self.x2_h_tf: self.x2_h, self.Yh_tf: self.Y_h,
						self.x1_f_tf: self.x1_f, self.x2_f_tf: self.x2_f, self.Yf_tf: self.Y_f,
						self.x1_hbD_tf: self.x1_hbD, self.x2_hbD_tf: self.x2_hbD, self.YhbD_tf: self.Y_hbD,
						self.x1_c_tf: self.x1_c, self.x2_c_tf: self.x2_c, self.t_c_tf: self.t_c, self.Yc_tf: self.Y_c,
						self.x1_fc_tf: self.x1_fc, self.x2_fc_tf: self.x2_fc, self.t_fc_tf: self.t_fc,
						self.x1_cbN1_tf: self.x1_cbN1, self.x2_cbN1_tf: self.x2_cbN1, self.t_cbN1_tf: self.t_cbN1, self.YcbN1_tf: self.Y_cbN1,
						self.x1_cbN2_tf: self.x1_cbN2, self.x2_cbN2_tf: self.x2_cbN2, self.t_cbN2_tf: self.t_cbN2, self.YcbN2_tf: self.Y_cbN2}

		if type_op == 1:
			self.index_opt = 1       
			self.optimizer_CAD.minimize(self.sess, 
										feed_dict = tf_dict_CAD,         
										fetches = [self.loss, self.loss_k, self.loss_h, self.loss_c],
										loss_callback = self.callback_CAD_fast)

		elif type_op == 2:
			if batchs == 0: # Vanilla
				start_time = time.time()
				for it in range(nIter):  # If nIter = 0, not run this sess
					_, loss = self.sess.run([self.optimizer_CAD, self.loss], tf_dict_CAD)

					self.rloss.append(loss)

					if it % num_print_opt == 0:
						loss_k,loss_h,loss_c,loss_f,loss_fc = self.sess.run([self.loss_k,self.loss_h,
							self.loss_c,self.loss_pde_f,self.loss_pde_fc],feed_dict=tf_dict_CAD)
						loss_cN1,loss_cN2 = self.sess.run([self.loss_pde_cN1,self.loss_pde_cN2],feed_dict=tf_dict_CAD)
						elapsed = time.time() - start_time

						print('It: {0: d} | Time: {1: .2f} | Loss: {2: .3e} | Loss_k: {3: .3e} | Loss_h: {4: .3e} | Loss_c: {5: .3e}'.\
							format(it, elapsed, loss, loss_k, loss_h, loss_c)) 
						print('--- Loss_f: {0: .3e} | Loss_fc: {1: .3e} | Loss_cN1: {2: .3e} | Loss_cN2: {3: .3e}'.format(loss_f,loss_fc,loss_cN1, loss_cN2))

						self.rloss_k.append(loss_k)
						self.rloss_h.append(loss_h)
						self.rloss_c.append(loss_c)
						self.rloss_pde_f.append(loss_f)
						self.rloss_pde_fc.append(loss_fc)
						self.rloss_pde_cN1.append(loss_cN1)
						self.rloss_pde_cN2.append(loss_cN2)

						start_time = time.time()

					if loss < self.conv_opt and self.conv_opt != 0:
						break

			elif batchs != 0:
				start_time = time.time()
				
				# ** Select more data from C fileds
				batch_size_c = min(self.Y_c.shape[0], 4 * batchs)

				# ** Select data from K, h fields
				batch_size_k = min(self.x1_k.shape[0], 2 * batchs)
				batch_size_h = min(self.x1_h.shape[0], 2 * batchs)

				if self.if_BC_h == 1:
					batch_size_hbD   = min(self.x1_hbD.shape[0], 2 * batchs)
					batch_size_hbN1  = min(self.x1_hbN1.shape[0], 2 * batchs)
					batch_size_hbN2  = min(self.x1_hbN2.shape[0], 2 * batchs)
				elif self.if_BC_h == 2: 
					batch_size_hbD   = min(self.x1_hbD.shape[0], 2 * batchs)

				for it in range(nIter):
					# *** C fileds ***
					'''Select Data'''
					randidx_c = np.random.randint(int(self.Y_c.shape[0]), size=batch_size_c)
					self.x1_c_batch, self.x2_c_batch, self.t_c_batch, self.Yc_batch = self.x1_c[randidx_c], self.x2_c[randidx_c], self.t_c[randidx_c], self.Y_c[randidx_c]

					'''Select Collocation'''
					randidx_fc = np.random.randint(int(self.x1_fc.shape[0]), size=batchs)
					randidx_cbN1 = np.random.randint(int(self.x1_cbN1.shape[0]), size=batchs)
					randidx_cbN2 = np.random.randint(int(self.x1_cbN2.shape[0]), size=batchs)

					self.x1_fc_batch, self.x2_fc_batch, self.t_fc_batch = self.x1_fc[randidx_fc], self.x2_fc[randidx_fc], self.t_fc[randidx_fc]
					self.x1_cbN1_batch, self.x2_cbN1_batch, self.t_cbN1_batch, self.YcbN1_batch = self.x1_cbN1[randidx_cbN1,:], self.x2_cbN1[randidx_cbN1,:], self.t_cbN1[randidx_cbN1,:], self.Y_cbN1[randidx_cbN1,:] 
					self.x1_cbN2_batch, self.x2_cbN2_batch, self.t_cbN2_batch, self.YcbN2_batch = self.x1_cbN2[randidx_cbN2,:], self.x2_cbN2[randidx_cbN2,:], self.t_cbN2[randidx_cbN2,:], self.Y_cbN2[randidx_cbN2,:] 

					# *** K fileds ***
					'''Select Data'''  
					randidx_k = np.random.randint(int(self.x1_k.shape[0]), size=batch_size_k)
					randidx_h = np.random.randint(int(self.x1_h.shape[0]), size=batch_size_h)

					self.x1_k_batch, self.x2_k_batch, self.Yk_batch = self.x1_k[randidx_k], self.x2_k[randidx_k], self.Y_k[randidx_k,:]
					self.x1_h_batch, self.x2_h_batch, self.Yh_batch = self.x1_h[randidx_h], self.x2_h[randidx_h], self.Y_h[randidx_h,:]
					
					'''Select Collocation'''
					randidx_f = np.random.randint(int(self.x1_f.shape[0]), size=batchs)
					self.x1_f_batch, self.x2_f_batch, self.Yf_batch = self.x1_f[randidx_f], self.x2_f[randidx_f], self.Y_f[randidx_f,:]

					'''Select Darcy B.C. Collocation'''
					randidx_hbD = np.random.randint(int(self.x1_hbD.shape[0]), size=batch_size_hbD)
					randidx_hbN1 = np.random.randint(int(self.x1_hbN1.shape[0]), size=batch_size_hbN1)
					randidx_hbN2 = np.random.randint(int(self.x1_hbN2.shape[0]), size=batch_size_hbN2)

					self.x1_hbD_batch, self.x2_hbD_batch, self.Y_hbD_batch = self.x1_hbD[randidx_hbD,:], self.x2_hbD[randidx_hbD,:], self.Y_hbD[randidx_hbD,:] 
					self.x1_hbN1_batch, self.x2_hbN1_batch, self.Y_hbN1_batch = self.x1_hbN1[randidx_hbN1,:], self.x2_hbN1[randidx_hbN1,:], self.Y_hbN1[randidx_hbN1,:] 
					self.x1_hbN2_batch, self.x2_hbN2_batch, self.Y_hbN2_batch = self.x1_hbN2[randidx_hbN2,:], self.x2_hbN2[randidx_hbN2,:], self.Y_hbN2[randidx_hbN2,:] 

					# *** Batches ***
					# Debuging @ 2020.04.14 [Done]
					'''mini-batch'''
					tf_dict_batch = {
						self.x1_k_tf: self.x1_k_batch, self.x2_k_tf: self.x2_k_batch, self.Yk_tf: self.Yk_batch,
						self.x1_h_tf: self.x1_h_batch, self.x2_h_tf: self.x2_h_batch, self.Yh_tf: self.Yh_batch,
						self.x1_f_tf: self.x1_f_batch, self.x2_f_tf: self.x2_f_batch, self.Yf_tf: self.Yf_batch,
						self.x1_hbD_tf: self.x1_hbD_batch, self.x2_hbD_tf: self.x2_hbD_batch, self.YhbD_tf: self.Y_hbD_batch,
						self.x1_hbN1_tf: self.x1_hbN1_batch, self.x2_hbN1_tf: self.x2_hbN1_batch, self.YhbN1_tf: self.Y_hbN1_batch,
						self.x1_hbN2_tf: self.x1_hbN2_batch, self.x2_hbN2_tf: self.x2_hbN2_batch, self.YhbN2_tf: self.Y_hbN2_batch,
						self.x1_c_tf: self.x1_c_batch,   self.x2_c_tf: self.x2_c_batch, self.t_c_tf: self.t_c_batch, self.Yc_tf: self.Yc_batch,
						self.x1_fc_tf: self.x1_fc_batch, self.x2_fc_tf: self.x2_fc_batch, self.t_fc_tf: self.t_fc_batch,
						self.x1_cbN1_tf: self.x1_cbN1_batch, self.x2_cbN1_tf: self.x2_cbN1_batch, self.t_cbN1_tf: self.t_cbN1_batch, self.YcbN1_tf: self.YcbN1_batch,
						self.x1_cbN2_tf: self.x1_cbN2_batch, self.x2_cbN2_tf: self.x2_cbN2_batch, self.t_cbN2_tf: self.t_cbN2_batch, self.YcbN2_tf: self.YcbN2_batch}

					'''Run Adam optimizer'''       
					_, loss_batch = self.sess.run([self.optimizer_CAD, self.loss], feed_dict = tf_dict_batch)

					self.rloss_batch.append(loss_batch)

					if it % num_print_opt == 0:
						# Mimi-batch
						loss_k_bch, loss_h_bch, loss_c_bch, loss_f_bch, loss_fc_bch = self.sess.run([self.loss_k, self.loss_h, 
							self.loss_c, self.loss_pde_f,self.loss_pde_fc], feed_dict= tf_dict_batch)

						# Full-batch						
						loss, loss_k,loss_h,loss_c,loss_f,loss_fc = self.sess.run([self.loss, self.loss_k,self.loss_h,
							self.loss_c,self.loss_pde_f,self.loss_pde_fc],feed_dict=tf_dict_CAD)
						loss_cN1,loss_cN2 = self.sess.run([self.loss_pde_cN1,self.loss_pde_cN2],feed_dict=tf_dict_CAD)

						elapsed = time.time() - start_time

						# Full-batch
						print('It: {0: d} | Time: {1: .2f} | Loss: {2: .3e} | Loss_k: {3: .3e} | Loss_h: {4: .3e} | Loss_c: {5: .3e}'.\
							format(it, elapsed, loss, loss_k, loss_h, loss_c)) 
						print('--- Loss_f: {0: .3e} | Loss_fc: {1: .3e} | Loss_cN1: {2: .3e} | Loss_cN2: {3: .3e}'.format(loss_f,loss_fc,loss_cN1, loss_cN2))
						
						# Mini-batch
						# print('It: {0: d}, Time: {1: .2f} , Loss: {2: .3e}, Loss_k: {3: .3e}, Loss_h: {4: .3e}, Loss_c: {5: .3e}'.format(it, elapsed, 
						#     loss_batch, loss_k_bch, loss_h_bch, loss_c_bch))
						# print('--- Loss_f: {0: .3e}, Loss_fc: {1: .3e}'.format(loss_f_bch,loss_fc_bch)) 
						# print(' Batch loss - It: {0: d} | mLoss: {1:.3e} | mLoss_c: {2:.3e} | mLoss_fc: {3:.3e}'.format(it,loss_batch,loss_c_bch,loss_fc_bch))

						self.rloss.append(loss) # Total loss
						self.rloss_k.append(loss_k)
						self.rloss_h.append(loss_h)
						self.rloss_c.append(loss_c)
						self.rloss_pde_f.append(loss_f)
						self.rloss_pde_fc.append(loss_fc)
						self.rloss_pde_cN1.append(loss_cN1)
						self.rloss_pde_cN2.append(loss_cN2)

						self.rloss_k_batch.append(loss_k_bch)
						self.rloss_h_batch.append(loss_h_bch)
						self.rloss_c_batch.append(loss_c_bch)

						self.rloss_pde_fc_batch.append(loss_fc_bch)
						self.rloss_pde_f_batch.append(loss_f_bch)

						start_time = time.time()

					if loss < self.conv_opt and self.conv_opt != 0:
						break

			self.index_opt = it
			if flag_solv == 'BFGS' or flag_solv == 'BFGS-f':
				loss, loss_k,loss_h,loss_c,loss_f,loss_fc = self.sess.run([self.loss, self.loss_k,self.loss_h,
					self.loss_c,self.loss_pde_f,self.loss_pde_fc],feed_dict=tf_dict_CAD)
				loss_cN1,loss_cN2 = self.sess.run([self.loss_pde_cN1,self.loss_pde_cN2],feed_dict=tf_dict_CAD)

				print('Before use BFGS - Iteration: {0: d} | Loss: {1: .3e} | Loss_k: {2: .3e} | Loss_h: {3: .3e} | Loss_c: {4: .3e}'.\
					format(it, loss, loss_k, loss_h, loss_c)) 
				print('Before use BFGS - Iteration: {0: d} | Loss: {1: .3e} | Loss_k: {2: .3e} | Loss_h: {3: .3e} | Loss_c: {4: .3e}'.\
					format(it, loss, loss_k, loss_h, loss_c),file=f) 
				print('--- Loss_f: {0: .3e} | Loss_fc: {1: .3e} | Loss_cN1: {2: .3e} | Loss_cN2: {3: .3e}'.format(loss_f,loss_fc,loss_cN1, loss_cN2),file=f)				

				self.index_opt = 1
				start_time = time.time()
				# if flag_solv == 'BFGS':
				# 	self.optimizer_CAD_BFGS.minimize(self.sess, 
				# 						feed_dict = tf_dict_CAD,         
				# 						fetches = [self.loss,self.loss_c,self.loss_pde_fc,self.loss_pde_cN1,self.loss_pde_cN2],
				# 						loss_callback = self.callback_AD)
				if flag_solv == 'BFGS-f':
					self.optimizer_CAD_BFGS.minimize(self.sess, 
												feed_dict = tf_dict_CAD,         
												fetches = [self.loss, self.loss_k, self.loss_h, self.loss_c],
												loss_callback = self.callback_CAD_fast)

				elapsed = time.time() - start_time
				print('BFGS Training Time: {}'.format(elapsed),file=f)

		loss_end = self.sess.run(self.loss,feed_dict=tf_dict_CAD)
		print('Total Iteration: {0: d} | loss: {1: e}'.format(self.index_opt,loss_end),file=f)

		if self.type_reg != 0:
			loss_reg = self.sess.run(self.reg_term)
			print('loss_reg: {0: e}'.format(loss_reg),file=f)

		# Save the variables to disk.
		if self.pro_params['model_save'] == 'True':
			save_path = self.saver.save(self.sess, self.save_file) # tf.compat.v1.train.Saver
			print("Model saved in path: %s" % save_path,file=f)

	def out_loss_components(self):
		f = self.f
		f1_loss = self.f1_loss
		flag_pro = self.flag_pro

		# Define a dictionary for associating placeholders with data
		if self.if_BC_h == 0:
			tf_dict = {
						self.x1_k_tf: self.x1_k, self.x2_k_tf: self.x2_k, self.Yk_tf: self.Y_k,
						self.x1_h_tf: self.x1_h, self.x2_h_tf: self.x2_h, self.Yh_tf: self.Y_h,
						self.x1_f_tf: self.x1_f, self.x2_f_tf: self.x2_f, self.Yf_tf: self.Y_f,		
						self.x1_c_tf: self.x1_c, self.x2_c_tf: self.x2_c, self.t_c_tf: self.t_c, self.Yc_tf: self.Y_c,
						self.x1_fc_tf: self.x1_fc, self.x2_fc_tf: self.x2_fc, self.t_fc_tf: self.t_fc,
						self.x1_cbN1_tf: self.x1_cbN1, self.x2_cbN1_tf: self.x2_cbN1, self.t_cbN1_tf: self.t_cbN1, self.YcbN1_tf: self.Y_cbN1,
						self.x1_cbN2_tf: self.x1_cbN2, self.x2_cbN2_tf: self.x2_cbN2, self.t_cbN2_tf: self.t_cbN2, self.YcbN2_tf: self.Y_cbN2}
		elif self.if_BC_h == 1:
			tf_dict = {
						self.x1_k_tf: self.x1_k, self.x2_k_tf: self.x2_k, self.Yk_tf: self.Y_k,
						self.x1_h_tf: self.x1_h, self.x2_h_tf: self.x2_h, self.Yh_tf: self.Y_h,
						self.x1_f_tf: self.x1_f, self.x2_f_tf: self.x2_f, self.Yf_tf: self.Y_f,
						self.x1_hbD_tf: self.x1_hbD, self.x2_hbD_tf: self.x2_hbD, self.YhbD_tf: self.Y_hbD,
						self.x1_hbN1_tf: self.x1_hbN1, self.x2_hbN1_tf: self.x2_hbN1, self.YhbN1_tf: self.Y_hbN1,
						self.x1_hbN2_tf: self.x1_hbN2, self.x2_hbN2_tf: self.x2_hbN2, self.YhbN2_tf: self.Y_hbN2,
						self.x1_c_tf: self.x1_c, self.x2_c_tf: self.x2_c, self.t_c_tf: self.t_c, self.Yc_tf: self.Y_c,
						self.x1_fc_tf: self.x1_fc, self.x2_fc_tf: self.x2_fc, self.t_fc_tf: self.t_fc,
						self.x1_cbN1_tf: self.x1_cbN1, self.x2_cbN1_tf: self.x2_cbN1, self.t_cbN1_tf: self.t_cbN1, self.YcbN1_tf: self.Y_cbN1,
						self.x1_cbN2_tf: self.x1_cbN2, self.x2_cbN2_tf: self.x2_cbN2, self.t_cbN2_tf: self.t_cbN2, self.YcbN2_tf: self.Y_cbN2}
		elif self.if_BC_h == 2:
			tf_dict = {
						self.x1_k_tf: self.x1_k, self.x2_k_tf: self.x2_k, self.Yk_tf: self.Y_k,
						self.x1_h_tf: self.x1_h, self.x2_h_tf: self.x2_h, self.Yh_tf: self.Y_h,
						self.x1_f_tf: self.x1_f, self.x2_f_tf: self.x2_f, self.Yf_tf: self.Y_f,
						self.x1_hbD_tf: self.x1_hbD, self.x2_hbD_tf: self.x2_hbD, self.YhbD_tf: self.Y_hbD,
						self.x1_c_tf: self.x1_c, self.x2_c_tf: self.x2_c, self.t_c_tf: self.t_c, self.Yc_tf: self.Y_c,
						self.x1_fc_tf: self.x1_fc, self.x2_fc_tf: self.x2_fc, self.t_fc_tf: self.t_fc,
						self.x1_cbN1_tf: self.x1_cbN1, self.x2_cbN1_tf: self.x2_cbN1, self.t_cbN1_tf: self.t_cbN1, self.YcbN1_tf: self.Y_cbN1,
						self.x1_cbN2_tf: self.x1_cbN2, self.x2_cbN2_tf: self.x2_cbN2, self.t_cbN2_tf: self.t_cbN2, self.YcbN2_tf: self.Y_cbN2}


		loss_k = self.sess.run(self.loss_k,feed_dict=tf_dict)
		loss_h = self.sess.run(self.loss_h,feed_dict=tf_dict)
		loss_f = self.sess.run(self.loss_pde_f,feed_dict=tf_dict)

		loss_c   = self.sess.run(self.loss_c,feed_dict=tf_dict)
		loss_fc  = self.sess.run(self.loss_pde_fc,feed_dict=tf_dict)
		loss_cN1 = self.sess.run(self.loss_pde_cN1,feed_dict=tf_dict)
		loss_cN2 = self.sess.run(self.loss_pde_cN2,feed_dict=tf_dict)

		print('Final: Problem type: {}'.format(flag_pro),file=f)
		print('Loss_k: {} | Loss_h: {} | Loss_f: {}'.format(loss_k, loss_h, loss_f),file=f)
		print('Loss_c: {} | Loss_fc: {} | loss_cN1: {} | loss_cN2: {}'.format(loss_c, loss_fc, loss_cN1, loss_cN2),file=f)

		# print('{0:.3e} {1:.3e} {2:.3e}'.format(loss_k,loss_h,loss_f),file=f1_loss)
		# print('Loss_k: {0: .3e}, Loss_h: {1: .3e}, Loss_f: {2: .3e}, Loss_D: {3: .3e}, Loss_N: {4: .3e}, Loss_Ns: {5: .3e}'\
		# 	.format(loss_end_loss_k,loss_end_loss_h,loss_end_loss_f,loss_end_loss_D,loss_end_loss_N,loss_end_loss_Ns),file=f)
		
		# print('Loss_c: {0: .3e}, Loss_fc: {1: .3e}, Loss_cD: {2: .3e}, Loss_cN1: {3: .3e}, Loss_cN2: {4: .3e}'\
		# 	.format(loss_end_loss_c,loss_end_loss_fc,loss_end_loss_cD,loss_end_loss_cN1,loss_end_loss_cN2),file=f)

		# print('Scaled Loss_k: {0: .3e}, Loss_h: {1: .3e}, Loss_f: {2: .3e}, Loss_D: {3: .3e}, Loss_N: {4: .3e}, Loss_Ns: {5: .3e}'\
		#     .format(loss_end_loss_k_sca,loss_end_loss_h_sca,loss_end_loss_f_sca,loss_end_loss_D_sca,loss_end_loss_N_sca,loss_end_loss_Ns_sca),file=f)
		
		# print('Scaled Loss_c: {0: .3e}, Loss_fc: {1: .3e}, Loss_cD: {2: .3e}, Loss_cN1: {3: .3e}, Loss_cN2: {4: .3e})'\
		#     .format(loss_end_loss_c_sca,loss_end_loss_fc_sca,loss_end_loss_cD_sca,loss_end_loss_cN1_sca,loss_end_loss_cN2_sca),file=f)
					   
		# print('{0:.3e} {1:.3e} {2:.3e} {3:.3e} {4:.3e} {5:.3e}'\
		#     .format(loss_end_loss_k, loss_end_loss_h, loss_end_loss_f,loss_end_loss_D,loss_end_loss_N,loss_end_loss_Ns),file=f1_loss)

		# print('{0:.3e} {1:.3e} {2:.3e} {3:.3e} {4:.3e} {5:.3e} {6:.3e} {7:.3e} {8:.3e} {9:.3e} {10:.3e}'\
		# 	.format(loss_end_loss_k, loss_end_loss_h, loss_end_loss_f,loss_end_loss_D,loss_end_loss_N,loss_end_loss_Ns,\
		# 		loss_end_loss_c,loss_end_loss_fc,loss_end_loss_cD,loss_end_loss_cN1,loss_end_loss_cN2),file=f1_loss)
	   
		# print('{0:.3e} {1:.3e} {2:.3e} {3:.3e} {4:.3e} {5:.3e} {6:.3e} {7:.3e} {8:.3e} {9:.3e} {10:.3e}'\
		#     .format(loss_end_loss_k_sca, loss_end_loss_h_sca, loss_end_loss_f_sca,loss_end_loss_D_sca,loss_end_loss_N_sca,loss_end_loss_Ns_sca,\
		#         loss_end_loss_c_sca,loss_end_loss_fc_sca,loss_end_loss_cD_sca,loss_end_loss_cN1_sca,loss_end_loss_cN2_sca),file=f1_loss2)

	## Evaluates predictions at test points           
	def predict_k(self, X_star):
		# Normalization
		lb = self.lb
		ub = self.ub
		scale_coe = self.scale_coe
		X_star = sub_normalization(X_star,lb,ub,scale_coe,self.flag_nor)   
		# Predict
		tf_dict = {self.x1_k_tf: X_star[:,0:1], self.x2_k_tf: X_star[:,1:2]}    
		k_star = self.sess.run(self.k_pred, tf_dict) 
		return k_star
	
	# Evaluates predictions at test points           
	def predict_h(self, X_star): 
		# X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
		lb = self.lb
		ub = self.ub
		scale_coe = self.scale_coe
		X_star = sub_normalization(X_star,lb,ub,scale_coe,self.flag_nor) 
		# Predict
		tf_dict = {self.x1_h_tf: X_star[:,0:1], self.x2_h_tf: X_star[:,1:2]}    
		h_star = self.sess.run(self.h_pred, tf_dict) 
		return h_star

	def predict_c(self, X_star, t_star): 
		# Center around the origin
		lb = self.lb
		ub = self.ub
		scale_coe = self.scale_coe
		X_star = sub_normalization(X_star,lb,ub,scale_coe,self.flag_nor)
		
		Tm = np.tile(t_star,X_star.shape[0]).T
		Tm = Tm[:,np.newaxis]
		Tm = sub_normalization(Tm,self.t_min,self.t_max,self.scale_coe,self.flag_nor)

		# Predict	
		tf_dict = {self.x1_c_tf: X_star[:,0:1], self.x2_c_tf: X_star[:,1:2], self.t_c_tf: Tm}    
		c_star = self.sess.run(self.c_pred, tf_dict) 
		return c_star
	
	# Evaluates predictions at test points           
	def predict_f(self, X_star): 
		# Center around the origin
		# X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
		lb = self.lb
		ub = self.ub
		scale_coe = self.scale_coe
		X_star = sub_normalization(X_star,lb,ub,scale_coe,self.flag_nor) 
		# Predict
		tf_dict = {self.x1_f_tf: X_star[:,0:1], self.x2_f_tf: X_star[:,1:2]}    
		f_star = self.sess.run(self.f_pred, tf_dict) 
		return f_star  

	# Evaluates predictions at test points           
	def predict_fc(self, X_star): 
		lb = self.lb
		ub = self.ub
		scale_coe = self.scale_coe
		X_star = sub_normalization(X_star,lb,ub,scale_coe,self.flag_nor) 
		# Predict
		tf_dict = {self.x1_fc_tf: X_star[:,0:1], self.x2_fc_tf: X_star[:,1:2]}    
		fc_star = self.sess.run(self.fc_pred, tf_dict) 
		return fc_star  

	def predict_v(self, X_star): 
		# Center around the origin
		lb = self.lb
		ub = self.ub
		scale_coe = self.scale_coe
		X_star = sub_normalization(X_star,lb,ub,scale_coe,self.flag_nor) 

		# Predict
		tf_dict = {self.x1_v_tf: X_star[:,0:1], self.x2_v_tf: X_star[:,1:2]}    
		v1_star, v2_star = self.sess.run([self.v1_pred, self.v2_pred], tf_dict) 
		return v1_star, v2_star


##########
	def errl2_k(self,X_star,k_star):
		k_pred = self.predict_k(X_star)
		# Relative L2 error
		error_k = np.linalg.norm(k_star - k_pred, 2)/np.linalg.norm(k_star, 2)
		return error_k

	def errl2_h(self,X_star,h_star):
		h_pred = self.predict_h(X_star)
		# Relative L2 error
		error_h = np.linalg.norm(h_star - h_pred, 2)/np.linalg.norm(h_star, 2)        
		return error_h
