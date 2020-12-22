"""
@author     : Qizhi He @ PNNL (qizhi.he@pnnl.gov)
Decription  : Multiphyics Model-Data Integration Neural Network (MsDINN)
"""
import tensorflow as tf
from model_MsDINN_CAD_dyn           import *

##################################################
### NN type
##################################################
def sub_NN_type(type_NN,input_dim):
	if type_NN == 23:
		layers = [input_dim] + [30] * 2 + [1]
	elif type_NN == 33:
		layers = [input_dim] + [30] * 3 + [1]
	elif type_NN == 34:
		layers = [input_dim] + [40] * 3 + [1]
	elif type_NN == 36:
		layers = [input_dim] + [60] * 3 + [1]
	elif type_NN == 46:
		layers = [input_dim] + [60] * 4 + [1]
	elif type_NN == 56:
		layers = [input_dim] + [60] * 5 + [1]
	else:
		raise NotImplementedError
	return layers

# ##################################################
# ### CAD
# ##################################################
class MDINN_CAD(model_MsDINN_CAD_dyn):
	'''Define Loss for forward'''
	def loss_func_ele_darcy(self):
		flag_lss = self.flag_lss
		coe_nor_k2 = self.coe_nor_k2
		coe_nor_h2 = self.coe_nor_h2

		if flag_lss == 0:
			loss_k      = tf.reduce_mean(tf.square(self.Yk_tf - self.k_pred))
			loss_h      = tf.reduce_mean(tf.square(self.Yh_tf - self.h_pred))
			loss_pde_f  = tf.reduce_mean(tf.square(self.Yf_tf - self.f_pred))
		elif flag_lss == 1:
			loss_k      = tf.reduce_mean(tf.square(self.Yk_tf - self.k_pred))
			loss_h      = coe_nor_h2/coe_nor_k2 * tf.reduce_mean(tf.square(self.Yh_tf - self.h_pred))
			loss_pde_f  = coe_nor_h2 * tf.reduce_mean(tf.square(self.Yf_tf - self.f_pred))
		elif flag_lss == 11:
			loss_k      = tf.reduce_mean(tf.square(self.Yk_tf - self.k_pred))
			loss_h      = coe_nor_h2/coe_nor_k2 * tf.reduce_mean(tf.square(self.Yh_tf - self.h_pred))
			loss_pde_f  = tf.reduce_mean(tf.square(self.Yf_tf - self.f_pred))
		elif flag_lss == 12: # (default in v6s4 for khr)
			loss_k      = tf.reduce_mean(tf.square(self.Yk_tf - self.k_pred))
			loss_h      = tf.reduce_mean(tf.square(self.Yh_tf - self.h_pred))
			loss_pde_f  = coe_nor_k2 * coe_nor_h2 * tf.reduce_mean(tf.square(self.Yf_tf - self.f_pred))

		if self.if_BC_h == 1:
			loss_pde_hD  = tf.reduce_mean(tf.square(self.YhbD_tf  - self.hbD_pred))
			if flag_lss == 0:
				loss_pde_hN1 = tf.reduce_mean(tf.square(self.YhbN1_tf - self.hbN1_pred))
				loss_pde_hN2 = tf.reduce_mean(tf.square(self.YhbN2_tf - self.hbN2_pred))
			elif flag_lss == 1:
				loss_pde_hN1 = coe_nor_h2 * tf.reduce_mean(tf.square(self.YhbN1_tf - self.hbN1_pred))
				loss_pde_hN2 = coe_nor_h2 * tf.reduce_mean(tf.square(self.YhbN2_tf - self.hbN2_pred))			
		elif self.if_BC_h == 2:
			loss_pde_hD  = tf.reduce_mean(tf.square(self.YhbD_tf  - self.hbD_pred))
			loss_pde_hN1 = 0.0
			loss_pde_hN2 = 0.0			
		else:
			loss_pde_hD  = 0.0
			loss_pde_hN1 = 0.0
			loss_pde_hN2 = 0.0

		return loss_k, loss_h, loss_pde_f, loss_pde_hD, loss_pde_hN1, loss_pde_hN2

	def loss_function_khr(self):
		para_k, para_h, para_c, para_kh, para_khc  = self.para_k, self.para_h, self.para_c, self.para_kh, self.para_khc
		if self.if_BC_h == 1:
			res  = para_k * self.loss_k + para_h * self.loss_h + para_kh *  self.loss_pde_f \
				+ para_h * self.loss_pde_hbD + para_kh * self.loss_pde_hbN1 + para_kh * self.loss_pde_hbN2
		elif self.if_BC_h == 2:
			res  = para_k * self.loss_k + para_h * self.loss_h + para_kh *  self.loss_pde_f \
				+ para_h * self.loss_pde_hbD
		else:
			res  = para_k * self.loss_k + para_h * self.loss_h + para_kh *  self.loss_pde_f
		return res

	def loss_function_hr(self):
		para_k, para_h, para_c, para_kh, para_khc  = self.para_k, self.para_h, self.para_c, self.para_kh, self.para_khc
		if self.if_BC_h == 1:
			res  = para_h * self.loss_h + para_kh *  self.loss_pde_f \
				+ para_h * self.loss_pde_hbD + para_kh * self.loss_pde_hbN1 + para_kh * self.loss_pde_hbN2
		elif self.if_BC_h == 2:
			res  = para_h * self.loss_h + para_kh *  self.loss_pde_f \
				+ para_h * self.loss_pde_hbD
		else:
			res  = para_h * self.loss_h + para_kh *  self.loss_pde_f
		return res

	def loss_func_ele_AD(self):
		flag_lss = self.flag_lss
		coe_nor_k2, coe_nor_h2, coe_nor_c2 = self.coe_nor_k2, self.coe_nor_h2, self.coe_nor_c2

		if flag_lss == 0:
			loss_c        = tf.reduce_mean(tf.square(self.Yc_tf - self.c_pred))
			loss_pde_fc   = tf.reduce_mean(tf.square(self.fc_pred))
			loss_pde_cN1  = tf.reduce_mean(tf.square(self.YcbN1_tf - self.cbN1_pred))
			loss_pde_cN2  = tf.reduce_mean(tf.square(self.YcbN2_tf - self.cbN2_pred))
		elif flag_lss == 1:
			loss_c        = coe_nor_c2/coe_nor_k2 * tf.reduce_mean(tf.square(self.Yc_tf - self.c_pred))
			loss_pde_fc   = coe_nor_c2 * coe_nor_h2 * tf.reduce_mean(tf.square(self.fc_pred))
			loss_pde_cN1  = coe_nor_c2 * tf.reduce_mean(tf.square(self.YcbN1_tf - self.cbN1_pred))
			loss_pde_cN2  = coe_nor_c2 * tf.reduce_mean(tf.square(self.YcbN2_tf - self.cbN2_pred))
		elif flag_lss == 11:
			loss_c        = coe_nor_c2/coe_nor_k2 * tf.reduce_mean(tf.square(self.Yc_tf - self.c_pred))
			loss_pde_fc   = tf.reduce_mean(tf.square(self.fc_pred))
			loss_pde_cN1  = tf.reduce_mean(tf.square(self.YcbN1_tf - self.cbN1_pred))
			loss_pde_cN2  = tf.reduce_mean(tf.square(self.YcbN2_tf - self.cbN2_pred))
		return loss_c, loss_pde_fc, loss_pde_cN1, loss_pde_cN2

	def loss_function(self):
		para_k, para_h, para_c, para_kh, para_khc  = self.para_k, self.para_h, self.para_c, self.para_kh, self.para_khc
		coe_nor_k2, coe_nor_h2, coe_nor_c2 = self.coe_nor_k2, self.coe_nor_h2, self.coe_nor_c2

		if self.flag_pro == 'i-CAD':
			if self.flag_lsty == 0:
				res  = self.loss_khr + para_c * self.loss_c + para_khc * (self.loss_pde_fc + self.loss_pde_cN1 + self.loss_pde_cN2)
			else:
				raise NotImplementedError
		return res
