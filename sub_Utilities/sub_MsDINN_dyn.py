"""
@author     : Qizhi He @ PNNL (qizhi.he@pnnl.gov)
Decription  : MsDINN subfunctions
"""
import numpy as np

def fun_BC2dyn(Xb,Yb,t_nod,pro_dim):
	'''Reshape time-independent BCs to training data with time axis'''
	num_xb = Xb.shape[0]
	Zm_b = np.zeros((num_xb*len(t_nod),pro_dim))
	Um_b = np.zeros((num_xb*len(t_nod),1))

	for i in range(len(t_nod)):
		ti = t_nod[i]
		Zm_b[i*num_xb:(i+1)*num_xb,0:pro_dim-1] = Xb
		Zm_b[i*num_xb:(i+1)*num_xb,pro_dim-1:pro_dim] = ti
		Um_b[i*num_xb:(i+1)*num_xb,:] = Yb
	return Zm_b, Um_b
