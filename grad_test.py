
import numpy as np

import GPy

from GPdoemd.kernels import Matern52 as kernel

from pdb import set_trace as st
import checkgrad


pmean = np.array([1.,0.5,0.8])
measvar = 0.0001*np.ones(3)

N = 5
X = np.random.rand(N,3)
P = pmean + 0.001*np.random.randn(N,3)
Z = np.c_[X,P]
Y = P*X + np.sqrt(measvar)*np.ones((N,3))

xnew = np.random.rand(2,3)

kernx = kernel(3,range(3),'kernx')
kernp = kernel(3,range(3,6),'kernp')
gp = GPy.models.GPRegression(Z,Y[:,[0]],kernx*kernp)
#gp = GPy.models.GPRegression(X,Y[:,[0]],kernx)
gp.update_model(False)
gp.initialize_parameter()
gp[:] = 5. * np.random.rand(len(gp[:]))
gp.update_model(True)

def grad (X):
	Z = np.array([x.tolist() + pmean.tolist() for x in X])
	k = gp.kern.K(Z,gp.X)
	kp = gp.kern.kernp.K(Z,gp.X)
	dk = kp[:,:,None] * gp.kern.kernx.d_k_d_x(Z,gp.X)
	return k,dk

def hess (X):
	Z = np.array([x.tolist() + pmean.tolist() for x in X])
	k = gp.kern.K(Z,gp.X)
	kp = gp.kern.kernp.K(Z,gp.X)
	ddk = kp[:,:,None,None] * gp.kern.kernx.d2_k_d_x2(Z,gp.X)
	return k,ddk

def r_grad (X):
	r = gp.kern._scaled_dist(X,gp.X)
	dr = gp.kern.d_r_d_x(X,gp.X)
	return r, dr

def r_hess (X):
	r = gp.kern._scaled_dist(X,gp.X)
	ddr = gp.kern.d2_r_d_x2(X,gp.X)
	return r, ddr

#ddydh = checkgrad.gradient(grad,xnew,flatten=True)
ddydh = checkgrad.hessian(hess,xnew,flatten=True)
#ddydh = checkgrad.gradient(r_grad,xnew,flatten=True)
#ddydh = checkgrad.hessian(r_hess,xnew,flatten=True)
print(ddydh)

