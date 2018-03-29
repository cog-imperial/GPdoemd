
import numpy as np 

import GPy.kern


class Stationary:
	def d_r_d_x (self, X1, X2):
		if X1.shape[1] > self.input_dim:
			X1 = X1[:, self.active_dims]
			X2 = X2[:, self.active_dims]
		dist = (X1[:, None, :] - X2[None, :, :])
		dr   = self._inv_dist(X1, X2)[:,:,None] * dist / self.lengthscale**2
		return dr

	def d2_r_d_x2 (self, X1, X2):
		if X1.shape[1] > self.input_dim:
			X1 = X1[:, self.active_dims]
			X2 = X2[:, self.active_dims]
		iL   = 1. / self.lengthscale**2
		dist = (X1[:, None, :] - X2[None, :, :]) * iL
		ir   = self._inv_dist(X1, X2)[:,:,None,None]
		rr   = dist[:,:,:,None] * dist[:,:,None,:]
		ddr  = ir * ( np.diag(iL)[None,None,:,:] - ir**2 * rr )
		return ddr

	def d_k_d_x (self, X1, X2):
		X1 = X1[:, self.active_dims]
		X2 = X2[:, self.active_dims]
		dk = self.dK_dr_via_X(X1, X2)
		return dk[:,:,None] * self.d_r_d_x(X1,X2)

	def d2_k_d_x2 (self, X1, X2):
		X1  = X1[:, self.active_dims]
		X2  = X2[:, self.active_dims]

		ddk = self.dK2_drdr_via_X(X1, X2)
		dr  = self.d_r_d_x(X1, X2)
		ddk = ddk[:,:,None,None] * (dr[:,:,:,None] * dr[:,:,None,:])

		dk  = self.dK_dr_via_X(X1, X2)
		ddr = self.d2_r_d_x2(X1, X2)
		dk  = dk[:,:,None,None] * ddr

		return ddk + dk


class RBF (GPy.kern.RBF, Stationary):
	def __init__ (self, d, drange, name):
		GPy.kern.RBF.__init__(self, input_dim=d, active_dims=drange, \
								name=name, ARD=True)
		Stationary.__init__(self)


class Matern32 (GPy.kern.Matern32, Stationary):
	def __init__ (self, d, drange, name):
		GPy.kern.Matern32.__init__(self,input_dim=d, active_dims=drange, \
									name=name, ARD=True)
		Stationary.__init__(self)

	def dK2_drdr_via_X(self, X1, X2):
		return self.dK2_drdr(self._scaled_dist(X1, X2))

	def dK2_drdr(self, r):
		ar = np.sqrt(3.)*r
		return 3. * self.variance * (ar - 1) * np.exp(-ar)


class Matern52 (GPy.kern.Matern52, Stationary):
	def __init__ (self, d, drange, name):
		GPy.kern.Matern52.__init__(self,input_dim=d, active_dims=drange, \
									name=name, ARD=True)
		Stationary.__init__(self)

	def dK2_drdr_via_X(self, X1, X2):
		return self.dK2_drdr(self._scaled_dist(X1, X2))

	def dK2_drdr(self, r):
		ar = np.sqrt(5)*r 
		return 5./3. * self.variance * (ar**2 - ar - 1) * np.exp(-ar)









