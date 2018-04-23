
import numpy as np 

import GPy.kern
from .kern import Kern

class Stationary (Kern):
	def d_r_d_x (self, X1, X2):
		# d r(X1,X2) / d X1
		if X1.shape[1] > self.input_dim:
			X1 = X1[:, self.active_dims]
			X2 = X2[:, self.active_dims]
		dist = (X1[:, None, :] - X2[None, :, :])
		dr   = self._inv_dist(X1, X2)[:,:,None] * dist / self.lengthscale**2
		return dr # ( X1.shape[0], X2.shape[0], X1.shape[1] )

	def d2_r_d_x2 (self, X1, X2):
		# d^2 r(X1,X2) / d X1^2
		if X1.shape[1] > self.input_dim:
			X1 = X1[:, self.active_dims]
			X2 = X2[:, self.active_dims]
		iL   = 1. / self.lengthscale**2
		dist = (X1[:, None, :] - X2[None, :, :]) * iL
		ir   = self._inv_dist(X1, X2)[:,:,None,None]
		rr   = dist[:,:,:,None] * dist[:,:,None,:]
		ddr  = ir * ( np.diag(iL)[None,None,:,:] - ir**2 * rr )
		return ddr # ( X1.shape[0], X2.shape[0], X1.shape[1], X1.shape[1] )

	def d_k_d_x (self, X1, X2):
		# d k(X1,X2) / d X1
		X1 = X1[:, self.active_dims]
		X2 = X2[:, self.active_dims]
		dk = self.dK_dr_via_X(X1, X2)[:,:,None] * self.d_r_d_x(X1,X2)
		return dk  # ( X1.shape[0], X2.shape[0], X1.shape[1] )

	def d2_k_d_x2 (self, X1, X2):
		# d^2 k(X1,X2) / d X1^2
		X1  = X1[:, self.active_dims]
		X2  = X2[:, self.active_dims]

		ddk = self.dK2_drdr_via_X(X1, X2)
		dr  = self.d_r_d_x(X1, X2)
		ddk = ddk[:,:,None,None] * (dr[:,:,:,None] * dr[:,:,None,:])

		dk  = self.dK_dr_via_X(X1, X2)
		ddr = self.d2_r_d_x2(X1, X2)
		dk  = dk[:,:,None,None] * ddr

		return ddk + dk # ( X1.shape[0], X2.shape[0], X1.shape[1], X1.shape[1] )

	def dK2_drdr_via_X(self, X1, X2):
		return self.dK2_drdr(self._scaled_dist(X1, X2))


"""
RBF / squared exponntial kernel
"""
class RBF (GPy.kern.RBF, Stationary):
	def __init__ (self, d, drange, name):
		GPy.kern.RBF.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)
		Stationary.__init__(self)

"""
Exponential kernel
"""
class Exponential (GPy.kern.Exponential, Stationary):
	def __init__ (self, d, drange, name):
		GPy.kern.Exponential.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)
		Stationary.__init__(self)

	def dK2_drdr(self, r):
		return self.K_of_r(r)

"""
Matern-3/2 kernel
"""
class Matern32 (GPy.kern.Matern32, Stationary):
	def __init__ (self, d, drange, name):
		GPy.kern.Matern32.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)
		Stationary.__init__(self)

	def dK2_drdr(self, r):
		ar = np.sqrt(3.)*r
		return 3. * self.variance * (ar - 1) * np.exp(-ar)

"""
Matern-5/2 kernel
"""
class Matern52 (GPy.kern.Matern52, Stationary):
	def __init__ (self, d, drange, name):
		GPy.kern.Matern52.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)
		Stationary.__init__(self)

	def dK2_drdr(self, r):
		ar = np.sqrt(5)*r 
		return 5./3. * self.variance * (ar**2 - ar - 1) * np.exp(-ar)

"""
Cosine kernel
"""
class Cosine (GPy.kern.Cosine, Stationary):
	def __init__ (self, d, drange, name):
		GPy.kern.Cosine.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)
		Stationary.__init__(self)

	def dK2_drdr(self, r):
		return -self.K_of_r(r)

"""
Rational quadratic kernel
"""
class RatQuad(GPy.kern.RatQuad, Stationary):
	def __init__ (self, d, drange, name):
		GPy.kern.RatQuad.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)
		Stationary.__init__(self)

	def dK2_drdr(self, r):
		r2  = np.square(r)
		a   = (self.power + 2) * np.log1p(r2/2.)
		dr1 = self.variance * self.power * (self.power + 1) * r2 * np.exp(-a)
		dr2 = self.dK_dr(r) / r
		return dr1 + dr2


