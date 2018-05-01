
import numpy as np 

import GPy.kern


"""
RBF / squared exponntial kernel
"""
class RBF (GPy.kern.RBF):
	def __init__ (self, d, drange, name):
		GPy.kern.RBF.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)

"""
Exponential kernel
"""
class Exponential (GPy.kern.Exponential):
	def __init__ (self, d, drange, name):
		GPy.kern.Exponential.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)

	def dK2_drdr(self, r):
		return self.K_of_r(r)

"""
Matern-3/2 kernel
"""
class Matern32 (GPy.kern.Matern32):
	def __init__ (self, d, drange, name):
		GPy.kern.Matern32.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)

	def dK2_drdr(self, r):
		ar = np.sqrt(3.)*r
		return 3. * self.variance * (ar - 1) * np.exp(-ar)

"""
Matern-5/2 kernel
"""
class Matern52 (GPy.kern.Matern52):
	def __init__ (self, d, drange, name):
		GPy.kern.Matern52.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)

	def dK2_drdr(self, r):
		ar = np.sqrt(5)*r 
		return 5./3. * self.variance * (ar**2 - ar - 1) * np.exp(-ar)

"""
Cosine kernel
"""
class Cosine (GPy.kern.Cosine):
	def __init__ (self, d, drange, name):
		GPy.kern.Cosine.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)

	def dK2_drdr(self, r):
		return -self.K_of_r(r)

"""
Rational quadratic kernel
"""
class RatQuad(GPy.kern.RatQuad):
	def __init__ (self, d, drange, name):
		GPy.kern.RatQuad.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)

	def dK2_drdr(self, r):
		r2  = np.square(r)
		lp  = np.log1p(r2 / 2.)
		a   = (self.power + 1) * lp
		dr1 = -self.variance * self.power * np.exp(-a)
		a  += lp
		dr2 = self.variance * self.power * (self.power + 1) * r2 * np.exp(-a)
		return dr1 + dr2