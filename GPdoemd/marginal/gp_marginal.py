"""
MIT License

Copyright (c) 2018 Simon Olofsson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from pdb import set_trace as st 

from ..utils import binary_dimensions

"""
Marginaliser class
"""
class GPMarginal:
	def __init__(self, model, param_mean):
		self.gps        = model.gps               # List of GPs
		self.param_mean = param_mean              # Parameter mean
		self.bin_var    = model.binary_variables  # Binary variable indices
		
	@property
	def Sigma (self):
		return None if not hasattr(self,'_Sigma') else self._Sigma
	@Sigma.setter
	def Sigma (self, value):
		assert isinstance(value, np.ndarray)
		dim_p = len(self.param_mean)
		assert value.shape == (dim_p, dim_p)
		self._Sigma = value

	def get_Z (self, X):
		return np.array([ x.tolist() + self.param_mean.tolist() for x in X ])

	def d_mu_d_p (self, gp, X):
		Z   = self.get_Z(X)
		dim = X.shape[1]
		dmu = gp.predictive_gradients(Z)[0][:, dim:, 0]
		return dmu

	def d2_mu_d_p2 (self, gp, X):
		Z   = self.get_Z(X)
		gpX = gp._predictive_variable
		dim = X.shape[1]

		tmp  = -gp.kern.kernx.K(Z, gpX) * gp.posterior.woodbury_vector.T
		ddmu = np.sum( gp.kern.kernp.gradients_XX(tmp, Z, gpX), axis=1 )
		return ddmu[:,dim:,dim:]

	def d_s2_d_p (self, gp, X):
		Z   = self.get_Z(X)
		dim = X.shape[1]
		ds2 = gp.predictive_gradients(Z)[1][:, dim:]
		return ds2

	def d2_s2_d_p2 (self, gp, X):
		Z   = self.get_Z(X)
		gpX = gp._predictive_variable
		dim = X.shape[1]

		# d k / d p
		kx = gp.kern.kernx.K(Z, gpX)
		dk = np.zeros((X.shape[0], gpX.shape[0], Z.shape[1] - dim))
		for i in range( gpX.shape[0] ):
			dk[:,i] = gp.kern.kernp.gradients_X(kx[:,[i]], Z, gpX[[i]])[:,dim:]
		dkiKdk = np.einsum('ijk,jn,inm->ikm', dk, gp.posterior.woodbury_inv, dk)

		# d^2 k / d p^2
		kiK = np.matmul( gp.kern.K(Z, gpX), gp.posterior.woodbury_inv )
		ddk = np.sum( gp.kern.kernp.gradients_XX(-kx*kiK, Z, gpX), axis=1 )
		ddk = ddk[:,dim:,dim:]

		# d^2 s2 / d p^2
		dds2ddp = -2. * ( ddk + dkiKdk )
		return dds2ddp
	

	def compute_param_covar (self, Xdata, meas_noise_var):
		# Dimensions
		E       = len(self.gps)
		D       = len(self.param_mean)
		R, I, J = binary_dimensions(Xdata, self.bin_var)
		Xdata   = Xdata[:, I] 

		if isinstance(meas_noise_var, (int, float)):
			meas_noise_var = np.array([meas_noise_var] * E)
		
		# Invert measurement noise covariance
		if meas_noise_var.ndim == 1: 
			imeasvar = np.diag(1./meas_noise_var)
		else: 
			imeasvar = np.linalg.inv(meas_noise_var)
		
		# Inverse covariance matrix
		iA = np.zeros( (D, D) )

		def get_d_mu_d_p (gps, X):
			dmu = np.zeros((len(X),D))
			for r in R:
				Jr = (J==r)
				if not np.any(Jr):
					continue
				dmu[Jr] = self.d_mu_d_p(gps[r], X[Jr])
			return dmu

		for e1 in range(E):
			dmu1 = get_d_mu_d_p(self.gps[e1], Xdata)

			iA += imeasvar[e1,e1] * np.matmul(dmu1.T, dmu1)

			if meas_noise_var.ndim == 1:
				continue
			for e2 in range(e1+1,E):
				if imeasvar[e1,e2] == 0.:
					continue
				dmu2 = get_d_mu_d_p(self.gps[e2], Xdata)

				iA += imeasvar[e1,e2] * np.matmul(dmu1.T, dmu2)
				iA += imeasvar[e2,e1] * np.matmul(dmu2.T, dmu1)

		self.Sigma = np.linalg.inv(iA)


