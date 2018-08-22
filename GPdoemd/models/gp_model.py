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

from GPy.models import GPRegression
from GPy.kern import Kern

from . import SurrogateModel
from ..utils import binary_dimensions


class GPModel (SurrogateModel):
	def __init__ (self, model_dict):
		super().__init__(model_dict)
		# Optional parameters
		self.gp_noise_var     = model_dict.get('gp_noise_var', 1e-6)


	"""
	Surrogate model kernels
	"""
	## Design variable kernels
	@property
	def kern_x (self):
		return None if not hasattr(self,'_kern_x') else self._kern_x
	@kern_x.setter
	def kern_x (self, value):
		if value is not None:
			assert issubclass(value, Kern)
			self._kern_x = value

	## Model parameter kernel
	@property
	def kern_p (self):
		return None if not hasattr(self,'_kern_p') else self._kern_x
	@kern_p.setter
	def kern_p (self, value):
		if value is not None:
			assert issubclass(value, Kern)
			self._kern_p = value

	def set_kernels (self, kern_x, kern_p):
		self.kern_x = kern_x
		self.kern_p = kern_p



	"""
	Surrogate model
	"""
	@property
	def gps (self):
		return None if not hasattr(self,'_gps') else self._gps
	@gps.setter
	def gps(self, value):
		assert len(value) == self.num_outputs
		self._gps = value
	@gps.deleter
	def gps (self):
		self._gps = None

	@property
	def gp_noise_var (self):
		return self._gp_noise_var
	@gp_noise_var.setter
	def gp_noise_var (self, value):
		assert isinstance(value, (int,float)) and value > 0.
		self._gp_noise_var = value


	def gp_surrogate (self, Z=None, Y=None, kern_x=None, kern_p=None):
		self.set_training_data(Z, Y)
		assert self.Z is not None and self.Y is not None

		self.set_kernels(kern_x, kern_p)
		assert self.kern_x is not None and self.kern_p is not None

		R, J = binary_dimensions(self.Z, self.binary_variables)

		gps = []
		for e in range( self.num_outputs ):
			gps.append([])
			for r in R:
				Jr = (J==r)

				if not np.any(Jr):
					gps[e].append(None)
					continue

				dim_xb = self.dim_x - self.dim_b
				dim    = self.dim_x + self.dim_p
				kernx  = self.kern_x(dim_xb, self.non_binary_variables, 'kernx')
				kernp  = self.kern_p(self.dim_p, range(self.dim_x, dim), 'kernp')
				#Zr     = self.Z[ np.ix_(Jr,  I ) ]
				Zr     = self.Z[ Jr ]
				Yr     = self.Y[ np.ix_(Jr, [e]) ]

				gp    = GPRegression(Zr, Yr, kernx * kernp)
				gps[e].append(gp)
		self.gps = gps


	def gp_load_hyp (self, index=None):
		if index is None:
			index = range( self.num_outputs )
		elif isinstance(index, int):
			index = [index]

		for e in index:
			gps  = self.gps[e]
			hyps = self.hyp[e]
			for gp,hyp in zip(gps,hyps):
				if gp is None:
					continue
				gp.update_model(False)
				gp.initialize_parameter()
				gp[:] = hyp
				gp.update_model(True)


	def gp_optimize (self, index=None, max_lengthscale=10):
		self.gp_optimise(index=index, max_lengthscale=max_lengthscale)

	def gp_optimise (self, index=None, max_lengthscale=10):
		if index is None:
			index = range( self.num_outputs )
		elif isinstance(index, int):
			index = [index]

		for e in index:
			gps = self.gps[e]
			for gp in gps:
				if gp is None:
					continue
				# Constrain noise variance
				gp.Gaussian_noise.variance.constrain_fixed(self._gp_noise_var)
				# Constrain kern_x lengthscales
				for j in range(self.dim_x-self.dim_b):
					gp.kern.kernx.lengthscale[[j]].constrain_bounded(
						lower=0., upper=max_lengthscale, warning=False )
				# Constrain kern_p lengthscales
				for j in range(self.dim_p):
					gp.kern.kernp.lengthscale[[j]].constrain_bounded(
						lower=0., upper=max_lengthscale, warning=False )
				# Optimise
				gp.optimize()

		hyp = []
		for e,gps in enumerate(self.gps):
			hyp.append([])
			for gp in gps:
				if gp is None:
					hyp[e].append(None)
				else:
					hyp[e].append(gp[:])
		self.hyp = hyp


	def _predict (self, xnew, p):
		znew = np.array([ x.tolist() + p.tolist() for x in xnew ])
		R, J = binary_dimensions(znew, self.binary_variables)
		#znew    = znew[:,I]

		n = len(znew)
		M = np.zeros((n, self.num_outputs))
		S = np.zeros((n, self.num_outputs))

		for r in R:
			Jr = J==r
			if not np.any(Jr):
				continue
			for e in range( self.num_outputs ):
				I          = np.ix_(Jr,[e])
				M[I], S[I] = self.gps[e][r].predict_noiseless(znew[Jr])
		return M, S


	"""
	Derivatives
	"""
	def _d_mu_d_p (self, e, X):
		R, J    = binary_dimensions(X, self.binary_variables)
		n, E, D = len(X), self.num_outputs, self.dim_p
		dx      = self.dim_x #- self.dim_b
		dmu     = np.zeros((n,D))
		for r in R:
			Jr = (J==r)
			if not np.any(Jr):
				continue
			Z  = self.get_Z(X[ Jr ])
			gp = self.gps[e][r]
			dmu[Jr] = gp.predictive_gradients(Z)[0][:,dx:,0]
		return dmu

	def _d2_mu_d_p2 (self, e, X):
		R, J    = binary_dimensions(X, self.binary_variables)
		n, E, D = len(X), self.num_outputs, self.dim_p
		dx      = self.dim_x #- self.dim_b
		ddmu    = np.zeros((n,D,D))
		for r in R:
			Jr  = (J==r)
			if not np.any(Jr):
				continue
			Z  = self.get_Z(X[ Jr ])
			gp  = self.gps[e][r]
			gpX = gp._predictive_variable
			tmp = -gp.kern.kernx.K(Z, gpX) * gp.posterior.woodbury_vector.T
			dt  = np.sum( gp.kern.kernp.gradients_XX(tmp, Z, gpX), axis=1 )
			ddmu[Jr] = dt[:,dx:,dx:]
		return ddmu

	def _d_s2_d_p (self, e, X):
		R, J    = binary_dimensions(X, self.binary_variables)
		n, E, D = len(X), self.num_outputs, self.dim_p
		dx      = self.dim_x #- self.dim_b
		ds2     = np.zeros((n,D))
		for r in R:
			Jr = (J==r)
			if not np.any(Jr):
				continue
			Z  = self.get_Z(X[ Jr ])
			gp = self.gps[e][r]
			ds2[Jr] = gp.predictive_gradients(Z)[1][:,dx:]
		return ds2

	def _d2_s2_d_p2 (self, e, X):
		R, J    = binary_dimensions(X, self.binary_variables)
		n, E, D = len(X), self.num_outputs, self.dim_p
		dx      = self.dim_x #- self.dim_b
		dds2    = np.zeros((n,D,D))
		for r in R:
			Jr  = (J==r)
			if not np.any(Jr):
				continue
			Z   = self.get_Z(X[ Jr ])
			gp  = self.gps[e][r]
			gpX = gp._predictive_variable
			iK  = gp.posterior.woodbury_inv

			# d k / d p
			kx = gp.kern.kernx.K(Z, gpX)
			dk = np.zeros((n, gpX.shape[0], self.dim_p))
			for i in range( gpX.shape[0] ):
				dt      = gp.kern.kernp.gradients_X(kx[:,[i]], Z, gpX[[i]])
				dk[:,i] = dt[:,dx:]
			dkiKdk = np.einsum('ijk,jn,inm->ikm', dk, iK, dk)

			# d^2 k / d p^2
			kiK = np.matmul( gp.kern.K(Z, gpX), iK )
			ddk = np.sum( gp.kern.kernp.gradients_XX(-kx*kiK, Z, gpX), axis=1 )
			ddk = ddk[:,dx:,dx:]

			# d^2 s2 / d p^2
			dds2[Jr] = -2. * ( ddk + dkiKdk )
		return dds2


	"""
	Clear model
	"""
	def clear_model (self):
		del self.gps
		super(GPModel,self).clear_model()
