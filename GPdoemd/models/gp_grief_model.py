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

from GPy.kern import Kern
from gp_grief.kern import GriefKernel, GPyKernel
from gp_grief.grid import InducingGrid
from gp_grief.models import GPGriefModel as GPGModel
from gp_grief.tensors import expand_SKC

from . import SurrogateModel
from ..utils import binary_dimensions


class GPGriefModel (SurrogateModel):
	def __init__ (self, model_dict):
		super().__init__(model_dict)
		# Optional parameters
		self.gp_noise_var = model_dict.get('gp_noise_var', 1e-6)

	"""
	Surrogate model kernels
	"""
	## Design variable kernels
	@property
	def kern_list (self):
		return None if not hasattr(self,'_kern_list') else self._kern_list
	@kern_list.setter
	def kern_list (self, value):
		if value is not None:
			if not isinstance(value, list):
				value = [value, ] * (self.dim_x + self.dim_p - self.dim_b) 
			for v in value:
				assert issubclass(v, Kern)
			self._kern_list = value


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


	def gp_surrogate (self, Z=None, Y=None, kern_list=None, mbar=25):
		self.set_training_data(Z, Y)
		assert self.Z is not None and self.Y is not None

		self.kern_list = kern_list
		assert self.kern_list is not None

		assert isinstance(mbar,int) and mbar >= 2
		
		R, J = binary_dimensions(self.Z, self.binary_variables)
		dim  = self.dim_x + self.dim_p
		I    = [i for i in range(dim) if not i in self.binary_variables]

		gps = []
		for e in range( self.num_outputs ):
			gps.append([])

			for r in R:
				Jr = (J==r)

				if not np.any(Jr):
					gps[e].append(None)
					continue
				
				Zr     = self.Z[ np.ix_(Jr,  I ) ]
				Yr     = self.Y[ np.ix_(Jr, [e]) ]

				kern_list = [ GPyKernel(1, k(1, [0], "k%d" % i)) \
							for i,k in enumerate(self.kern_list) ]
				#grid_x    = [[ np.linspace(*xb, eq=True \
				#				num=self.num_inducing).tolist() \
				#				for xb in self. ]]
				grid = InducingGrid( x=self.Z[:,I], mbar=mbar, eq=True )

				kern = GriefKernel(kern_list, grid, n_eigs=1000,\
								reweight_eig_funs=False, opt_kernel_params=True)
				gp   = GPGModel(Zr, Yr, kern, noise_var=self._gp_noise_var)
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
				for k, h in zip(gp.kern.kern_list, hyp):
					k.parameters = h


	def gp_optimize (self, index=None, max_lengthscale=10, **kwargs):
		self.gp_optimise(index=index, max_lengthscale=max_lengthscale, **kwargs)

	def gp_optimise (self, index=None, max_lengthscale=10, **kwargs):
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
				gp.noise_var_constraint = 'fixed'
				# TODO: lengthscale constraint: bounded to (0, max_lengthscale)
				# Optimise
				gp.optimize(**kwargs)

		hyp = []
		for e,gps in enumerate(self.gps):
			hyp.append([])
			for r,gp in enumerate(gps):
				hyp[e].append([])
				if gp is None:
					hyp[e][0].append(None)
				else:
					for k in gp.kern.kern_list:
						hyp[e][r].append(k.parameters)
		self.hyp = hyp


	def _predict (self, xnew, p):
		znew = np.array([ x.tolist() + p.tolist() for x in xnew ])
		R, J = binary_dimensions(znew, self.binary_variables)
		dim  = self.dim_x + self.dim_p
		I    = [i for i in range(dim) if not i in self.binary_variables]
		znew = znew[:, I]

		n = len(znew)
		M = np.zeros((n, self.num_outputs))
		S = np.zeros((n, self.num_outputs))

		for r in R:
			Jr = J==r
			if not np.any(Jr):
				continue
			for e in range( self.num_outputs ):
				I          = np.ix_(Jr,[e])
				Mt, St     = self.gps[e][r].predict(znew[Jr])
				M[I], S[I] = Mt, np.diag(St)[:,None]
		return M, S


	"""
	Derivatives
	"""
	def _d_mu_d_p (self, e, X):
		R, J    = binary_dimensions(X, self.binary_variables)
		n, E, D = len(X), self.num_outputs, self.dim_p
		dx      = self.dim_x
		dmu     = np.zeros((n,D))
		for r in R:
			Jr = (J==r)
			if not np.any(Jr):
				continue
			Z  = self.get_Z(X[ Jr ])
			gp = self.gps[e][r]
			for d in range(D):
				#Phi    = self.cov_grad(gp, Z, self.dim_x-self.dim_b+d)
				I      = np.ix_(Jr, [d])
				#dmu[I] = Phi.dot(gp.kern._alpha_p)
				dmu[I] = gp.d_Yhat_d_x(Z, d)
		return dmu

	def _d2_mu_d_p2 (self, e, X):
		raise NotImplementedError

	def _d_s2_d_p (self, e, X):
		raise NotImplementedError

	def _d2_s2_d_p2 (self, e, X):
		raise NotImplementedError


	"""
	Clear model
	"""
	def clear_model (self):
		del self.gps
		super(GPGriefModel,self).clear_model()
