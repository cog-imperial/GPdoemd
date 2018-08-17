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
import warnings

from . import Model
from ..transform import BoxTransform, MeanTransform


class SurrogateModel (Model):
	def __init__ (self, model_dict):
		super().__init__(model_dict)
		# Optional parameters
		self.binary_variables = model_dict.get('binary_variables', [])

	"""
	Properties
	"""
	@property
	def binary_variables (self):
		return self._binary_variables
	@binary_variables.setter
	def binary_variables (self, value):
		if isinstance(value, list):
			for v in value:
				assert isinstance(v, int), 'Value not integer'
				assert 0 <= v < self.dim_x, 'Value outside range'
			self._binary_variables = value
		elif isinstance (value, int):
			assert 0 <= value < self.dim_x, 'Value outside range'
			self._binary_variables = [value]
		else:
			raise ValueError('Binary variable must be list or integer')

	@property
	def non_binary_variables (self):
		return [i for i in range(self.dim_x) if not i in self.binary_variables]

	## Number of binary variables
	@property
	def dim_b (self):
		return len( self.binary_variables )

	"""
	Surrogate model training data
	"""
	## Training variable [X, P]
	@property
	def Z (self):
		return None if not hasattr(self,'_Z') else self._Z
	@Z.setter
	def Z (self, value):
		if value is not None:
			assert value.shape[1] == self.dim_x + self.dim_p, 'Incorrect shape'
			self._zmin = np.min(value, axis=0)
			self._zmax = np.max(value, axis=0)
			for i in self.binary_variables:
				self._zmin[i] = 0
				self._zmax[i] = 1
			self.trans_z = BoxTransform(self.zmin, self.zmax)
			self.trans_x = BoxTransform(self.xmin, self.xmax)
			self.trans_p = BoxTransform(self.pmin, self.pmax)
			self._Z      = self.trans_z(value)
	@Z.deleter
	def Z (self):
		if self.Z is not None:
			del self._zmin
			del self._zmax
			del self.trans_z
			del self.trans_x
			del self.trans_p
		self._Z = None
	@property
	def zmin (self):
		return self._zmin
	@property
	def zmax (self):
		return self._zmax

	def get_Z (self, X, P=None):
		if P is None:
			assert self.pmean is not None
			P = np.asarray([self.trans_p(self.pmean)] * len(X))
		assert P.shape == ( len(X), self.dim_p )
		return np.array([ x.tolist() + p.tolist() for x,p in zip(X,P) ])

	## Training design variable values
	@property
	def X (self):
		return None if self.Z is None else self.Z[:,:self.dim_x]
	@property
	def xmin (self):
		return self.zmin[:self.dim_x]
	@property
	def xmax (self):
		return self.zmax[:self.dim_x]

	## Training model parameter values
	@property
	def P (self):
		return None if self.Z is None else self.Z[:,self.dim_x:]
	@property
	def pmin (self):
		return self.zmin[self.dim_x:]
	@property
	def pmax (self):
		return self.zmax[self.dim_x:]

	## Training targets
	@property
	def Y (self):
		return None if not hasattr(self,'_Y') else self._Y
	@Y.setter
	def Y (self, value):
		if value is not None:
			assert value.shape[1] == self.num_outputs
			self._ymean  = np.mean(value, axis=0)
			self._ystd   = np.std(value, axis=0)
			self.trans_y = MeanTransform(self.ymean, self.ystd)
			#self._Y      = self.transform_y(value)
			self._Y      = self.trans_y(value)
	@Y.deleter
	def Y (self):
		if self.Y is not None:
			del self._ymean
			del self._ystd
			del self.trans_y
		self._Y = None
	@property
	def ymean (self):
		return self._ymean
	@property
	def ystd (self):
		return self._ystd


	def set_training_data (self, Z, Y, _Y=None):
		if _Y is None:
			self.Z = Z
			self.Y = Y
		else:
			self.Z = np.c_[Z, Y]
			self.Y = _Y

	"""
	Prediction
	"""
	def predict (self, xnew):
		assert self.pmean is not None
		assert self.Z is not None and self.Y is not None
		assert self.hyp is not None
		xt   = self.trans_x(xnew)
		pt   = self.trans_p(self.pmean)
		M, S = self._predict(xt, pt)
		#return self.backtransform_prediction(M, S)
		return self.trans_y.prediction(M, S, back=True)
	def _predict (self, *args):
		raise NotImplementedError

	"""
	Derivatives
	"""
	def d_mu_d_p (self, e, X):
		#xnew = self.transform_x(X)
		xnew = self.trans_x(X)
		der  = self._d_mu_d_p(e, xnew)
		return der * self.ystd[e] / (self.pmax - self.pmin)
	def _d_mu_d_p (self, e, X):
		raise NotImplementedError

	def d2_mu_d_p2 (self, e, X):
		#xnew = self.transform_x(X)
		xnew = self.trans_x(X)
		der  = self._d2_mu_d_p2(e, xnew)
		diff = (self.pmax - self.pmin)
		return der * self.ystd[e] / (diff[:,None] * diff[None,:])
	def _d2_mu_d_p2 (self, e, X):
		raise NotImplementedError

	def d_s2_d_p (self, e, X):
		#xnew = self.transform_x(X)
		xnew = self.trans_x(X)
		der  = self._d_s2_d_p(e, xnew)
		return der * self.ystd[e]**2 / (self.pmax - self.pmin)
	def _d_s2_d_p (self, e, X):
		raise NotImplementedError

	def d2_s2_d_p2 (self, e, X):
		#xnew = self.transform_x(X)
		xnew = self.trans_x(X)
		der  = self._d2_s2_d_p2(e, xnew)
		diff = (self.pmax - self.pmin)
		return der * self.ystd[e]**2 / (diff[:,None] * diff[None,:])
	def _d2_s2_d_p2 (self, e, X):
		raise NotImplementedError


	"""
	Surrogate model hyperparameters
	"""
	@property
	def hyp (self):
		return None if not hasattr(self,'_hyp') else self._hyp
	@hyp.setter
	def hyp (self, value):
		if value is not None:
			# Should be list with hyperparameters for each output
			# num_outputs x num_bin_var x num_hyperparameters
			assert len(value) == self.num_outputs
			self._hyp = value
	@hyp.deleter
	def hyp (self):
		self._hyp = None


	"""
	Clear model
	"""
	def clear_model (self):
		del self.Z
		del self.Y
		del self.hyp
		super(SurrogateModel,self).clear_model()

	"""
	Save and load model
	"""
	def _get_save_dict (self):
		d = super()._get_save_dict()
		d['hyp'] = self.hyp
		d['Z']   = self._save_var('Z', lambda z: self.trans_z(z, back=True))
		d['Y']   = self._save_var('Y', lambda y: self.trans_y(y, back=True))
		return d

	def _load_save_dict (self, save_dict):
		super()._load_save_dict(save_dict)
		self.Z   = save_dict['Z']
		self.Y   = save_dict['Y']
		self.hyp = save_dict['hyp']

