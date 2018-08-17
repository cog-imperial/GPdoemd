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
from ..utils import binary_dimensions
from ..transform import BoxTransform, MeanTransform


class SurrogateModel (Model):
	def __init__ (self, model_dict):
		super().__init__(model_dict)
		# Optional parameters
		self.binary_variables = model_dict.get('binary_variables', [])

	"""
	Properties
	"""
	# Measurement noise variance transformed to GP y-space
	"""
	@property
	def transformed_meas_noise_var (self):
		if self.meas_noise_var.ndim == 1:
			#return self.transform_y_var(self.meas_noise_var)
			return self.trans_y.var(self.meas_noise_var)
		else:
			#return self.transform_y_cov(self.meas_noise_var)
			return self.trans_y.cov(self.meas_noise_var)
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
		#return None if not hasattr(self,'_Z') else self._Z
		return self._Z
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
			#self._Z     = self.transform_z(value)
			self._Z      = self.trans_z(value)
	@Z.deleter
	def Z (self):
		#self._Z    = None
		#self._zmin = None
		#self._zmax = None
		del self._Z   
		del self._zmin
		del self._zmax
		del self.trans_z
		del self.trans_x
		del self.trans_p
	@property
	def zmin (self):
		#return None if not hasattr(self,'_zmin') else self._zmin
		return self._zmin
	@property
	def zmax (self):
		#return None if not hasattr(self,'_zmax') else self._zmax
		return self._zmax

	def get_Z (self, X, P=None):
		if P is None:
			assert self.pmean is not None
			#P = np.asarray([self.transform_p(self.pmean)] * len(X))
			P = np.asarray([self.trans_p(self.pmean)] * len(X))
		assert P.shape == ( len(X), self.dim_p )
		return np.array([ x.tolist() + p.tolist() for x,p in zip(X,P) ])

	## Training design variable values
	@property
	def X (self):
		#return None if self.Z is None else self.Z[:,:self.dim_x]
		return self.Z[:,:self.dim_x]
	@property
	def xmin (self):
		#return None if self.zmin is None else self.zmin[:self.dim_x]
		return self.zmin[:self.dim_x]
	@property
	def xmax (self):
		#return None if self.zmax is None else self.zmax[:self.dim_x]
		return self.zmax[:self.dim_x]

	## Training model parameter values
	@property
	def P (self):
		#return None if self.Z is None else self.Z[:,self.dim_x:]
		return self.Z[:,self.dim_x:]
	@property
	def pmin (self):
		#return None if self.zmin is None else self.zmin[self.dim_x:]
		return self.zmin[self.dim_x:]
	@property
	def pmax (self):
		#return None if self.zmax is None else self.zmax[self.dim_x:]
		return self.zmax[self.dim_x:]

	## Training targets
	@property
	def Y (self):
		#return None if not hasattr(self,'_Y') else self._Y
		return self._Y
	@Y.setter
	def Y (self, value):
		#if value is not None:
		assert value.shape[1] == self.num_outputs
		self._ymean  = np.mean(value, axis=0)
		self._ystd   = np.std(value, axis=0)
		self.trans_y = MeanTransform(self.ymean, self.ystd)
		#self._Y      = self.transform_y(value)
		self._Y      = self.trans_y(value)
	@Y.deleter
	def Y (self):
		#self._Y     = None
		#self._ymean = None
		#self._ystd  = None
		del self._Y    
		del self._ymean
		del self._ystd
		del self.trans_y
	@property
	def ymean (self):
		#return None if not hasattr(self,'_ymean') else self._ymean
		return self._ymean
	@property
	def ystd (self):
		#return None if not hasattr(self,'_ystd') else self._ystd
		return self._ystd


	def set_training_data (self, Z, Y, _Y=None):
		if _Y is None:
			self.Z = Z
			self.Y = Y
		else:
			self.Z = np.c_[Z, Y]
			self.Y = _Y


	"""
	Variable, parameter and target transformations
	"""
	"""
	def _none_check (self, arglist):
		if np.any([a is None for a in arglist]):
			return True
		return False

	## Transform input to interval [0,1]
	def box_trans (self, X, xmin, xmax, reverse=False):
		if reverse:
			return xmin + X * (xmax - xmin)
		return (X - xmin) / (xmax - xmin)

	def box_var_trans (self, X, xmin, xmax, reverse=False):
		m = xmax - xmin
		if reverse:
			return X * m**2
		return X / m**2

	def box_cov_trans (self, X, xmin, xmax, reverse=False):
		m = xmax - xmin
		if reverse:
			return X * (m[:,None] * m[None,:])
		return X / (m[:,None] * m[None,:])


	# Transform such that input has mean zero
	def mean_trans (self, X, mean, std, reverse=False):
		if reverse:
			return mean + X * std
		return (X - mean) / std

	def mean_var_trans (self, X, mean, std, reverse=False):
		if reverse:
			return X * std**2
		return X / std**2

	def mean_cov_trans (self, X, mean, std, reverse=False):
		if reverse:
			return X * (std[:,None] * std[None,:])
		return X / (std[:,None] * std[None,:])


	# Transform to transform-space
	def transform (self, trans, X, x1, x2, reverse=False):
		if self._none_check([X, x1, x2]):
			warnings.warn('Transform value is None')
			return np.NaN
		return trans( X, x1, x2, reverse=reverse )

	# Transform back to original space
	def backtransform (self, trans, X, x1, x2):
		return self.transform(trans, X, x1, x2, reverse=True)
	
	## Different transforms
	def transform_x (self, X):
		return self.transform(self.box_trans, X, self.xmin, self.xmax)

	def backtransform_x (self, X):
		return self.backtransform(self.box_trans, X, self.xmin, self.xmax)

	def transform_p (self, P):
		return self.transform(self.box_trans, P, self.pmin, self.pmax)
	
	def backtransform_p (self, P):
		return self.backtransform(self.box_trans, P, self.pmin, self.pmax)
	
	def transform_z (self, Z):
		return self.transform(self.box_trans, Z, self.zmin, self.zmax)
	
	def backtransform_z (self, Z):
		return self.backtransform(self.box_trans, Z, self.zmin, self.zmax)
	
	def transform_y (self, Y):
		return self.transform(self.mean_trans, Y, self.ymean, self.ystd)
	
	def backtransform_y (self, Y):
		return self.backtransform(self.mean_trans, Y, self.ymean, self.ystd)

	def transform_p_var (self, C):
		return self.transform(self.box_var_trans, C, self.pmin, self.pmax)
	
	def backtransform_p_var (self, C):
		return self.backtransform(self.box_var_trans, C, self.pmin, self.pmax)
	
	def transform_p_cov (self, C):
		return self.transform(self.box_cov_trans, C, self.pmin, self.pmax)
	
	def backtransform_p_cov (self, C):
		return self.backtransform(self.box_cov_trans, C, self.pmin, self.pmax)
	
	def transform_y_var (self, C):
		return self.transform(self.mean_var_trans, C, self.ymean, self.ystd)
	
	def backtransform_y_var (self, C):
		return self.backtransform(self.mean_var_trans, C, self.ymean, self.ystd)
	
	def transform_y_cov (self, C):
		return self.transform(self.mean_cov_trans, C, self.ymean, self.ystd)
	
	def backtransform_y_cov (self, C):
		return self.backtransform(self.mean_cov_trans, C, self.ymean, self.ystd)

	def backtransform_prediction (self, M, S):
		M = self.backtransform_y(M)
		if S.ndim == 2:
			S = self.backtransform_y_var(S)
		else:
			S = self.backtransform_y_cov(S)
		return M, S
	"""

	"""
	Prediction
	"""
	def predict (self, xnew):
		#xt   = self.transform_x(xnew)
		#pt   = self.transform_p(self.pmean)
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
		#return None if not hasattr(self,'_hyp') else self._hyp
		return self._hyp
	@hyp.setter
	def hyp (self, value):
		if value is not None:
			# Should be list with hyperparameters for each output
			# num_outputs x num_bin_var x num_hyperparameters
			assert len(value) == self.num_outputs
			self._hyp = value
	@hyp.deleter
	def hyp (self):
		#self._hyp = None
		del self._hyp

		
	"""
	Model parameter covariance
	"""
	@property
	def Sigma_trans (self):
		#return None if not hasattr(self,'_Sigma_trans') else self._Sigma_trans
		return self._Sigma_trans
	@Sigma_trans.setter
	def Sigma_trans (self, value):
		assert isinstance(value, np.ndarray)
		assert value.shape == (self.dim_p, self.dim_p)
		self._Sigma_trans = value.copy()
	@Sigma_trans.deleter
	def Sigma_trans (self):
		#self._Sigma_trans = None
		del self._Sigma_trans


	"""
	Clear model
	"""
	def clear_model (self):
		del self.Z
		del self.Y
		del self.hyp
		del self.Sigma
		super(SurrogateModel,self).clear_model()

	"""
	Save and load model
	"""
	def _get_save_dict (self):
		d = super()._get_save_dict()
		d['hyp']         = self.hyp
		d['Z']           = self._save_var('Z', self.backtransform_z)
		d['Y']           = self._save_var('Y', self.backtransform_y)
		d['Sigma_trans'] = self.Sigma_trans
		return d

	def _load_save_dict (self, save_dict):
		super()._load_save_dict(save_dict)
		self.Z           = save_dict['Z']
		self.Y           = save_dict['Y']
		self.hyp         = save_dict['hyp']
		self.Sigma_trans = save_dict['Sigma_trans']

