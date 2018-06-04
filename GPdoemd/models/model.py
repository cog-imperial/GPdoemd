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

from os.path import isfile

import numpy as np 
import pickle

from ..marginal import Analytic, Numerical

from pdb import set_trace as st

class Model:
	def __init__ (self, model_dict):
		# Read dictionnary
		self.name        = model_dict.get('name','model')
		self.call        = model_dict['call']
		self.dim_x       = model_dict['dim_x']
		self.dim_p       = model_dict['dim_p']
		self.num_outputs = model_dict.get('num_outputs',1)
		# Optional parameters
		self.p_bounds         = model_dict.get('p_bounds', [])
		self.meas_noise_var   = model_dict.get('meas_noise_var', 1.)
		self.binary_variables = []

	"""
	Properties
	"""
	## Model name
	@property
	def name (self):
		return self._name
	@name.setter 
	def name (self, value):
		assert isinstance(value, str)
		self._name = value

	## Model function handle
	@property
	def call (self):
		return self._call
	@call.setter 
	def call (self, value):
		assert callable(value)
		self._call = value

	## Number of outputs/target dimensions
	@property
	def num_outputs (self):
		return self._num_outputs
	@num_outputs.setter 
	def num_outputs (self, value):
		assert isinstance(value, int) and value > 0
		self._num_outputs = value

	## Measurement noise variance
	@property
	def meas_noise_var (self):
		return self._meas_noise_var
	@meas_noise_var.setter
	def meas_noise_var (self, value):
		if isinstance(value, (int, float)):
			value = np.array([value] * self.num_outputs)
		assert isinstance(value, np.ndarray)
		assert np.all( value > 0. )
		self._meas_noise_var = value
	@property
	def meas_noise_covar (self):
		if self.meas_noise_var.ndim == 1:
			return np.diag( self.meas_noise_var )
		else:
			return self.meas_noise_var

	## Number of design variables
	@property
	def dim_x (self):
		return self._dim_x
	@dim_x.setter
	def dim_x (self, value):
		assert isinstance(value, int) and value > 0
		self._dim_x = value

	## Number of model parameters
	@property
	def dim_p (self):
		return self._dim_p
	@dim_p.setter
	def dim_p (self, value):
		assert isinstance(value, int) and value > 0
		self._dim_p = value



	"""
	Parameter estimation
	"""	
	## Best-fit model parameter values
	@property
	def pmean (self):
		return None if not hasattr(self,'_pmean') else self._pmean
	@pmean.setter
	def pmean (self, value):
		if value is not None:
			assert value.shape == (self.dim_p,)
		self._pmean = value
		if not hasattr(self,'_old_pmean'):
			self._old_pmean = None
	@pmean.deleter
	def pmean (self):
		self._old_pmean = None if self.pmean is None else self.pmean.copy()
		self._pmean     = None
		self.pmean      = None

	def param_estim (self, Xdata, Ydata, method, p_bounds=None):
		self.pmean = method(self, Xdata, Ydata, p_bounds)

	# Model prediction
	def predict (self, xnew, p=None):
		if p is None:
			p = self.pmean
		M = np.array([self.call(x, p) for x in xnew])
		S = np.zeros(M.shape)
		return M, S





	"""
	Marginal predictions
	"""
	@property
	def gprm (self):
		return None if not hasattr(self,'_gprm') else self._gprm
	@gprm.setter
	def gprm (self, value):
		assert isinstance(value, (Numerical, Analytic))
		self._gprm = value
	@gprm.deleter
	def gprm (self):
		self._gprm = None

	def marginal_init (self, method):
		self.gprm = method( self, self.pmean )

	def marginal_compute_covar (self, Xdata):
		if self.gprm is None:
			return None
		mvar = self.meas_noise_var
		self.gprm.compute_param_covar(Xdata, mvar)

	def marginal_init_and_compute_covar (self, method, Xdata):
		self.marginal_init(method)
		self.marginal_compute_covar(Xdata)

	def marginal_predict (self, xnew):
		if self.gprm is None:
			return None
		return self.gprm(xnew)




	"""
	Save and load model
	"""
	def _save_var (self, value, operation=None):
		if not hasattr(self, value):
			return None 
		value = eval('self.' + value)
		if value is None:
			return None
		return value if operation is None else operation(value)

	def _get_save_dict (self):
		#'probability': self.probability
		return {
				'pmean':       self.pmean,
		        'old_pmean':   self._save_var('_old_pmean')
		        }

	def _load_save_dict (self, save_dict):
		self.pmean        = save_dict['pmean']
		self._old_pmean   = save_dict['old_pmean']

	def save (self, filename):
		assert isinstance(filename, str)
		# Filename ending
		suffix = '.gpdoemd.model'
		lensuf = len(suffix)
		if len(filename) <= lensuf or not filename[-lensuf:] == suffix:
			filename += suffix
		# Save in pickle
		save_dict = self._get_save_dict()
		with open(filename,'wb') as f:
			pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)

	def load (self, filename):
		assert isinstance(filename, str)
		# Filename ending
		suffix = '.gpdoemd.model'
		lensuf = len(suffix)
		if len(filename) <= lensuf or not filename[-lensuf:] == suffix:
			filename += suffix
		# Load file
		assert isfile(filename)
		with open(filename,'rb') as f:
			save_dict = pickle.load(f)
		self._load_save_dict(save_dict)



