
from os.path import isfile

import numpy as np 
import pickle
import warnings

from GPy.models import GPRegression

from GPdoemd.kernels import Kern
from GPdoemd.marginals import Marginal

from pdb import set_trace as st

class Model:
	"""
	Model class

	Initialised using a dictionary of the form
		model_dict = {
			'name': 
			'call': 
			'v_bounds': 
			'p_bounds': 
			'num_outputs': 

			(optional)
			'noisevar' - GP noise variance
		}
	"""
	def __init__ (self, model_dict=None):
		# Read dictionnary
		self.name        = model_dict['name']
		self.call        = model_dict['call']
		self.x_bounds    = model_dict['x_bounds']
		self.p_bounds    = model_dict['p_bounds']
		self.num_outputs = model_dict['num_outputs']
		# Optional parameters
		self.gp_noise_var   = model_dict.get('gp_noise_var', 1e-6)
		self.meas_noise_var = model_dict.get('meas_noise_var', 1.)

	def __call__ (self, x, p):
		return self.call(x, p)

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

	## Design variable bounds
	@property
	def x_bounds (self):
		return self._x_bounds
	@x_bounds.setter 
	def x_bounds (self, value):
		assert value.ndim == 2 and value.shape[1] == 2
		self._x_bounds = value

	## Model parameter bounds
	@property
	def p_bounds (self):
		return self._p_bounds
	@p_bounds.setter 
	def p_bounds (self, value):
		assert value.ndim == 2 and value.shape[1] == 2
		self._p_bounds = value

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
		self._old_pmean = None if self._pmean is None else self._pmean.copy()
		self._pmean     = None

	def param_estim (self, Xdata, Ydata, method):
		self.pmean = method(self, Xdata, Ydata)



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
			assert value.shape[1] == self.dim_x + self.dim_p
			self._zmin = np.min(value, axis=0)
			self._zmax = np.max(value, axis=0)
			self._Z    = self.transform_z(value)
	@Z.deleter
	def Z (self):
		self._Z = None
	@property
	def zmin (self):
		return None if not hasattr(self,'_zmin') else self._zmin
	@property
	def zmax (self):
		return None if not hasattr(self,'_zmax') else self._zmax

	## Training design variable values
	@property
	def X (self):
		return None if self.Z is None else self.Z[:,:self.dim_x]
	@property
	def xmin (self):
		return None if self.zmin is None else self.zmin[:self.dim_x]
	@property
	def xmax (self):
		return None if self.zmax is None else self.zmax[:self.dim_x]

	## Training model parameter values
	@property
	def P (self):
		return None if self.Z is None else self.Z[:,self.dim_x:]
	@property
	def pmin (self):
		return None if self.zmin is None else self.zmin[self.dim_x:]
	@property
	def pmax (self):
		return None if self.zmax is None else self.zmax[self.dim_x:]

	## Training targets
	@property
	def Y (self):
		return self._Y
	@Y.setter
	def Y (self, value):
		if value is not None:
			assert value.shape[1] == self.num_outputs
			self._ymin = np.min(value, axis=0)
			self._ymax = np.max(value, axis=0)
			self._Y    = self.transform_y(value)
	@Y.deleter
	def Y (self):
		self._Y = None
	@property
	def ymin (self):
		return None if not hasattr(self,'_ymin') else self._ymin
	@property
	def ymax (self):
		return None if not hasattr(self,'_ymax') else self._ymax

	## Number of design variables
	@property
	def dim_x (self):
		return len( self.x_bounds )

	## Number of model parameters
	@property
	def dim_p (self):
		return len( self.p_bounds )

	def set_training_data (self, Z, Y):
		self.Z = Z
		self.Y = Y

	def clear_training_data (self):
		del self.Z
		del self.Y




	"""
	Variable, parameter and target transformations
	"""
	def _none_check (self,arglist):
		if np.any([a is None for a in arglist]):
			warnings.warn('Transform value is None')
			return True
		return False

	# Transform to [-1, 1]-space
	def transform (self, X, xmin, xmax):
		if self._none_check([X, xmin, xmax]):
			return np.NaN
		return (2 * X - xmax - xmin) / (xmax - xmin)

	# Transform to original space
	def backtransform (self, X, xmin, xmax):
		if self._none_check([X, xmin, xmax]):
			return np.NaN
		return 0.5 * (xmax + xmin + X * (xmax - xmin))

	def transform_x (self, X):
		return self.transform(X, self.xmin, self.xmax)
	def backtransform_x (self, X):
		return self.backtransform(X, self.xmin, self.xmax)
	def transform_p (self, P):
		return self.transform(P, self.pmin, self.pmax)
	def backtransform_p (self, P):
		return self.backtransform(P, self.pmin, self.pmax)
	def transform_z (self, Z):
		return self.transform(Z, self.zmin, self.zmax)
	def backtransform_z (self, Z):
		return self.backtransform(Z, self.zmin, self.zmax)
	def transform_y (self, Y):
		return self.transform(Y, self.ymin, self.ymax)
	def backtransform_y (self, Y):
		return self.backtransform(Y, self.ymin, self.ymax)

	# Transform to [-1, 1]-space
	def transform_var (self, C, xmin, xmax):
		if self._none_check([C, xmin, xmax]):
			return np.NaN
		return 4 * C / (xmax - xmin)**2

	# Transform to original space
	def backtransform_var (self, C, xmin, xmax):
		if self._none_check([C, xmin, xmax]):
			return np.NaN
		return 0.25 * C * (xmax - xmin)**2

	# Transform to [-1, 1]-space
	def transform_cov (self, C, xmin, xmax):
		if self._none_check([C, xmin, xmax]):
			return np.NaN
		m = xmax - xmin
		return 4 * C / (m[:,None] * m[None,:])

	# Transform to original space
	def backtransform_cov (self, C, xmin, xmax):
		if self._none_check([C, xmin, xmax]):
			return np.NaN
		m = xmax - xmin
		return 0.25 * C * (m[:,None] * m[None,:])

	def transform_p_var (self, C):
		return self.transform_var(C, self.pmin, self.pmax)
	def backtransform_p_var (self, C):
		return self.backtransform_var(C, self.pmin, self.pmax)
	def transform_p_cov (self, C):
		return self.transform_cov(C, self.pmin, self.pmax)
	def backtransform_p_cov (self, C):
		return self.backtransform_cov(C, self.pmin, self.pmax)
	def transform_y_var (self, C):
		return self.transform_var(C, self.ymin, self.ymax)
	def backtransform_y_var (self, C):
		return self.backtransform_var(C, self.ymin, self.ymax)
	def transform_y_cov (self, C):
		return self.transform_cov(C, self.ymin, self.ymax)
	def backtransform_y_cov (self, C):
		return self.backtransform_cov(C, self.ymin, self.ymax)

	def backtransform_prediction (self, M, S):
		M = self.backtransform_y(M)
		if S.ndim == 2:
			S = self.backtransform_y_var(S)
		else:
			S = self.backtransform_y_cov(S)
		return M, S




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

	## Surrogate model hyperparameters
	@property
	def hyp (self):
		return None if not hasattr(self,'_hyp') else self._hyp
	@hyp.setter
	def hyp (self, value):
		if value is not None:
			assert len(value) == self.num_outputs
			assert np.all( [np.all(v > 0) for v in value] )
			self._hyp = value
	@hyp.deleter
	def hyp (self):
		self._hyp = None




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
		Z = self.Z
		Y = self.Y

		self.set_kernels(kern_x, kern_p)
		kern_x = self.kern_x
		kern_p = self.kern_p

		assert not np.any([ value is None for value in [Z, Y, kern_x, kern_p] ])

		gps = []
		for e in range( self.num_outputs ):
			dim   = self.dim_x + self.dim_p
			kernx = kern_x(self.dim_x, range(self.dim_x), 'kernx')
			kernp = kern_p(self.dim_p, range(self.dim_x, dim), 'kernp')
			gp    = GPRegression(Z, Y[:,[e]], kernx * kernp)
			gps.append(gp)
		self.gps = gps

	def gp_load_hyp (self, index=None):
		if index is None:
			index = range( self.num_outputs )
		elif isinstance(index, int):
			index = [index]

		for e in index:
			gp = self.gps[e]
			gp.update_model(False)
			gp.initialize_parameter()
			gp[:] = self.hyp[e]
			gp.update_model(True)
			self.gps[e] = gp

	def gp_optimize (self, index=None):
		if index is None:
			index = range( self.num_outputs )
		elif isinstance(index, int):
			index = [index]

		for e in index:
			gp = self.gps[e]
			# Constrain noise variance
			gp.Gaussian_noise.variance.constrain_fixed(self._gp_noise_var)
			# Constrain kern_x lengthscales
			for j in range(self.dim_x):
				gp.kern.kernx.lengthscale[[j]].constrain_bounded(
					lower=0., upper=10., warning=False )
			# Constrain kern_p lengthscales
			for j in range(self.dim_p):
				gp.kern.kernp.lengthscale[[j]].constrain_bounded(
					lower=0., upper=10., warning=False )
			# Optimise
			gp.optimize()
			self.gps[e] = gp

	def predict (self, xnew):
		znew = np.array([ x.tolist() + self.pmean.tolist() for x in xnew ])
		znew = self.transform_z(znew)

		n = len(znew)
		M = np.zeros((n, self.num_outputs))
		S = np.zeros((n, self.num_outputs))
		
		for e in range( self.num_outputs ):
			M[:,[e]], S[:,[e]] = self.gps[e].predict(znew)

		return self.backtransform_prediction(M,S)




	"""
	Marginal surrogate predictions
	"""
	@property
	def gprm (self):
		return None if not hasattr(self,'_gprm') else self._gprm
	@gprm.setter
	def gprm (self, value):
		assert isinstance(value, Marginal)
		self._gprm = value
	@gprm.deleter
	def gprm (self):
		self._gprm = None

	def marginal_init (self, method):
		assert issubclass(method, Marginal)
		self.gprm = method( self.gps, self.transform_p(self.pmean) )

	def marginal_compute_covar (self, Xdata):
		if self.gprm is None:
			return None
		Xdata = self.transform_x(Xdata)

		if self.meas_noise_var.ndim == 1:
			meas_noise_var = self.transform_y_var(self.meas_noise_var)
		else:
			meas_noise_var = self.transform_y_cov(self.meas_noise_var)

		self.gprm.compute_param_covar(Xdata, meas_noise_var)

	def marginal_init_and_compute_covar (self, method, Xdata):
		self.marginal_init(method)
		self.marginal_compute_covar(Xdata)

	def marginal_predict (self, xnew):
		if self.gprm is None:
			return None
		xnew = self.transform_x(xnew)
		M, S = self.gprm(xnew)
		return self.backtransform_prediction(M, S)




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

	def save (self, filename):
		assert isinstance(filename, str)
		d      = {'hyp':   self.hyp, 
				  'pmean': self.pmean}

		d['Z']         = self._save_var('Z', self.backtransform_z)
		d['Y']         = self._save_var('Y', self.backtransform_y)
		d['old_pmean'] = self._save_var('_old_pmean')
		# Filename ending
		suffix = '.gpdoemd.model'
		lensuf = len(suffix)
		if len(filename) <= lensuf or not filename[-lensuf:] == suffix:
			filename += suffix
		# Save in pickle
		with open(filename,'wb') as f:
			pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

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
			d = pickle.load(f)
		# Variables
		self.Z          = d['Z']
		self.Y          = d['Y']
		self.hyp        = d['hyp']
		self.pmean      = d['pmean']
		self._old_pmean = d['old_pmean']



