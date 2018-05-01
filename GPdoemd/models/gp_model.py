
import numpy as np 
import warnings

from GPy.models import GPRegression
from GPy.kern import Kern

from . import Model
from ..marginal import GPMarginal
from ..utils import binary_dimensions


class GPModel (Model):
	def __init__ (self, model_dict):
		super().__init__(model_dict)
		# Optional parameters
		self.gp_noise_var     = model_dict.get('gp_noise_var', 1e-6)
		self.binary_variables = model_dict.get('binary_variables', [])


	"""
	Properties
	"""
	# Measurement noise variance transformed to GP y-space
	@property
	def transformed_meas_noise_var (self):
		if self.meas_noise_var.ndim == 1:
			return self.transform_y_var(self.meas_noise_var)
		else:
			return self.transform_y_cov(self.meas_noise_var)



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
			for i in self.binary_variables:
				self._zmin[i] = 0
				self._zmax[i] = 1
			self._Z    = self.transform_z(value)
	@Z.deleter
	def Z (self):
		self._Z    = None
		self._zmin = None
		self._zmax = None
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
		return None if not hasattr(self,'_Y') else self._Y
	@Y.setter
	def Y (self, value):
		if value is not None:
			assert value.shape[1] == self.num_outputs
			self._ymin = np.min(value, axis=0)
			self._ymax = np.max(value, axis=0)
			self._Y    = self.transform_y(value)
	@Y.deleter
	def Y (self):
		self._Y    = None
		self._ymin = None
		self._ymax = None
	@property
	def ymin (self):
		return None if not hasattr(self,'_ymin') else self._ymin
	@property
	def ymax (self):
		return None if not hasattr(self,'_ymax') else self._ymax

	## Number of binary variables
	@property
	def dim_b (self):
		return len( self.binary_variables )

	def set_training_data (self, Z, Y):
		self.Z = Z
		self.Y = Y

	def clear_training_data (self):
		del self.Z
		del self.Y




	"""
	Variable, parameter and target transformations
	"""
	def _none_check (self, arglist):
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
		dim_x  = self.dim_x - self.dim_b
		dim_p  = self.dim_p
		dim    = dim_x + dim_p

		R, I, J = binary_dimensions(Z, self.binary_variables)

		assert not np.any([ value is None for value in [Z, Y, kern_x, kern_p] ])

		gps = []
		for e in range( self.num_outputs ):
			gps.append([])
			for r in R:
				Jr = (J==r)

				if not np.any(Jr):
					gps[e].append(None)
					continue

				kernx = kern_x(dim_x, range(dim_x), 'kernx')
				kernp = kern_p(dim_p, range(dim_x, dim), 'kernp')
				Zr    = Z[ np.ix_(Jr,  I ) ]
				Yr    = Y[ np.ix_(Jr, [e]) ]
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


	def gp_optimize (self, index=None):
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
						lower=0., upper=10., warning=False )
				# Constrain kern_p lengthscales
				for j in range(self.dim_p):
					gp.kern.kernp.lengthscale[[j]].constrain_bounded(
						lower=0., upper=10., warning=False )
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


	def predict (self, xnew):
		znew    = np.array([ x.tolist() + self.pmean.tolist() for x in xnew ])
		znew    = self.transform_z(znew)
		R, I, J = binary_dimensions(znew, self.binary_variables)
		znew    = znew[:,I]

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

		return self.backtransform_prediction(M,S)

	def clear_surrogate_model (self):
		del self.gps
		del self.hyp
		self.clear_training_data()
		if not self.gprm is None:
			del self.gprm




	"""
	Marginal surrogate predictions
	"""
	@property
	def gprm (self):
		return None if not hasattr(self,'_gprm') else self._gprm
	@gprm.setter
	def gprm (self, value):
		assert isinstance(value, GPMarginal)
		self._gprm = value
	@gprm.deleter
	def gprm (self):
		self._gprm = None

	def marginal_init (self, method):
		self.gprm = method( self, self.transform_p(self.pmean) )

	def marginal_compute_covar (self, Xdata):
		if self.gprm is None:
			return None
		Xdata = self.transform_x(Xdata)
		mvar  = self.transformed_meas_noise_var
		self.gprm.compute_param_covar(Xdata, mvar)

	def marginal_predict (self, xnew):
		if self.gprm is None:
			return None
		M, S = self.gprm( self.transform_x(xnew) )
		return self.backtransform_prediction(M, S)



	"""
	Save and load model
	"""
	def _get_save_dict (self):
		d = super()._get_save_dict()
		d['hyp'] = self.hyp
		d['Z']   = self._save_var('Z', self.backtransform_z)
		d['Y']   = self._save_var('Y', self.backtransform_y)
		return d

	def _load_save_dict (self, save_dict):
		super()._load_save_dict(save_dict)
		self.Z   = save_dict['Z']
		self.Y   = save_dict['Y']
		self.hyp = save_dict['hyp']

