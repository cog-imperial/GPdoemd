
import numpy as np 
from scipy.optimize import differential_evolution as diffevol

class ParameterEstimator:
	def __init__ (self,model,weights,optimiser=None,verbose=True):
		# Model
		self.model = model
		# Weights
		self.W = weights
		assert isinstance(self.W,np.ndarray) or self.W is None, \
				'Weights must be of type numpy array'
		# Optimiser
		self.optimiser = optimiser
		# Verbosity
		self.verbose = verbose
		assert isinstance(self.verbose,bool), \
				'Variable \'verbose\' must be of type bool'
		# Error maximum, given to loss function at computational failure
		self.error_max = 1e8 

	def __call__ (self,Y,X):
		# Number of data points and model outputs
		n_data, E = Y.shape
		# Check weights
		W = self._none_weights(Y)

		# Define residual function
		def residuals (p):
			try:
				with stdout_redirected():
					y = np.array([self.model(x,p) for x in X])
			except:
				return self.error_max
			# non-NaN indices
			ind = []
			for i in range(n_data):
				if not np.any(np.isnan(Y[i])):
					ind.append(i)
			return W*(y[ind]-Y[ind])

		# Define loss function
		def lossfunc (p,*args):
			if p.ndim == 1:
				return np.sum(residuals(p)**2)
			return np.array([np.sum(residuals(pi)**2) for pi in p])

		# Optimise loss function
		if self.verbose: 
			print('    Parameter estimation for ' + self.model.name)
		fun = self.error_max
		failures = -1
		while fun > 0.5 * self.error_max and failures < 100:
			failures += 1
			if self.optimiser is not None:
				fun,popt = self.optimiser(lossfunc,self.model)
			else:
				res = diffevol(lossfunc, self.model.p_bounds.tolist())
				fun,popt = res['fun'],np.array(res['x'])
		return popt

	def _none_weights (self,Y):
		if self.W is not None: return np.sqrt(self.W)
		return np.ones(Y.shape[1])/(1.+np.abs(Y[0])) \
				if Y.shape[0] == 1 else 1./np.nanstd(Y,axis=0)











