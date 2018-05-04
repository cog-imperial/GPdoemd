
import numpy as np 
from scipy.optimize import differential_evolution
from scipy.optimize import least_squares as lstsq


def residuals (p, model, X, Y):
	L   = np.array([model.call(x,p) for x in X]) - Y
	cov = model.meas_noise_var
	if cov.ndim == 1:
		L  = L / np.sqrt(cov)
	else:
		ch = np.linalg.cholesky( np.linalg.inv(cov) )
		L  = np.matmul(L, ch.T)
	return L
	

def diff_evol (model, X, Y, p_bounds):
	"""
	Minimisation using differential evolution
	"""
	assert p_bounds is not None, 'Parameter bounds required for param. estim.'

	def loss_function (p):
		L = residuals(p, model, X, Y)
		return np.sum( L**2 )

	res = differential_evolution(loss_function, p_bounds.tolist())
	return res['x']


def least_squares (model, X, Y, p_bounds):
	"""
	Minimisation using a least squares-method
	"""
	assert p_bounds is not None, 'Parameter bounds required for param. estim.'

	def loss_function (p):
		L = residuals(p, model, X, Y)
		return L.flatten()

	if model.pmean is not None:
		p0 = model.pmean
	elif hasattr(model,'_old_mean') and model._old_pmean is not None:
		p0 = model._old_pmean
	else:
		p0 = np.mean(p_bounds, axis=1)

	res = lstsq(loss_function, p0, bounds=p_bounds.T.tolist())
	return res['x']


