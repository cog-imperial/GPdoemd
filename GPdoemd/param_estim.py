
import numpy as np 
import scipy.optimize


def residuals (p, model, X, Y):
	L   = np.array([model(x,p) for x in X]) - Y
	cov = model.meas_noise_var
	if cov.ndim == 1:
		L  = L / np.sqrt(cov)
	else:
		ch = np.linalg.cholesky( np.linalg.inv(cov) )
		L  = np.matmul(L, ch.T)
	return L
	

def diff_evol (model, X, Y):
	"""
	Minimisation using differential evolution
	"""
	def loss_function (p):
		L = residuals(p, model, X, Y)
		return np.sum( L**2 )

	bounds = model.p_bounds.tolist()
	res    = scipy.optimize.differential_evolution(loss_function, bounds)
	return res['x']


def least_squares (model, X, Y):
	"""
	Minimisation using a least squares-method
	"""
	def loss_function (p):
		L = residuals(p, model, X, Y)
		return L.flatten()

	if model.pmean is not None:
		p0 = model.pmean
	elif hasattr(model,'_old_mean') and model._old_pmean is not None:
		p0 = model._old_pmean
	else:
		p0 = 0.5 * np.sum(model.p_bounds, axis=1)

	bounds = model.p_bounds.T.tolist()
	res    = scipy.optimize.least_squares(loss_function, p0, bounds=bounds)
	return res['x']


