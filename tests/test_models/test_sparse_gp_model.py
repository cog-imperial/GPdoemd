
import pytest
import numpy as np 
import warnings

from GPy.models import SparseGPRegression

from GPdoemd.models import SparseGPModel
from GPdoemd.kernels import RBF

"""
SET UP MODEL ONCE
"""

x_bounds = np.array([[10., 20.], [5., 8.]])
p_bounds = np.array([[ 2.,  4.], [3., 5.]])
z_bounds = np.array( x_bounds.tolist() + p_bounds.tolist() )

def f (x, p):
	return x * p

ymin = np.array([20,15])
ymax = np.array([80,40])

d = {
	'name':        'testmodel',
	'call':        f,
	'dim_x':       len(x_bounds),
	'dim_p':       len(p_bounds),
	'num_outputs': 2
}


X  = np.array([[10., 5.], [10., 8.], [20., 5.], [20., 8.]])
Xs = np.random.uniform([10, 5], [20, 8], size=(10,2))
X = np.vstack(( X, Xs ))

P  = np.array([[2., 3.], [2., 5.], [4., 3.], [4., 5.]])
Ps = np.random.uniform([2., 3.], [4., 5.], size=(10,2))
P = np.vstack(( P, Ps ))

Z = np.c_[X, P]
Y = f(X, P)


"""
TESTS
"""

class TestGPModel:

	"""
	Test GP surrogate
	"""
	def test_gp_surrogate (self):
		Mt = SparseGPModel(d)
		Mt.set_training_data(Z, Y)
		# Initialised as None
		assert Mt.gps is None

		# Set up GP surrogate
		Mt.gp_surrogate(kern_x = RBF, kern_p = RBF, num_inducing=10)
		assert len( Mt.gps ) == 2 # Number of outputs
		for gps in Mt.gps:
			assert len( gps ) == 1 # Number of binary variables
			assert isinstance( gps[0], SparseGPRegression )

		# GP noise variance
		assert isinstance( Mt.gp_noise_var, float) 

		# Hyperparameters
		assert Mt.hyp is None
		hyps  = [[ (i+0.1)*(j+0.5)*gp[:] for j,gp in enumerate(gps)] \
					for i,gps in enumerate(Mt.gps)]
		Mt.hyp = hyps
		assert Mt.hyp is not None

		Mt.gp_load_hyp()
		for hyp, gps in zip( hyps, Mt.gps ):
			for h, gp in zip( hyp, gps ):
				assert np.all( h == gp[:] )

		Mt.pmean = np.array([3., 4.])
		M,S = Mt.predict(Xs)
		assert M.shape == (len(Xs),2)
		assert S.shape == (len(Xs),2)
