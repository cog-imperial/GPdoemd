
import random
random.seed(12345)

import pytest
import numpy as np 

from GPy.models import GPRegression

from GPdoemd.models import VanillaGPModel
from GPdoemd.kernels import RBF
#from GPdoemd.marginal import TaylorFirstOrder

"""
SET UP MODEL ONCE
"""

x_bounds = np.array([[10., 20.], [5., 8.]])
p_bounds = np.array([[ 2.,  4.], [3., 5.]])
z_bounds = np.array( x_bounds.tolist() + p_bounds.tolist() )

def f (x, p):
	return x * p

#ymin = np.array([20,15])
#ymax = np.array([80,40])

d = {
	'name':        'testmodel',
	'call':        f,
	'dim_x':       len(x_bounds),
	'dim_p':       len(p_bounds),
	'num_outputs': 2
}
M = VanillaGPModel(d)


X  = np.array([[10., 5.], [10., 8.], [20., 5.], [20., 8.]])
Xs = np.random.uniform([10, 5], [20, 8], size=(10,2))
X = np.vstack(( X, Xs ))

P  = np.array([[2., 3.], [2., 5.], [4., 3.], [4., 5.]])
Ps = np.random.uniform([2., 3.], [4., 5.], size=(10,2))
P = np.vstack(( P, Ps ))

Z = np.c_[X, P]
Y = f(X, P)
M.set_training_data(Z, Y)

ymean = np.mean( Y, axis=0 )
ystd  = np.std(  Y, axis=0 )


"""
TESTS
"""

class TestVanillaGPModel:

	"""
	Test GP surrogate
	"""
	def test_gp_surrogate (self):
		Mt = VanillaGPModel(d)
		Mt.set_training_data(Z, Y)
		# Initialised as None
		assert Mt.gps is None

		# Set up GP surrogate
		Mt.gp_surrogate(kern_x = RBF, kern_p = RBF)
		assert len( Mt.gps ) == 2 # Number of outputs
		for gps in Mt.gps:
			assert len( gps ) == 1 # Number of binary variables
			assert isinstance( gps[0], GPRegression )

		# GP noise variance
		assert isinstance( Mt.gp_noise_var, float)

		# Hyperparameters
		assert Mt.hyp is None
		hyps  = [[ (i + 0.1) * (j + 0.5) * gp[:] for j,gp in enumerate(gps)] \
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


	"""
	Test GP optimise
	"""
	def test_gp_optimize (self):
		Mt = VanillaGPModel(d)
		Mt.set_training_data(Z, Y)
		# Initialised as None
		assert Mt.gps is None

		# Set up GP surrogate
		Mt.gp_surrogate(kern_x = RBF, kern_p = RBF)
		assert len( Mt.gps ) == 2 # Number of outputs
		for gps in Mt.gps:
			assert len( gps ) == 1 # Number of binary variables
			assert isinstance( gps[0], GPRegression )

		# GP noise variance
		assert Mt.hyp is None
		Mt.gp_optimize()
		assert Mt.hyp is not None


	"""
	Marginal surrogate
	"""
	"""
	def test_gprm (self):
		p = np.array([3., 4.])

		Mt = VanillaGPModel(d)
		Mt.pmean = np.array([3., 4.])
		assert Mt.gprm is None
		Mt.gp_surrogate(Z=Z, Y=Y, kern_x=RBF, kern_p=RBF)
		res = Mt.marginal_predict(Xs)
		assert res is None
		Mt.marginal_init_and_compute_covar(TaylorFirstOrder, Xs)
		M,S = Mt.marginal_predict(Xs)
		assert M.shape == (len(Xs),2)
		assert S.shape == (len(Xs),2,2)

		# Clear surrogate model
		Mt.clear_surrogate_model()
		assert Mt.Z is None
		assert Mt.Y is None
		assert Mt.gps is None
		assert Mt.hyp is None
		assert Mt.gprm is None
	"""



