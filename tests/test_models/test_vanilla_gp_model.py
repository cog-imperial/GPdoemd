
import random
random.seed(12345)

import pytest
import numpy as np 

from GPy.models import GPRegression
from GPy.kern import RBF

from GPdoemd.models import VanillaGPModel

"""
SET UP MODEL ONCE
"""

x_bounds = np.array([[10., 20.], [5., 8.]])
p_bounds = np.array([[ 2.,  4.], [3., 5.]])

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
ymean = np.mean( Y, axis=0 )
ystd  = np.std(  Y, axis=0 )


class Kern (RBF):
	def __init__ (self, d, drange, name):
		RBF.__init__(self, input_dim=d, active_dims=drange, name=name, ARD=True)
M.pmean = np.random.uniform(*p_bounds.T)
M.gp_surrogate(Z, Y, Kern, Kern)

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
		Mt.gp_surrogate(kern_x = Kern, kern_p = Kern)
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

	def test_gp_optimize (self):
		Mt = VanillaGPModel(d)
		Mt.set_training_data(Z, Y)
		# Initialised as None
		assert Mt.gps is None

		# Set up GP surrogate
		Mt.gp_surrogate(kern_x = Kern, kern_p = Kern)
		assert len( Mt.gps ) == 2 # Number of outputs
		for gps in Mt.gps:
			assert len( gps ) == 1 # Number of binary variables
			assert isinstance( gps[0], GPRegression )

		# GP noise variance
		assert Mt.hyp is None
		Mt.gp_optimize()
		assert Mt.hyp is not None

	def test_d_mu_d_p (self):
		for e in range( M.num_outputs ):
			der = M.d_mu_d_p(e, X)
			assert der.shape == (len(X), M.dim_p)

	def test_d2_mu_d_p2 (self):
		for e in range( M.num_outputs ):
			der = M.d2_mu_d_p2(e, X)
			assert der.shape == (len(X), M.dim_p, M.dim_p)

	def test_d_s2_d_p (self):
		for e in range( M.num_outputs ):
			der = M.d_s2_d_p(e, X)
			assert der.shape == (len(X), M.dim_p)

	def test_d2_s2_d_p2 (self):
		for e in range( M.num_outputs ):
			der = M.d2_s2_d_p2(e, X)
			assert der.shape == (len(X), M.dim_p, M.dim_p)

	def test_clear_model (self):
		Mt     = VanillaGPModel(d)
		Mt.gps = [None] * Mt.num_outputs
		assert Mt.gps is not None
		Mt.clear_model() 
		assert Mt.gps is None 