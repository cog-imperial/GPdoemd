
import random
random.seed(12345)

import pytest
import numpy as np 

import gp_grief.models
from GPy.kern import RBF

from GPdoemd.models import GPGriefModel

"""
SET UP MODEL ONCE
"""

x_bounds = np.array([[10., 20.], [5., 8.]])
p_bounds = np.array([[ 2.,  4.], [3., 5.]])

def f (x, p):
	return x * p

d = {
	'name':        'testmodel',
	'call':        f,
	'dim_x':       len(x_bounds),
	'dim_p':       len(p_bounds),
	'num_outputs': 2
}
M = GPGriefModel(d)


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
kernlist = [Kern,] * (M.dim_x + M.dim_p)

M.pmean = np.random.uniform(*p_bounds.T)
M.gp_surrogate(Z, Y, kernlist)

class TestGPGriefModel:

	"""
	Test GP surrogate
	"""
	def test_gp_grief_surrogate (self):
		Mt = GPGriefModel(d)
		Mt.set_training_data(Z, Y)
		# Initialised as None
		assert Mt.gps is None

		# Set up GP surrogate
		Mt.gp_surrogate(kern_list = kernlist)
		assert len( Mt.gps ) == 2 # Number of outputs
		for gps in Mt.gps:
			assert len( gps ) == 1 # Number of binary variables
			assert isinstance( gps[0], gp_grief.models.GPGriefModel )

		# GP noise variance
		assert isinstance( Mt.gp_noise_var, float)

		# Hyperparameters
		assert Mt.hyp is None
		hyps  = [[[ (i + 0.1) * (j + 0.5) * k.parameters 
					for k in gp.kern.kern_list] \
					for j,gp in enumerate(gps)] \
					for i,gps in enumerate(Mt.gps)]
		Mt.hyp = hyps
		assert Mt.hyp is not None

		Mt.gp_load_hyp()
		for hyp, gps in zip( hyps, Mt.gps ):
			for h, gp in zip( hyp, gps ):
				for i,k in enumerate(gp.kern.kern_list):
					assert np.all( h[i] == k.parameters )

		Mt.pmean = np.array([3., 4.])
		M,S = Mt.predict(Xs)
		assert M.shape == (len(Xs),2)
		assert S.shape == (len(Xs),2)

	def test_gp_grief_optimize (self):
		Mt = GPGriefModel(d)
		Mt.set_training_data(Z, Y)
		# Initialised as None
		assert Mt.gps is None

		# Set up GP surrogate
		Mt.gp_surrogate(kern_list = kernlist)
		assert len( Mt.gps ) == 2 # Number of outputs
		for gps in Mt.gps:
			assert len( gps ) == 1 # Number of binary variables
			assert isinstance( gps[0], gp_grief.models.GPGriefModel )

		# GP noise variance
		assert Mt.hyp is None
		Mt.gp_optimize(max_iters=4)
		assert Mt.hyp is not None

	def test_d_mu_d_p (self):
		for e in range( M.num_outputs ):
			der = M.d_mu_d_p(e, X)
			assert der.shape == (len(X), M.dim_p)
	
	"""
	def test_d2_mu_d_p2 (self):
		return

	def test_d_s2_d_p (self):
		return

	def test_d2_s2_d_p2 (self):
		return
	"""

	def test_clear_model (self):
		Mt     = GPGriefModel(d)
		Mt.gps = [None] * Mt.num_outputs
		assert Mt.gps is not None
		Mt.clear_model() 
		assert Mt.gps is None

