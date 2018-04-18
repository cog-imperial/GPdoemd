
import pytest
import numpy as np 
from GPdoemd.models import GPModel

from pdb import set_trace as st

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
	'x_bounds':    x_bounds,
	'p_bounds':    p_bounds,
	'num_outputs': 2
}
M = GPModel(d)


X  = np.array([[10., 5.], [10., 8.], [20., 5.], [20., 8.]])
Xs = np.random.uniform([10, 5], [20, 8], size=(10,2))
X = np.vstack(( X, Xs ))

P  = np.array([[2., 3.], [2., 5.], [4., 3.], [4., 5.]])
Ps = np.random.uniform([2., 3.], [4., 5.], size=(10,2))
P = np.vstack(( P, Ps ))

Z = np.c_[X, P]
Y = f(X, P)
M.set_training_data(Z, Y)


"""
TESTS
"""

class TestGPModel:

	def test_call(self):
		p = np.array([3., 4.])
		F = M.call(X, p)
		assert np.all(F == f(X, p))

	def test_trans_z_min (self):
		zmin = np.array([-1,-1,-1,-1])
		assert np.all( np.min(M.Z, axis=0) == zmin )

	def test_trans_z_max (self):
		zmax = np.array([ 1, 1, 1, 1])
		assert np.all( np.max(M.Z, axis=0) == zmax )

	def test_trans_x_min (self):
		xmin = np.array([-1,-1])
		assert np.all( np.min(M.X, axis=0) == xmin )

	def test_trans_x_max (self):
		xmax = np.array([ 1, 1])
		assert np.all( np.max(M.X, axis=0) == xmax )

	def test_trans_p_min (self):
		pmin = np.array([-1,-1])
		assert np.all( np.min(M.P, axis=0) == pmin )

	def test_trans_p_max (self):
		pmax = np.array([ 1, 1])
		assert np.all( np.max(M.P, axis=0) == pmax )

	def test_trans_y_min (self):
		ymin = np.array([-1,-1])
		assert np.all( np.min(M.Y, axis=0) == ymin )

	def test_trans_y_max (self):
		ymax = np.array([ 1, 1])
		assert np.all( np.max(M.Y, axis=0) == ymax )

	def test_backtrans_y_min (self):
		ymin = np.array([20,15])
		yt   = M.backtransform_y( np.array([-1,-1]) )
		assert np.all(yt == ymin)

	def test_backtrans_y_max (self):
		ymax = np.array([80,40])
		yt   = M.backtransform_y( np.array([ 1, 1]) )
		assert np.all(yt == ymax)
