
import unittest
import numpy as np 
from GPdoemd.model import Model

class TestClass (unittest.TestCase):

	def __init__ (self):
		self.x_bounds = np.array([[10., 20.], [5., 8.]])
		self.p_bounds = np.array([[ 2.,  4.], [3., 5.]])

		X  = np.array([[10., 5.], [10., 8.], [20., 5.], [20., 8.]])
		Xs = np.random.uniform([10, 5], [20, 8], size=(10,2))
		self.X = np.vstack(( X, Xs ))

		P  = np.array([[2., 3.], [2., 5.], [4., 3.], [4., 5.]])
		Ps = np.random.uniform([2., 3.], [4., 5.], size=(10,2))
		self.P = np.vstack(( P, Ps ))

		self.Z = np.c[self.X, self.P]

		def f (x, p):
			return x * p
		
		self.p = np.array([3., 4.])
		self.Y = f(self.X, self.p)

		self.d = {
			'name':        'testmodel',
			'call':        f,
			'x_bounds':    self.x_bounds,
			'p_bounds':    self.p_bounds,
			'num_outputs': self.Y.shape[1]
		}
		self.M = Model(self.d)

		self.M.set_training_data(self.Z, self.Y)

	def test_call(self):
		F = self.M.call(self.X, self.p)
		self.assertEqual(F, self.Y)

	def test_zmin (self):
		zmin = np.array([-1,-1])
		self.assertEqual(self.M.zmin, zmin)

	def test_zmax (self):
		zmax = np.array([1,1])
		self.assertEqual(self.M.zmax, zmax)

	def test_ymin (self):
		ymin = np.array([-1,-1])
		self.assertEqual(self.M.ymin, ymin)

	def test_ymax (self):
		ymax = np.array([1,1])
		self.assertEqual(self.M.ymax, ymax)

	def backtrans_y_min (self):
		ymin = np.array([20,15])
		yt   = self.M.backtransform_y(np.array([-1,-1]))
		self.assertEqual(yt, ymin)

	def backtrans_y_max (self):
		ymax = np.array([80,40])
		yt   = self.M.backtransform_y(np.array([1,1]))
		self.assertEqual(yt, ymax)

if __name__ == '__main__':
	unittest.main()



