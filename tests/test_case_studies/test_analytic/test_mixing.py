
import pytest
import numpy as np 
from GPdoemd.case_studies.analytic.mixing import get


"""
TESTS
"""

class TestCaseStudy:

	def test_models(self):
		_, Ms = get()
		for M in Ms:
			assert isinstance(M.name, str)
			E = M.n_outputs
			assert isinstance(E, int)

			xb = M.x_bounds
			pb = M.p_bounds
			assert isinstance(xb, np.ndarray)
			assert isinstance(pb, np.ndarray)
			x = np.random.uniform(xb[:,0],xb[:,1])
			p = np.random.uniform(pb[:,0],pb[:,1])

			x[2] = 0
			y = M( x, p )
			assert y.shape == (E,)
			D = len(p)
			y, dy = M( x, p, grad=True )
			assert y.shape == (E,)
			assert dy.shape == (E,D)

			x[2] = 1
			y = M( x, p )
			assert y.shape == (E,)
			D = len(p)
			y, dy = M( x, p, grad=True )
			assert y.shape == (E,)
			assert dy.shape == (E,D)

	def test_datagen(self):
		M, Ms = get()
		assert isinstance(M.truemodel, int)
		assert isinstance(M.measvar, (float,np.ndarray))

		E  = Ms[0].n_outputs
		xb = Ms[0].x_bounds
		x  = np.random.uniform(xb[:,0],xb[:,1])

		x[2] = 0
		y  = M( x )
		assert y.shape == (E,)

		x[2] = 1
		y  = M( x )
		assert y.shape == (E,)

	def test_name (self):
		i   = 1
		M,_ = get(i)
		assert isinstance(M.name,str)
		assert M.name == 'M2'

	def test_overflow_protection (self):
		M     = get()[1][4]
		x, p  = np.array([1., 0.01, 0]), 0.025
		C, dC = M(x,p,grad=True)
		assert 0.999 <= C[0] <= 1.001
		assert -0.001 <= dC[0,0] <= 0.001