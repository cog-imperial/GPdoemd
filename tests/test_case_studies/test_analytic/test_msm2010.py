
import pytest
import numpy as np 
from GPdoemd.case_studies.analytic.msm2010 import get


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
		y  = M( x )
		assert y.shape == (E,)