
import pytest
import numpy as np 
from GPdoemd.case_studies.ode.vthr2014ode import get


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
			x = np.random.uniform(*xb.T)
			p = np.random.uniform(*pb.T)
			y = M( x, p )
			assert y.shape == (E,)

	def test_datagen(self):
		M, Ms = get()
		assert isinstance(M.truemodel, int)
		assert isinstance(M.measvar, (float, np.ndarray))

		E  = Ms[0].n_outputs
		xb = Ms[0].x_bounds
		x  = np.random.uniform(*xb.T)
		y  = M( x )
		assert y.shape == (E,)