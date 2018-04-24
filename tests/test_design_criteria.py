
import pytest
import numpy as np 
from GPdoemd.design_criteria import _reshape, HR, BH, BF, AW, JR

from pdb import set_trace as st

N = 15
M = 3
E = 2

mu  = np.random.randn(N,M,E)
s2  = 0.1 * np.random.rand(N,M,E,E)
s2 += s2.transpose(0,1,3,2) + np.array([[np.eye(E)]*M]*N)
noisevar = 0.1 * np.eye(E)

pps = np.ones( M ) / M

# Set biggest divergence
mu[13] = ( 1 + np.arange(M*E).reshape((M,E)) ) * 10
s2[13] = np.array( [ 0.001 * np.eye(E)] * M )


"""
TESTS
"""

class TestDesignCriteria:

	def test_HR(self):
		d = HR(mu,s2,noisevar,pps)
		assert d.shape == (N,)
		assert np.argmax(d) == 13

	def test_BH(self):
		d = BH(mu,s2,noisevar,pps)
		assert d.shape == (N,)
		assert np.argmax(d) == 13

	def test_BF(self):
		d = BF(mu,s2,noisevar,pps)
		assert d.shape == (N,)
		assert np.argmax(d) == 13

	def test_AW(self):
		d = AW(mu,s2,noisevar,pps)
		assert d.shape == (N,)
		assert np.argmax(d) == 13

	def test_JR(self):
		d = JR(mu,s2,noisevar,pps)
		assert d.shape == (N,)
		assert np.argmax(d) == 13

	def test_reshape(self):
		e = 1
		m = np.random.randn(N,M)
		s = np.random.randn(N,M)
		R = _reshape(m,s)
		# mu, s2, noise_var, pps, n, M, E
		assert R[0].shape == (N,M,e)
		assert R[1].shape == (N,M,e,e)
		assert R[2].shape == (e,e)
		assert R[3].shape == (M,)
		assert 0.999 < np.sum(R[3]) < 1.001
		assert R[4] == N
		assert R[5] == M
		assert R[6] == e

		R = _reshape(m,s,1,[1.]*M)
		assert R[2].shape == (e,e)
		assert R[3].shape == (M,)
		assert 0.999 < np.sum(R[3]) < 1.001

		R = _reshape(mu,s2,np.ones(E))
		assert R[2].shape == (E,E)