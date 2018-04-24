
import pytest
import numpy as np 

from GPdoemd.discrimination_criteria import chi2, aicw
from GPdoemd.discrimination_criteria import gaussian_likelihood as gl
from GPdoemd.discrimination_criteria import gaussian_likelihood_update as glu


"""
CASES
"""
def _get_equals (N=10,M=3,E=2):
	Y  = np.ones((N,E))
	mu = np.ones((N,M,E))
	s2 = 0.05 * np.array( [[np.eye(E)]*M]*N )
	return Y, mu, s2


def _get_has_winner (N=10,M=3,E=2,i=0):
	Y  = np.ones((N,E))
	mu = np.zeros((N,M,E)) + 0.1
	mu[:,i] = 1.
	s2 = 0.05 * np.array( [[np.eye(E)]*M]*N )
	return Y, mu, s2


"""
TESTS
"""
class TestDiscriminationCriteria:

	def test_gl(self):
		M = 3

		# All p(M_i) ~= 1/M
		Y,mu,s2 = _get_equals(M=M)
		P = gl(Y,mu,s2)
		assert P.shape == (M,)
		assert 0.999 < np.sum(P) < 1.001
		for p in P:
			assert np.abs(p - 1./M) < 0.001
		# Update
		P = glu(Y[-1],mu[-1],s2[-1],P)
		assert P.shape == (M,)
		assert 0.999 < np.sum(P) < 1.001
		for p in P:
			assert np.abs(p - 1./M) < 0.001

		# One p(M_i) ~= 1
		Y,mu,s2 = _get_has_winner(M=M)
		P = glu(Y[-1],mu[-1],s2[-1],P)
		assert P.shape == (M,)
		assert 0.999 < np.sum(P) < 1.001
		assert P[0] > 0.99
		for p in P[1:]:
			assert 0. <= p < 0.001
		# New start
		P = gl(Y,mu,s2)
		assert P.shape == (M,)
		assert 0.999 < np.sum(P) < 1.001
		assert P[0] > 0.99
		for p in P[1:]:
			assert 0. <= p < 0.001

	def test_aicw(self):
		M = 3
		D = np.array([2]*M)

		# All p(M_i) ~= 1/M
		Y,mu,s2 = _get_equals(M=M)
		P = aicw(Y,mu,s2,D)
		assert P.shape == (M,)
		assert 0.999 < np.sum(P) < 1.001
		for p in P:
			assert np.abs(p - 1./M) < 0.001
			
		# One p(M_i) ~= 1
		Y,mu,s2 = _get_has_winner(M=M)
		P = aicw(Y,mu,s2,D)
		assert P.shape == (M,)
		assert 0.999 < np.sum(P) < 1.001
		assert P[0] > 0.99
		for p in P[1:]:
			assert 0. <= p < 0.001

	def test_chi2(self):
		M = 3
		D = np.array([2]*M)

		""" All p(M_i) ~= 1/M """
		Y,mu,s2 = _get_equals(M=M)
		# s2 : ( N x M x E x E )
		P = chi2(Y,mu,s2,D)
		assert P.shape == (M,)
		for p in P:
			assert p > 0.1
		# s2 : ( E x E )
		P = chi2(Y,mu,s2[0,0],D)
		assert P.shape == (M,)
		for p in P:
			assert p > 0.1
		# s2 : ( E )
		P = chi2(Y,mu,np.diag(s2[0,0]),D)
		assert P.shape == (M,)
		for p in P:
			assert p > 0.1
			
		""" One p(M_i) ~= 1 """
		Y,mu,s2 = _get_has_winner(M=M)
		# s2 : ( N x M x E x E )
		P = chi2(Y,mu,s2,D)
		assert P.shape == (M,)
		assert P[0] > 0.1
		for p in P[1:]:
			assert 0. <= p < 0.01
		# s2 : ( E x E )
		P = chi2(Y,mu,s2[0,0],D)
		assert P.shape == (M,)
		assert P[0] > 0.1
		for p in P[1:]:
			assert 0. <= p < 0.01
		# s2 : ( E )
		P = chi2(Y,mu,np.diag(s2[0,0]),D)
		assert P.shape == (M,)
		assert P[0] > 0.1
		for p in P[1:]:
			assert 0. <= p < 0.01