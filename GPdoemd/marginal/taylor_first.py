
import numpy as np 

from .gpmarginal import GPMarginal
from ..utils import binary_dimensions

"""
First-order Taylor approximation
"""
class TaylorFirstOrder (GPMarginal):
	def __init__(self, *args):
		super().__init__(*args)
	
	def __call__ (self, xnew):
		N       = len(xnew)
		E       = len(self.gps)
		D       = len(self.param_mean)
		R, I, J = binary_dimensions(xnew, self.bin_var)
		# Test points + pmean
		xnew = xnew[:,I]
		Z    = self.get_Z(xnew)

		M   = np.zeros((N,E))
		s2  = np.zeros((N,E))
		dmu = np.zeros((N,E,D))

		for e1 in range(E):
			gp = self.gps[e1]
			for r in R:
				Jr = (J==r)
				if not np.any(Jr):
					continue
				i = np.ix_(Jr,[e1])
				M[i], s2[i] = gp[r].predict(Z[Jr])
				""" d mu / d p """
				dmu[Jr,e1]  = self.d_mu_d_p(gp[r], xnew[Jr])

		# Cross-covariance terms
		S = np.zeros((N,E,E))
		for n in range(N):
			mSm  = np.matmul( dmu[n], np.matmul(self.Sigma, dmu[n].T) )
			S[n] = np.diag(s2[n]) + mSm

		for e in range(E):
			S[:,e,e] = np.maximum(1e-15,S[:,e,e])
		return M, S


