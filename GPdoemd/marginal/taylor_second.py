
import numpy as np 

from . import GPMarginal
from ..utils import binary_dimensions

"""
Second-order Taylor approximation
"""
class TaylorSecondOrder (GPMarginal):
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

		M     = np.zeros((N,E))
		s2    = np.zeros((N,E))
		dmu   = np.zeros((N,E,D))
		ddmu  = np.zeros((N,D,D))
		dds2  = np.zeros((N,D,D))
		ddmuA = np.zeros((N,E,D,D))
		
		for e1 in range(E):
			gp = self.gps[e1]

			for r in R:
				Jr = (J==r)
				if not np.any(Jr):
					continue
				i = np.ix_(Jr,[e1])
				M[i], s2[i] = gp[r].predict_noiseless(Z[Jr])
				""" d mu / d p """
				dmu[Jr,e1]  = self.d_mu_d_p(gp[r], xnew[Jr])
				""" d^2 mu / d p^2 """
				ddmu[Jr]    = self.d2_mu_d_p2(gp[r], xnew[Jr])
				""" d^2 s2 / d p^2 """
				dds2[Jr]    = self.d2_s2_d_p2(gp[r], xnew[Jr])

			""" d^2 mu / d p^2 * S_p """
			ddmuA[:,e1] = np.matmul(ddmu, self.Sigma)
			""" trace ( d^2 s2 / d p^2 * S_p ) """
			trdds2A     = np.sum(dds2 * self.Sigma, axis=(1,2))

			M[:,e1]  += 0.5 * np.trace(ddmuA[:,e1],axis1=1,axis2=2)
			s2[:,e1] += 0.5 * trdds2A
			
		# Cross-covariance terms
		S = np.zeros((N,E,E))
		for n in range(N):
			mSm  = np.matmul( dmu[n], np.matmul(self.Sigma, dmu[n].T) )
			S[n] = np.diag(s2[n]) + mSm
						
			for e1 in range(E):
				for e2 in range(e1,E):
					S[n,e1,e2] += 0.5 * np.sum(ddmuA[n,e1] * ddmuA[n,e2].T)
					if not e1 == e2:
						S[n,e2,e1] = S[n,e1,e2]
				# Safety check
				S[n,e1,e1] = np.maximum(1e-15, S[n,e1,e1])

		return M, S

