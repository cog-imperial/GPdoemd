
import numpy as np 

from .marginal import Marginal

"""
Second-order Taylor approximation
"""
class TaylorSecondOrder (Marginal):
	def __init__(self, gps, param_mean, meas_var, Xdata):
		super().__init__(gps, param_mean, meas_var, Xdata)
	
	def __call__ (self,xnew):
		# Dimensions
		N = len(xnew)
		E = len(self.gps)
		D = len(self.m)

		# Test points + pmean
		Z = self.get_Z(xnew)

		# mean and variance at {xnew,pmean}
		M, s2 = np.zeros((N,E)), np.zeros((N,E))
		# Gradients and Hessians of mu
		dmu, ddmuA = np.zeros((N,E,D)), np.zeros((N,E,D,D))
		
		# Mean and variance
		for e1 in range(E):
			gp = self.gps[e1]
			""" d mu / d p """
			dmu[:,e1] = self.d_mu_d_p(gp, xnew)
			""" d^2 mu / d p^2 * S_p """
			ddmu        = self.d2_mu_d_p2(gp, xnew)
			ddmuA[:,e1] = np.matmul(ddmu, self.s)
			""" trace ( d^2 s2 / d p^2 * S_p ) """
			dds2    = self.d2_s2_d_p2(gp, xnew)
			trdds2A = np.sum(dds2 * self.s, axis=(1,2))

			tmp = gp.predict(Z)
			M[:,e1]  = tmp[0][:,0] + 0.5 * np.trace(ddmuA[:,e1],axis1=1,axis2=2)
			s2[:,e1] = tmp[1][:,0] + 0.5 * trdds2A
			
		# Cross-covariance terms
		S = np.zeros((N,E,E))
		for n in range(N):
			mSm  = np.matmul( dmu[n], np.matmul(self.s, dmu[n].T) )
			S[n] = np.diag(s2[n]) + mSm
						
			for e1 in range(E):
				for e2 in range(e1,E):
					S[n,e1,e2] += 0.5 * np.sum(ddmuA[n,e1] * ddmuA[n,e2].T)
					if not e1 == e2:
						S[n,e2,e1] = S[n,e1,e2]
				# Safety check
				S[n,e1,e1] = np.maximum(1e-15, S[n,e1,e1])

		return M, S



