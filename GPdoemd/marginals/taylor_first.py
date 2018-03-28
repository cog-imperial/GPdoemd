
import numpy as np 

from .marginal import Marginal

"""
First-order Taylor approximation
"""
class TaylorFirstOrder (Marginal):
	def __init__(self, gps, param_mean, meas_var, Xdata):
		Marginal.__init__(self, gps, param_mean, meas_var, Xdata)

	def d_mu_s2_d_P (self,gp,xnew=None):
		if xnew is None: 
			xnew = self.Xdata
		# Parameter training data, and param_mean matrix
		P, pmean = gp.X[:,-self.D:], self.m*np.ones((xnew.shape[0],self.D))
		# Length scales
		leng_p = np.array(gp.kern.kernp.lengthscale)**2
		# Differences between param_mean and parameter training data
		z = (pmean.reshape([xnew.shape[0],1,self.D]) - \
			 P.reshape([1,self.N,self.D]))/leng_p
		# beta := inv(K+sigma*I)*y
		beta = gp.posterior.woodbury_vector
		# Predictions
		Z = np.c_[xnew,pmean]
		mu, s2 = gp.predict(Z)
		# Covariances
		kxP = gp.kern.K(Z,gp.X)
		# d k / d p
		dKdp = -kxP.reshape([xnew.shape[0],self.N,1])*z[:xnew.shape[0]]
		# d mu / d p
		dmudp = np.sum(beta.reshape([1,self.N,1]) * dKdp, axis=1)
		return mu[:,0], s2[:,0], dmudp
	
	def __call__ (self,xnew):
		n, E, D = xnew.shape[0], self.E, self.D
		M, s2, dmu = np.zeros((n,E)), np.zeros((n,E)), np.zeros((n,E,D))
		for e1 in range(E):
			M[:,e1], s2[:,e1], dmu[:,e1,:] = self.d_mu_s2_d_P(self.gps[e1],xnew)
		S = np.array([ np.diag(s2[i]) + \
						np.matmul(dmu[i],np.matmul(self.s,dmu[i].T)) \
						for i in range(len(xnew))])

		for e in range(self.E):
			S[:,e,e] = np.maximum(1e-15,S[:,e,e])
		return M, S