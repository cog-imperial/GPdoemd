
import numpy as np 

from .marginal import Marginal

"""
Second-order Taylor approximation
"""
class TaylorSecondOrder (Marginal):
	def __init__(self, gps, param_mean, meas_var, Xdata):
		Marginal.__init__(self, gps, param_mean, meas_var, Xdata)
	
	def predictions_and_grads (self,gp,xnew=None):
		if xnew is None: xnew = self.Xdata
		# Dimensions
		n, D, N = xnew.shape[0], self.D, self.N
		# Parameter training data, and param_mean matrix
		P, pmean = gp.X[:,-D:], self.m*np.ones((n,D))
		# Length scales
		leng_p = np.array(gp.kern.kernp.lengthscale)**2
		# Differences between param_mean and parameter training data
		z = (pmean.reshape((n,1,D))-P.reshape((1,N,D)))/leng_p
		
		# Test points + pmean
		Z = np.c_[xnew,pmean]
		""" mu and s2 """
		mu, s2 = gp.predict(Z)
		
		# Covariance function
		k = gp.kern.K(Z,gp.X)
		# d k / d p
		dkdp = -k.reshape((n,N,1))*z
		# beta := inv(K+sigma*I)*y
		beta = gp.posterior.woodbury_vector.reshape((1,N,1))
		""" d mu / d p """
		dmu = np.sum(beta*dkdp, axis=1)
		
		# d^2 k / d p^2
		ddkddp = k.reshape((n,N,1,1)) * \
					(z.reshape((n,N,D,1))*z.reshape((n,N,1,D)) \
					- np.diag(1./leng_p).reshape((1,1,D,D)))
		
		# beta := inv(K+sigma*I)*y
		beta = beta.reshape((1,N,1,1))
		""" d^2 mu / d p^2 * A """
		ddmuA = np.matmul(np.sum(beta*ddkddp,axis=1), self.s)
		
		iK = gp.posterior.woodbury_inv
		kiK = np.matmul(k,iK)
		dkiK = np.matmul(np.transpose(dkdp,(0,2,1)),iK)
		dkiK = np.transpose(dkiK,(0,2,1)).reshape((n,N,D,1))
		dkiKdk = np.sum(dkiK*dkdp.reshape((n,N,1,D)), axis=1)
		# d^2 s2 / d p^2
		dds2ddp = -2.*np.sum(kiK.reshape((n,N,1,1))*ddkddp, axis=1) \
				  - 2.*dkiKdk
		""" d^2 s2 / d p^2 * A """
		dds2A = np.matmul(dds2ddp, self.s)
		
		return mu[:,0], s2[:,0], dmu, ddmuA, dds2A
	
	
	def __call__ (self,xnew):
		# Dimensions
		n, E, D = xnew.shape[0], self.E, self.D
		# mean and variance at {xnew,pmean}
		M, s2 = np.zeros((n,E)), np.zeros((n,E))
		# Gradients and Hessians of mu
		dmu, ddmuA = np.zeros((n,E,D)), np.zeros((n,E,D,D))
		
		# Mean and variance
		for e1 in range(E):
			# Predictions and gradient information
			M[:,e1],s2[:,e1],dmu[:,e1,:],ddmuA[:,e1,:,:],dds2A = \
				self.predictions_and_grads(self.gps[e1],xnew)
			# Update mean and variance
			M[:,e1] += 0.5*np.trace(ddmuA[:,e1,:,:],axis1=1,axis2=2)
			s2[:,e1] += 0.5*np.trace(dds2A,axis1=1,axis2=2)
			
		# Cross-covariance terms
		S = np.zeros((n,E,E))
		for n in range(xnew.shape[0]):
			S[n,:,:] = np.diag(s2[n]) + \
						np.matmul(dmu[n],np.matmul(self.s,dmu[n].T))
			for e1 in range(E):
				S[n,e1,e1] += 0.5 * np.trace(
					np.matmul(ddmuA[n,e1,:,:],ddmuA[n,e1,:,:]))
				for e2 in range(e1+1,E):
					t = 0.5 * np.trace(
						np.matmul(ddmuA[n,e1,:,:],ddmuA[n,e2,:,:]))
					S[n,e1,e2] += t; S[n,e2,e1] += t
				# Safety check
				S[n,e1,e1] = np.maximum(1e-15, S[n,e1,e1])

		return M, S