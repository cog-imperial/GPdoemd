
import numpy as np

"""
Marginaliser class
"""
class Marginal (object):
	def __init__(self, gps, param_mean, meas_var, Xdata):
		# List of GPs, one for each output
		self.gps = gps
		self.X = self.gps[0].X.copy()
		self.Xdata = Xdata.copy()
		# Parameter mean
		self.m = param_mean.copy()
		# Measurement noise covariance
		self.meas_var = meas_var.copy()
		# Dimensions
		self.D, self.E = len(self.m), len(gps)
		self.N, self.Ndata = len(self.X), len(self.Xdata)
		self.Dx = self.X.shape[1] - self.D
		# Compute parameter covariance
		self.s = self.param_covar()

	def d_mu_d_P (self,gp):
		# Parameter training data, and param_mean matrix
		P, pmean = gp.X[:,-self.D:], self.m*np.ones((self.Ndata,self.D))
		# Length scales
		leng_p = np.array(gp.kern.kernp.lengthscale)**2
		# Differences between param_mean and parameter training data
		z = (pmean.reshape([self.Ndata,1,self.D]) \
			- P.reshape([1,self.N,self.D]))/leng_p
		# beta := inv(K+sigma*I)*y
		beta = gp.posterior.woodbury_vector

		Zdata = np.c_[self.Xdata,pmean[:self.Ndata]]
		# Covariances
		kXP = gp.kern.K(Zdata,gp.X)
		# d k / d p
		dKdP = -kXP.reshape([self.Ndata,self.N,1])*z[:self.Ndata]
		# d mu / d p
		dmudP = np.sum(beta.reshape([1,self.N,1]) * dKdP, axis=1)
		return dmudP
	
	def param_covar (self):
		#minx,maxx,minp,maxp,miny,maxy = mins
		# Dimensions
		E, D = self.E, self.D
		
		# Invert measurement noise covariance
		if self.meas_var.ndim == 1: imeasvar = np.diag(1./self.meas_var)
		else: imeasvar = np.linalg.inv(self.meas_var)
		
		# Covariance matrix
		iA = np.zeros((D,D))
		# Outer loop
		for e1 in range(E):
			# Compute gradient matrix
			dmudP1 = self.d_mu_d_P(self.gps[e1])
			
			# Inner loop
			for e2 in range(e1,E):
				# Check if off-diagonal noise is zero
				if (self.meas_var.ndim == 1 and not e1 == e2) \
					or imeasvar[e1,e2] == 0.:
					continue
				# Compute gradient matrix
				dmudP2 = self.d_mu_d_P(self.gps[e2])

				iA += imeasvar[e1,e2]*np.matmul(dmudP1.T,dmudP2)
				if not e1 == e2:
					iA += imeasvar[e2,e1]*np.matmul(dmudP2.T,dmudP1)

		A = np.linalg.inv(iA); 
		A = 0.5 * (A + A.T)
		return A







