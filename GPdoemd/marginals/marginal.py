
import numpy as np

#from ..utils import expand_dims

"""
Marginaliser class
"""
class Marginal:
	def __init__(self, gps, param_mean, meas_var, Xdata):
		# List of GPs, one for each output
		self.gps = gps
		self.Xdata = Xdata.copy()
		# Parameter mean
		self.m = param_mean.copy()
		# Measurement noise covariance
		self.meas_var = meas_var.copy()
		# Compute parameter covariance
		self.s = self.param_covar()

	def get_Z (self,X):
		return np.array([ x.tolist() + self.m.tolist() for x in X ])

	def d_mu_d_p (self, gp, X=None):
		if X is None:
			X = self.Xdata
		Z = self.get_Z(X)

		# d k / d p
		#dk   = np.expand_dims(gp.kern.kernx.K(Z, gp.X), 2)
		dk   = gp.kern.kernx.K(Z, gp.X)[:,:,None]
		dk   = dk * gp.kern.kernp.d_k_d_x(Z, gp.X)
		# beta := inv(K + sigma*I) * y
		beta = gp.posterior.woodbury_vector.reshape([1, len(gp.X), 1])
		# d mu / d p
		dmu  = np.sum( beta * dk, axis=1 )
		return dmu

	def d2_mu_d_p2 (self, gp, X=None):
		if X is None:
			X = self.Xdata
		Z = self.get_Z(X)

		# d^2 k / d p^2
		#ddk  = expand_dims(gp.kern.kernx.K(Z, gp.X), [2,3])
		ddk  = gp.kern.kernx.K(Z, gp.X)[:,:,None,None]
		ddk  = ddk * gp.kern.kernp.d2_k_d_x2(Z, gp.X)
		# beta := inv(K + sigma*I) * y
		beta = gp.posterior.woodbury_vector.reshape([1, len(gp.X), 1, 1])
		# d mu / d p
		ddmu = np.sum( beta * ddk, axis=1 )
		return ddmu

	def d_s2_d_p (self, gp, X=None):
		return NotImplementedError

	def d2_s2_d_p2 (self, gp, X=None):
		if X is None:
			X = self.Xdata
		Z = self.get_Z(X)
		k = gp.kern.K(Z, gp.X)

		# d k / d p
		#dk   = np.expand_dims(gp.kern.kernx.K(Z, gp.X), 2)
		dk   = gp.kern.kernx.K(Z, gp.X)[:,:,None]
		dk   = dk * gp.kern.kernp.d_k_d_x(Z, gp.X)
		# d^2 k / d p^2
		#ddk  = expand_dims(gp.kern.kernx.K(Z, gp.X), [2,3])
		ddk  = gp.kern.kernx.K(Z, gp.X)[:,:,None,None]
		ddk  = ddk * gp.kern.kernp.d2_k_d_x2(Z, gp.X)

		iK   = gp.posterior.woodbury_inv
		kiK  = np.matmul(k,iK)
		#kiK  = expand_dims(kiK,[2,3])
		kiK  = kiK[:,:,None,None]
		ddk  = np.sum(kiK * ddk, axis=1)

		dkiK   = np.matmul( np.transpose(dk,(0,2,1)), iK )
		dkiK   = np.transpose(dkiK,(0,2,1))
		#dkiK   = np.expand_dims(dkiK, 3)
		#dk     = np.expand_dims(dk, 2)
		dkiKdk = np.sum(dkiK[:,:,:,None] * dk[:,:,None,:], axis=1)
		# d^2 s2 / d p^2
		dds2ddp = -2. * ( ddk + dkiKdk )
		return dds2ddp

	
	def param_covar (self):
		# Dimensions
		E = len(self.gps)
		D = len(self.m)
		
		# Invert measurement noise covariance
		if self.meas_var.ndim == 1: 
			imeasvar = np.diag(1./self.meas_var)
		else: 
			imeasvar = np.linalg.inv(self.meas_var)
		
		# Inverse covariance matrix
		iA = np.zeros( (D, D) )

		for e1 in range(E):
			dmu1 = self.d_mu_d_x(self.gps[e1])

			iA += imeasvar[e1,e1] * np.matmul(dmu1.T,dmu1)

			if self.meas_var.ndim == 1:
				continue
			for e2 in range(e1+1,E):
				if imeasvar[e1,e2] == 0.:
					continue
				dmu2 = self.d_mu_d_x(self.gps[e2])

				iA += imeasvar[e1,e2] * np.matmul(dmu1.T,dmu2)
				iA += imeasvar[e2,e1] * np.matmul(dmu2.T,dmu1)

		return np.linalg.inv(iA)











