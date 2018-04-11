
import numpy as np

from ..utils import binary_dimensions

"""
Marginaliser class
"""
class GPMarginal:
	def __init__(self, model, param_mean):
		self.gps        = model.gps               # List of GPs
		self.param_mean = param_mean              # Parameter mean
		self.bin_var    = model.binary_variables  # Binary variable indices
		
	@property
	def Sigma (self):
		return None if not hasattr(self,'_Sigma') else self._Sigma
	@Sigma.setter
	def Sigma (self, value):
		assert isinstance(value, np.ndarray)
		dim_p = len(self.param_mean)
		assert value.shape == (dim_p, dim_p)
		self._Sigma = value

	def get_Z (self, X):
		return np.array([ x.tolist() + self.param_mean.tolist() for x in X ])

	def d_mu_d_p (self, gp, X):
		Z = self.get_Z(X)

		# d k / d p
		dk   = gp.kern.kernx.K(Z, gp.X)[:,:,None]
		dk   = dk * gp.kern.kernp.d_k_d_x(Z, gp.X)
		# beta := inv(K + sigma*I) * y
		beta = gp.posterior.woodbury_vector.reshape([1, len(gp.X), 1])
		# d mu / d p
		dmu  = np.sum( beta * dk, axis=1 )
		return dmu

	def d2_mu_d_p2 (self, gp, X):
		Z = self.get_Z(X)

		# d^2 k / d p^2
		ddk  = gp.kern.kernx.K(Z, gp.X)[:,:,None,None]
		ddk  = ddk * gp.kern.kernp.d2_k_d_x2(Z, gp.X)
		# beta := inv(K + sigma*I) * y
		beta = gp.posterior.woodbury_vector.reshape([1, len(gp.X), 1, 1])
		# d mu / d p
		ddmu = np.sum( beta * ddk, axis=1 )
		return ddmu

	def d_s2_d_p (self, gp, X):
		return NotImplementedError

	def d2_s2_d_p2 (self, gp, X):
		Z = self.get_Z(X)
		k = gp.kern.K(Z, gp.X)

		# d k / d p
		dk   = gp.kern.kernx.K(Z, gp.X)[:,:,None]
		dk   = dk * gp.kern.kernp.d_k_d_x(Z, gp.X)
		# d^2 k / d p^2
		ddk  = gp.kern.kernx.K(Z, gp.X)[:,:,None,None]
		ddk  = ddk * gp.kern.kernp.d2_k_d_x2(Z, gp.X)

		iK   = gp.posterior.woodbury_inv
		kiK  = np.matmul(k,iK)
		kiK  = kiK[:,:,None,None]
		ddk  = np.sum(kiK * ddk, axis=1)

		dkiK   = np.matmul( np.transpose(dk,(0,2,1)), iK )
		dkiK   = np.transpose(dkiK,(0,2,1))
		dkiKdk = np.sum(dkiK[:,:,:,None] * dk[:,:,None,:], axis=1)
		# d^2 s2 / d p^2
		dds2ddp = -2. * ( ddk + dkiKdk )
		return dds2ddp
	
	def compute_param_covar (self, Xdata, meas_noise_var):
		# Dimensions
		E       = len(self.gps)
		D       = len(self.param_mean)
		R, I, J = binary_dimensions(Xdata, self.bin_var)
		Xdata   = Xdata[:, I] 

		if isinstance(meas_noise_var, (int, float)):
			meas_noise_var = np.array([meas_noise_var] * E)
		
		# Invert measurement noise covariance
		if meas_noise_var.ndim == 1: 
			imeasvar = np.diag(1./meas_noise_var)
		else: 
			imeasvar = np.linalg.inv(meas_noise_var)
		
		# Inverse covariance matrix
		iA = np.zeros( (D, D) )

		def get_d_mu_d_p (gps, X):
			dmu = np.zeros((len(X),D))
			for r in R:
				Jr = (J==r)
				if not np.any(Jr):
					continue
				dmu[Jr] = self.d_mu_d_p(gps[r], X[Jr])
			return dmu

		for e1 in range(E):
			dmu1 = get_d_mu_d_p(self.gps[e1], Xdata)

			iA += imeasvar[e1,e1] * np.matmul(dmu1.T, dmu1)

			if meas_noise_var.ndim == 1:
				continue
			for e2 in range(e1+1,E):
				if imeasvar[e1,e2] == 0.:
					continue
				dmu2 = get_d_mu_d_p(self.gps[e2], Xdata)

				iA += imeasvar[e1,e2] * np.matmul(dmu1.T, dmu2)
				iA += imeasvar[e2,e1] * np.matmul(dmu2.T, dmu1)

		self.Sigma = np.linalg.inv(iA)


