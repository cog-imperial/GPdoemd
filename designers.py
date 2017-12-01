
import numpy as np


class Designer:
	"""
	Functions for computing various design criteria.
	Used for experimental design for model selection.

	Inputs:
		mu			[ n , M , (E) ]		Means of model output distributions.
		s2			[ n , M , (E, E) ]	Covariance matrices of model output 
											distributions.
		pps			[ M ]				(Optional) Prior model probabilities.
											If omitted, p(model) = 1/M
		noise_var	[ (E) ]				(Optional, float/int/ndarray) 
											Variance of measurement noise.
											If omitted, noise_var = 0
		
		n is the number of test points.
		M is the number of different models.
		N is the number of observations.
		E (optional) is the number of output dimensions/observable states.

	Output:
		Design criterion	[ n ]	

	Note:
	This code does not come with any guarantees or warranty of any kind.
	Copyright (C) 2017, Simon Olofsson, simon.olofsson15@imperial.ac.uk
	"""
	def __init__ (self,mu,s2,pps=None,noise_var=None):
		"""
		Check that inputs have correct types and dimensions
		"""
		assert isinstance(mu,np.ndarray) and isinstance(s2,np.ndarray)
		assert (mu.ndim == 2 or mu.ndim == 3)
		self.mu, self.s2 = mu.copy(), s2.copy()

		# Matrix/tensor shapes
		if self.mu.ndim == 2:
			self.mu = self.mu.reshape((self.mu.shape[0],self.mu.shape[1],1))
		self.n,self.M,self.E = self.mu.shape
		self.s2 = self.s2.reshape((self.n,self.M,self.E,self.E))

		"""
		Noise variance
		"""
		# No noise variance given
		if noise_var is None: 
			noise_var = 0.
		# Noise variance given as a scalar number
		if isinstance(noise_var,float) or isinstance(noise_var,int):
			assert noise_var >= 0.
			# Turn it into diagonal matrix
			noise_var = noise_var * np.eye(self.E)
		# Noise variance given as vector
		if isinstance(noise_var,np.ndarray):
			if noise_var.ndim == 1:
				assert np.all(noise_var>=0.)
				if noise_var.shape == (1,):
					self.noise_var = noise_var * np.eye(self.E)
				elif noise_var.shape == (self.E,):
					self.noise_var = np.diag(noise_var)
				else:
					raise ValueError('Measurement noise variance ' + \
						'has incorrect shape: ' + str(noise_var.shape))
			elif noise_var.ndim == 2:
				assert noise_var.shape == (self.E,self.E)
				self.noise_var = 0.5 * (noise_var + noise_var.T)
				assert np.all(np.diag(self.noise_var) >= 0.)
		else:
			raise ValueError('Noise variance must be given as ' + \
				'scalar number or numpy array')
		# Add measurement noise variance to the general variance 
		self.s2 += self.noise_var
		self.is2 = None

		# Prior probabilities
		if pps is None:
			pps = 1./self.M * np.ones(self.M)
		else:
			assert isinstance(pps,np.ndarray) and np.all(pps >= 0)
			pps = pps / np.sum(pps)
		assert len(pps)==self.M and pps.ndim==1,
		self.pps = pps.copy()

	def __call__ (self,name):
		"""
		Retrieve design criterion by name
		"""
		if name == 'HR': func = self._HR
		elif name == 'BH': func = self._BH
		elif name == 'BF': func = self._BF
		elif name == 'AW': func = self._AW
		else:
			raise ValueError('Unknown design criterion\n' + \
				'Implemented design criteria are: HR, BH, BF, AW')
		return func(self.mu,self.s2,self.pps,self.noise_var)

	def _HR (self,mu,s2,pps,noise_var=None):
		"""
		Hunter and Reiner's design criterion

		- Hunter and Reiner (1965)
			Designs for discriminating between two rival models
			Technometrics 7(3):307-323
		"""
		dc = np.zeros(self.n)
		for i in range(self.M-1):
			for j in range(i+1,self.M):
				dc += np.sum( (mu[:,i]-mu[:,j])**2, axis=1 )
		return dc

	def _BH (self,mu,s2,pps,noise_var=None):
		"""
		Box and Hill's design criterion, extended to multiresponse 
		models by Prasad and Someswara Rao.

		- Box and Hill (1967)
			Discrimination among mechanistic models
			Technometrics 9(1):57-71
		- Prasad and Someswara Rao (1977)
			Use of expected likelihood in sequential model 
			discrimination in multiresponse systems.
			Chem. Eng. Sci. 32:1411-1418
		"""
		iS = np.linalg.inv(s2)
		dc = np.zeros(self.n)
		for i in range(self.M-1):
			for j in range(i+1,self.M):
				t1 = np.trace( np.matmul(s2[:,i],iS[:,j]) \
							 + np.matmul(s2[:,j],iS[:,i]) \
							 - 2 * np.eye(self.E),axis1=1,axis2=2)
				r1 = np.expand_dims(mu[:,i] - mu[:,j],2) 
				t2 = np.sum(r1*np.matmul(iS[:,i] + iS[:,j],r1),axis=(1,2))
				dc += pps[i]*pps[j]*(t1+t2)
		return 0.5*dc

	def _BF (self,mu,s2,pps,noise_var):
		"""
		Buzzi-Ferraris et al.'s design criterion.

		- Buzzi-Ferraris and Forzatti (1983)
			Sequential experimental design for model discrimination 
			in the case of multiple responses
			Chem. Eng. Sci. 39(1):81-85
		- Buzzi-Ferraris et al. (1984)
			Sequential experimental design for model discrimination 
			in the case of multiple responses
			Chem. Eng. Sci. 39(1):81-85
		- Buzzi-Ferraris et al. (1990)
			An improved version of sequential design criterion for 
			discrimination among rival multiresponse models
			Chem. Eng. Sci. 45(2):477-481
		"""
		dc = np.zeros(self.n)
		for i in range(self.M-1):
			for j in range(i+1,self.M):
				iSij = np.linalg.inv(s2[:,i]+s2[:,j])
				t1 = np.trace( np.matmul(noise_var,iSij),axis1=1,axis2=2)
				r1 = np.expand_dims(mu[:,i] - mu[:,j],2) 
				t2 = np.sum(r1*np.matmul(iSij,r1),axis=(1,2))
				dc += t1+t2
		return dc

	def _AW (self,mu,s2,pps,noise_var=None):
		"""
		Modified Expected Akaike Weights Decision Criterion.

		- Michalak et al. (2010). 
			Optimal Experimental Design for Discriminating Numerous 
			Model Candidates: The AWDC Criterion.
			In: Ind. Eng. Chem. Res. 49:913-919
		"""
		iS = np.linalg.inv(s2)
		# Compute expected Akaike weights
		dc = np.zeros((self.n,self.M))
		for i in range(self.M):
			for j in range(self.M):
				r1 = np.expand_dims(mu[:,i] - mu[:,j],2) 
				t1 = np.sum(r1*np.matmul(iS[:,i],r1),axis=(1,2))
				dc[:,i] += np.exp(-0.5*t1)
		# Compute AWDC
		return np.sum( (1./dc) * pps, axis=1 )
