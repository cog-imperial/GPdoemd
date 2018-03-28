
import numpy as np

"""
Functions for computing various design criteria.
Used for experimental design for model selection.

Inputs:
	mu			[ n , M , (E) ]		Means of model output distributions.
	s2			[ n , M , (E, E) ]	Covariance matrices of model output 
										distributions.
	noise_var	[ (E), (E) ]		(Optional, float/int/ndarray) 
										Variance of measurement noise.
										If omitted, noise_var = 0
	pps			[ M ]				(Optional) Prior model probabilities.
										If omitted, p(model) = 1/M
	
	n is the number of test points.
	M is the number of different models.
	E (optional) is the number of target dimensions/measured states.

Output:
	Design criterion	[ n ]	

This code does not come with any guarantees or warranty of any kind.
Copyright (C) 2018, Simon Olofsson, simon.olofsson15@imperial.ac.uk
"""

def _reshape (mu,s2,noise_var=None,pps=None):
	""" MEAN"""
	if mu.ndim == 2:
		mu = np.expand_dims( mu, axis=2 )
	n, M, E = mu.shape

	""" NOISE VARIANCE """
	# None
	if noise_var is None: 
		noise_var = np.zeros((E,E))

	# Scalar
	if isinstance(noise_var,(int,float)):
		noise_var = np.array([noise_var])

	# Numpy vector
	if noise_var.ndim == 1:
		tmp = np.eye(E)
		np.fill_diagonal(tmp, noise_var)
		noise_var = tmp
			
	assert noise_var.shape == (E,E)
	assert np.all( np.diag(noise_var) >= 0. )

	""" COVARIANCE """
	s2  = s2.reshape( (n, M, E, E) )

	""" MODEL PROBABILITIES """
	if pps is None:
		pps = np.ones(M)

	assert pps.shape == (M,) and np.all(pps >= 0)
	pps = pps / np.sum(pps)

	# Return
	return mu, s2, noise_var, pps, n, M, E


def HR (mu,s2,pps,noise_var=None):
	"""
	Hunter and Reiner's design criterion

	- Hunter and Reiner (1965)
		Designs for discriminating between two rival models
		Technometrics 7(3):307-323
	"""
	mu, _, _, _, n, M, _ = _reshape(mu,s2,pps,noise_var)

	dc = np.zeros(n)
	for i in range(M-1):
		for j in range(i+1,M):
			dc += np.sum( (mu[:,i]-mu[:,j])**2, axis=1 )
	return dc


def BH (mu,s2,pps,noise_var=None):
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
	mu, s2, noise_var, pps, n, M, E = _reshape(mu,s2,pps,noise_var)

	s2 += noise_var
	iS  = np.linalg.inv(s2)
	dc  = np.zeros(n)
	for i in range(M-1):
		for j in range(i+1,M):
			t1 = np.trace( np.matmul(s2[:,i], iS[:,j]) \
						 + np.matmul(s2[:,j], iS[:,i]) \
						 - 2 * np.eye(E), axis1=1, axis2=2)
			r1 = np.expand_dims(mu[:,i] - mu[:,j],2) 
			t2 = np.sum(r1 * np.matmul(iS[:,i] + iS[:,j], r1), axis=(1, 2))
			dc += pps[i] * pps[j] * (t1 + t2)
	return 0.5*dc


def BF (mu,s2,pps,noise_var):
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
	mu, s2, noise_var, _, n, M, _ = _reshape(mu,s2,pps,noise_var)

	s2 += noise_var
	dc  = np.zeros(n)
	for i in range(M-1):
		for j in range(i+1,M):
			iSij = np.linalg.inv(s2[:,i] + s2[:,j])
			t1   = np.trace( np.matmul(noise_var, iSij), axis1=1, axis2=2 )
			r1   = np.expand_dims(mu[:,i] - mu[:,j],2) 
			t2   = np.sum( r1 * np.matmul(iSij, r1), axis=(1,2) )
			dc  += t1 + t2
	return dc


def AW (mu,s2,pps,noise_var=None):
	"""
	Modified Expected Akaike Weights Decision Criterion.

	- Michalak et al. (2010). 
		Optimal Experimental Design for Discriminating Numerous 
		Model Candidates: The AWDC Criterion.
		In: Ind. Eng. Chem. Res. 49:913-919
	"""
	mu, s2, noise_var, pps, n, M, _ = _reshape(mu,s2,pps,noise_var)

	iS = np.linalg.inv(s2 + noise_var)
	dc = np.zeros((n,M))
	for i in range(M):
		for j in range(M):
			r1 = np.expand_dims(mu[:,i] - mu[:,j],2) 
			t1 = np.sum(r1*np.matmul(iS[:,i],r1), axis=(1,2))
			dc[:,i] += np.exp(-0.5*t1)
	# Compute AWDC
	return np.sum( (1./dc) * pps, axis=1 )
