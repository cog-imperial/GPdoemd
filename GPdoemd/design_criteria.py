"""
MIT License

Copyright (c) 2018 Simon Olofsson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
"""

def HR (mu, s2, noise_var=None, pps=None):
	"""
	Hunter and Reiner's design criterion

	- Hunter and Reiner (1965)
		Designs for discriminating between two rival models.
		Technometrics 7(3):307-323
	"""
	mu, _, _, _, n, M, _ = _reshape(mu,s2,None,None)

	dc = np.zeros(n)
	for i in range(M-1):
		for j in range(i+1,M):
			dc += np.sum( (mu[:,i]-mu[:,j])**2, axis=1 )
	return dc


def BH (mu, s2, noise_var=None, pps=None):
	"""
	Box and Hill's design criterion, extended to multiresponse 
	models by Prasad and Someswara Rao.

	- Box and Hill (1967)
		Discrimination among mechanistic models.
		Technometrics 9(1):57-71
	- Prasad and Someswara Rao (1977)
		Use of expected likelihood in sequential model 
		discrimination in multiresponse systems.
		Chem. Eng. Sci. 32:1411-1418
	"""
	mu, s2, noise_var, pps, n, M, E = _reshape(mu, s2, noise_var, pps)

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


def BF (mu, s2, noise_var=None, pps=None):
	"""
	Buzzi-Ferraris et al.'s design criterion.

	- Buzzi-Ferraris and Forzatti (1983)
		Sequential experimental design for model discrimination 
		in the case of multiple responses.
		Chem. Eng. Sci. 39(1):81-85
	- Buzzi-Ferraris et al. (1984)
		Sequential experimental design for model discrimination 
		in the case of multiple responses.
		Chem. Eng. Sci. 39(1):81-85
	- Buzzi-Ferraris et al. (1990)
		An improved version of sequential design criterion for 
		discrimination among rival multiresponse models.
		Chem. Eng. Sci. 45(2):477-481
	"""
	mu, s2, noise_var, _, n, M, _ = _reshape(mu, s2, noise_var, None)

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


def AW (mu, s2, noise_var=None, pps=None):
	"""
	Modified Expected Akaike Weights Decision Criterion.

	- Michalik et al. (2010). 
		Optimal Experimental Design for Discriminating Numerous 
		Model Candidates: The AWDC Criterion.
		Ind. Eng. Chem. Res. 49:913-919
	"""
	mu, s2, noise_var, pps, n, M, _ = _reshape(mu, s2, noise_var, pps)

	iS = np.linalg.inv(s2 + noise_var)
	dc = np.zeros((n,M))
	for i in range(M):
		for j in range(M):
			r1 = np.expand_dims(mu[:,i] - mu[:,j],2) 
			t1 = np.sum(r1*np.matmul(iS[:,i],r1), axis=(1,2))
			dc[:,i] += np.exp(-0.5*t1)
	return np.sum( (1./dc) * pps, axis=1 )


def JR (mu, s2, noise_var=None, pps=None):
	"""
	Quadratic Jensen-Renyi divergence.

	- Olofsson et al. (Future publication)
	"""
	mu, s2, noise_var, pps, _, M, E = _reshape(mu, s2, noise_var, pps)

	# Pre-compute
	S   = s2 + noise_var
	iS  = np.linalg.inv( S )
	dS  = np.linalg.det( S )
	ldS = np.log( dS )

	""" Sum of entropies """
	T1 = np.sum( pps * ( (E/2)*np.log(4*np.pi) + 0.5*ldS ), axis=1)

	""" Entropy of sum """
	# Diagonal elements: (i,i)
	T2 = np.sum( pps*pps / ( 2**(E/2.) * np.sqrt(dS) ), axis=1 )
	
	# Off-diagonal elements: (i,j)
	for i in range(M):
		# mu_i^T * inv(Si) * mu_i 
		mi     = np.expand_dims(mu[:,i], 2)
		iSmi   = np.matmul(iS[:,i], mi)
		miiSmi = np.sum(mi * iSmi, axis=(1,2))

		for j in range(i+1, M):
			# mu_j^T * inv(Sj) * mu_j
			mj     = np.expand_dims(mu[:,j], 2)
			iSmj   = np.matmul(iS[:,j], mj)
			mjiSmj = np.sum(mj * iSmj, axis=(1,2))

			# inv( inv(Si) + inv(Sj) )
			iSiS  = iS[:,i] + iS[:,j]
			iiSiS = np.linalg.inv( iSiS )
			liSiS = np.log( np.linalg.det( iSiS ))

			# mu_ij^T * inv( inv(Si) + inv(Sj) ) * mu_ij
			mij   = iSmi + iSmj
			iiSSj = np.sum(mij * np.matmul(iiSiS, mij), axis=(1,2))

			phi = miiSmi + mjiSmj - iiSSj + ldS[:,i] + ldS[:,j] + liSiS
			T2 += 2 * pps[i] * pps[j] * np.exp(-0.5*phi)

	T2 = E/2 * np.log(2*np.pi) - np.log(T2)
	return T2 - T1


"""
Support methods
"""

# Reshape inputs
def _reshape (mu, s2, noise_var=None, pps=None):
	""" MEAN"""
	if mu.ndim == 2:
		mu = np.expand_dims( mu, axis=2 )
	n, M, E = mu.shape

	""" NOISE VARIANCE """
	# None
	if noise_var is None: 
		noise_var = np.zeros( (E, E) )

	# Scalar
	if isinstance(noise_var, (int,float)):
		noise_var = np.array([noise_var])

	# Numpy vector
	if noise_var.ndim == 1:
		tmp = np.eye(E)
		np.fill_diagonal(tmp, noise_var)
		noise_var = tmp
			
	assert noise_var.shape == (E, E)
	assert np.all( np.diag(noise_var) >= 0. )

	""" COVARIANCE """
	s2  = s2.reshape( (n, M, E, E) )

	""" MODEL PROBABILITIES """
	if pps is None:
		pps = np.ones(M)
	if isinstance(pps, (list,tuple)):
		pps = np.array(pps)

	assert pps.shape == (M,) and np.all(pps >= 0)
	pps = pps / np.sum(pps)

	return mu, s2, noise_var, pps, n, M, E






