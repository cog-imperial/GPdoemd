
import numpy as np
from scipy.stats import multivariate_normal as mv


def expand_dims (a,axis):
	"""
	Expand dims where axis can be a list
	"""
	if isinstance(axis,int):
		return np.expand_dims(a,axis)
	b = a.copy()
	for ax in axis:
		b = np.expand_dims(b,ax)
	return b


def binary_dimensions (Z, binary_variables):
	"""
	Return 

	Inputs
		Z 					Array of design variables (+ parameter (optional))
		binary_variables 	List of indices for binary design variables

	Outputs
		Range ( len( binary_variables ) )
		Column indices for non-binary variables (+ parameters)
		Row indices for different binary variables
	"""
	lb = len( binary_variables )

	if lb == 0:
		n1, n2 = Z.shape
		return [0], np.ones(n2,dtype=bool), np.zeros(n1,dtype=bool)

	B = np.meshgrid( *( [[-1,1]] * lb ))
	B = np.vstack( map( np.ravel, B) ).T

	I  = range( Z.shape[1] )
	I  = [ i for i in I if not i in binary_variables ]

	Zt = Z[:, binary_variables]
	J  = []
	for z in Zt:
		for j, b in enumerate(B):
			if np.all(b * z > 0):
				J.append(j)
				break
	return range(lb), np.array(I), np.array(J)


#def marginal_predict (X, Ms):
	"""
	Return means and covariances for predictions from all models

	inputs
		X   [ N x d ]	Array of designs
		Ms  [ M ] 		List of models

	Outputs
		mu 	[ N x M x E]		Prediction means
		s2  [ N x M x E x E ]	Prediction covariances
	"""
	"""
	N, M, E = len(X), len(Ms), 1
	mu, s2  = np.zeros((N,M,E)), np.zeros((N,M,E,E))
	for i, M in enumerate(Ms):
		mu[:,i], s2[:,i] = M.marginal_predict(X)
	return mu, s2
	"""

