
import numpy as np
from scipy.stats import multivariate_normal as mv


def expand_dims (a,axis):
	if isinstance(axis,int):
		return np.expand_dims(a,axis)
	b = a.copy()
	for ax in axis:
		b = np.expand_dims(b,ax)
	return b

def binary_dimensions (Z, binary_variables):
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
	return range(len(B)), np.array(I), np.array(J)


def marginal_predict (X, Ms):
	N, M, E = len(X), len(Ms), 1
	mu, s2  = np.zeros((N,M,E)), np.zeros((N,M,E,E))
	for i, M in enumerate(Ms):
		mu[:,i], s2[:,i] = M.marginal_predict(X)
	return mu, s2


def model_probabilities (X, Y, models=None, prediction=None):
	if models is None:
		mu, s2 = prediction
	else:
		mu, s2 = marginal_predict(X, models)
	M = mu.shape[1]
	
	pis = []
	for i in range( M ):
		pi = np.sum([ mv.logpdf(y,m[i],s[i]) for y,m,s in zip(Y,mu,s2) ])
		pis.append(pi)
	pis = [ 1. / np.sum([ np.exp( np.min(( np.max((p2 - p1, -1000)), 100 )))
			for p2 in pis]) for p1 in pis ]
	return np.array(pis)

