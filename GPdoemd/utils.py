
import numpy as np
from scipy.stats import multivariate_normal as mv


def expand_dims (a,axis):
	if isinstance(axis,int):
		return np.expand_dims(a,axis)
	b = a.copy()
	for ax in axis:
		b = np.expand_dims(b,ax)
	return b


def marginal_predict (Ms, X):
	N, M, E = len(X), len(Ms), 1
	mu, s2  = np.zeros((N,M,E)), np.zeros((N,M,E,E))
	for i, M in enumerate(Ms):
		mu[:,i], s2[:,i] = M.marginal_predict(X)
	return mu, s2


def model_probabilities (Ms, X, Y):
	mu, s2 = marginal_predict(Ms, X)
	
	pis = []
	for i in range( len(Ms) ):
		pi = np.sum([ mv.logpdf(y,m[i],s[i]) for y,m,s in zip(Y,mu,s2) ])
		pis.append(pi)
	pis = [ 1. / np.sum([ np.exp( np.min(( np.max((p2 - p1, -1000)), 100 )))
			for p2 in pis]) for p1 in pis ]
	return np.array(pis)

