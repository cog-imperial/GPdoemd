
import numpy as np 
from scipy.stats import multivariate_normal as mv
from scipy.stats import chi2 as scipy_chi2

"""
Gaussian likelihood
"""
def gaussian_likelihood (Y, M, S):
	"""
	y     [ N x E ]
	M     [ N x M x E ]
	S     [ N x M x E x E ]
	"""
	logpis = []
	for i in range( M.shape[1] ):
		logpi = np.sum([ mv.logpdf(y,m[i],s[i]) for y,m,s in zip(Y,M,S) ])
		logpis.append(logpi)
	pis = []
	for p1 in logpis:
		pi = 0
		for p2 in logpis:
			pi += np.exp( np.min(( np.max((p2 - p1, -1000)), 100 )))
		pis.append(1. / pi)
	return np.array( pis )

"""
Update Gaussian likelihood
"""
def gaussian_likelihood_update (y, M, S, prior):
	"""
	y     [ E ]
	M     [ M x E ]
	S     [ M x E x E ]
	"""
	p   = np.array([ mv.pdf(y, mu, s2) for mu, s2 in zip(M, S) ])
	pup = p * prior
	return pup / np.sum(pup)


"""
Chi-squared adequacy test
"""
def chi2 (Y, M, S, D):
	"""
	y     [ N x E ]
	M     [ N x M x E ]
	S     [ E ], [ E x E] or [ N x M x E x E ]
	"""
	N, E = Y.shape
	maha = _maha_sum(Y, M, S)
	return 1. - scipy_chi2.cdf(maha, N*E - D)

# Summed squared Mahalanobis distance
def _maha_sum (Y, F, S):
	n, M = F.shape[:2]
	ms = np.zeros(M)
	if S.ndim == 1:
		for i in range(M):
			d = F[:,i]-Y
			ms[i] += np.sum(d**2/S)
		return ms
	elif S.ndim == 2:
		iS = np.linalg.inv(S)
		for i in range(M):
			for j in range(n):
				d = F[j,i]-Y[j]
				ms[i] += np.sum(d * np.matmul(d,iS))
		return ms
	else:
		iS = np.linalg.inv(S)
		for i in range(M):
			for j in range(n):
				d = F[j,i]-Y[j]
				ms[i] += np.sum(d * np.matmul(d,iS[j,i]))
		return ms


"""
Akaike information criterion weights
"""
def aicw (Y, M, S, D):
	"""
	y     [ N x E ]
	M     [ N x M x E ]
	S     [ N x M x E x E ]
	D     [ M ]
	"""
	# Log Gaussian likelihood
	logL = []
	for i in range( M.shape[1] ):
		L = np.sum([ mv.logpdf(y,m[i],s[i]) for y,m,s in zip(Y,M,S) ])
		logL.append( L )
	# Akaike information criterion
	aic = 2 * np.array(logL) - 2 * D
	# Akaike weights 
	aws = []
	for a1 in aic:
		aw = 0
		for a2 in aic:
			aw += np.exp( np.min(( np.max((a2 - a1, -1000)), 100 )))
		aws.append( 1./aw )
	return np.array( aws )





