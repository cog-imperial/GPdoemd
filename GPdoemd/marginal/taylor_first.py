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

from . import GPMarginal
from ..utils import binary_dimensions

"""
First-order Taylor approximation
"""
class TaylorFirstOrder (GPMarginal):
	def __init__(self, *args):
		super().__init__(*args)
	
	def __call__ (self, xnew):
		N       = len(xnew)
		E       = len(self.gps)
		D       = len(self.param_mean)
		R, I, J = binary_dimensions(xnew, self.bin_var)
		# Test points + pmean
		xnew = xnew[:,I]
		Z    = self.get_Z(xnew)

		M   = np.zeros((N,E))
		s2  = np.zeros((N,E))
		dmu = np.zeros((N,E,D))

		for e1 in range(E):
			gp = self.gps[e1]
			for r in R:
				Jr = (J==r)
				if not np.any(Jr):
					continue
				i = np.ix_(Jr,[e1])
				M[i], s2[i] = gp[r].predict_noiseless(Z[Jr])
				""" d mu / d p """
				dmu[Jr,e1]  = self.d_mu_d_p(gp[r], xnew[Jr])

		# Cross-covariance terms
		S = np.zeros((N,E,E))
		for n in range(N):
			mSm  = np.matmul( dmu[n], np.matmul(self.Sigma, dmu[n].T) )
			S[n] = np.diag(s2[n]) + mSm

		for e in range(E):
			S[:,e,e] = np.maximum(1e-15,S[:,e,e])
		return M, S


