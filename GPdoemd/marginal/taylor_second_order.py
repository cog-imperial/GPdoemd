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

def taylor_second_order (model, xnew):
	N, E, D = len(xnew), model.num_outputs, model.dim_p
	M, s2   = model.predict(xnew)
	assert M.shape == (N, E) and s2.shape == (N, E)
	dmu     = np.zeros((N,E,D))
	ddmuA   = np.zeros((N,E,D,D))
	
	for e in range(E):
		dmu[:,e] = model.d_mu_d_p(e, xnew)
		ddmu     = model.d2_mu_d_p2(e, xnew)
		dds2     = model.d2_s2_d_p2(e, xnew)

		""" d^2 mu / d p^2 * S_p """
		ddmuA[:,e] = np.matmul(ddmu, model.Sigma)
		""" trace ( d^2 s2 / d p^2 * S_p ) """
		trdds2A     = np.sum(dds2 * model.Sigma, axis=(1,2))

		M[:,e]  += 0.5 * np.trace(ddmuA[:,e], axis1=1, axis2=2)
		s2[:,e] += 0.5 * trdds2A
		
	# Cross-covariance terms
	S = np.zeros((N,E,E))
	for n in range(N):
		mSm  = np.matmul( dmu[n], np.matmul(model.Sigma, dmu[n].T) )
		S[n] = np.diag(s2[n]) + mSm
					
		for e1 in range(E):
			for e2 in range(e1,E):
				S[n,e1,e2] += 0.5 * np.sum(ddmuA[n,e1] * ddmuA[n,e2].T)
				if not e1 == e2:
					S[n,e2,e1] = S[n,e1,e2]
			# Safety check
			S[n,e1,e1] = np.maximum(1e-15, S[n,e1,e1])
	return M, S