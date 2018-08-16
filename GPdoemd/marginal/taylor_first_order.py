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

def taylor_first_order (model, xnew):
	N, E, D = len(xnew), model.num_outputs, model.dim_p
	M, s2   = model.predict(xnew)
	assert M.shape == (N, E) and s2.shape == (N, E)
	dmu     = np.zeros((N, E, D))
	for e in range(E):
		dmu[:,e] = model.d_mu_d_p(e, xnew)

	# Cross-covariance terms
	S = np.zeros((N,E,E))
	for n in range(N):
		mSm  = np.matmul( dmu[n], np.matmul(model.Sigma, dmu[n].T) )
		S[n] = np.diag(s2[n]) + mSm

	for e in range(E):
		S[:,e,e] = np.maximum(1e-15, S[:,e,e])
	return M, S