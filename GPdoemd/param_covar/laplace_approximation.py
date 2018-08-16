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

def laplace_approximation (model, Xdata):
	# Dimensions
	E = model.num_outputs
	D = model.dim_p

	meas_noise_var = model.meas_noise_var
	if isinstance(meas_noise_var, (int, float)):
		meas_noise_var = np.array([meas_noise_var] * E)
	
	# Invert measurement noise covariance
	if meas_noise_var.ndim == 1: 
		imeasvar = np.diag(1./meas_noise_var)
	else: 
		imeasvar = np.linalg.inv(meas_noise_var)
	
	# Inverse covariance matrix
	iA = np.zeros( (D, D) )

	for e1 in range(E):
		dmu1 = model.d_mu_d_p(e1, Xdata)
		iA  += imeasvar[e1,e1] * np.matmul(dmu1.T, dmu1)

		if meas_noise_var.ndim == 1:
			continue
		for e2 in range(e1+1,E):
			if imeasvar[e1,e2] == 0.:
				continue
			dmu2 = model.d_mu_d_p(e2, Xdata)
			iA  += imeasvar[e1,e2] * np.matmul(dmu1.T, dmu2)
			iA  += imeasvar[e2,e1] * np.matmul(dmu2.T, dmu1)

	Sigma = np.linalg.inv(iA)
	return Sigma


