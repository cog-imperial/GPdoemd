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

from .param_estim import _residuals
from scipy.optimize import least_squares as lstsq

def least_squares (model, X, Y, p_bounds):
	"""
	Minimisation using a least squares-method
	"""
	assert p_bounds is not None, 'Parameter bounds required for param. estim.'

	def loss_function (p):
		L = _residuals(p, model, X, Y)
		return L.flatten()

	if model.pmean is not None:
		p0 = model.pmean
	elif hasattr(model,'_old_mean') and model._old_pmean is not None:
		p0 = model._old_pmean
	else:
		p0 = np.mean(p_bounds, axis=1)

	res = lstsq(loss_function, p0, bounds=p_bounds.T.tolist())
	return res['x']


