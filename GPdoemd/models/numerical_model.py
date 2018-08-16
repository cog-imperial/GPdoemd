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

from . import Model

class NumericalModel (Model):
	def __init__ (self, model_dict):
		super().__init__(model_dict)
		self.eps = model_dict.get('eps', 1e-6)

	"""
	Properties
	"""
	## Step length for finite difference derivatives
	@property
	def eps (self):
		return self._eps
	@eps.setter 
	def eps (self, value):
		if isinstance(value, float):
			self._eps = value * np.ones(self.dim_p)
		elif isinstance(value, (list,tuple,np.ndarray)):
			self._eps = np.asarray(value)
		assert self._eps.shape == (self.dim_p, )

	"""
	Derivatives
	"""
	def d_mu_d_p (self, e, X):
		N, E, D = len(X), self.num_outputs, self.dim_p
		dmu     = np.zeros( (N, E, D) )
		Y       = np.array([ self.call(x, self.pmean) for x in X ])
		# Numerical differentiation
		for d in range(D):
			p0    = np.zeros(D)
			p0[d] = self.eps[d]
			for n in range(N):
				yp = self.call(X[n], self.pmean+p0)
				dmu[n,:,d] = (yp - Y[n]) / self.eps[d]
		return dmu[:,e]
