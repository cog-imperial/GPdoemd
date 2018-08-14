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
from ..marginal import Analytic, Numerical

class AnalyticModel (Model):
	def __init__ (self, model_dict):
		super().__init__(model_dict)
		# Optional parameters
		self.binary_variables = []


	"""
	Marginal predictions
	"""
	@property
	def gprm (self):
		return None if not hasattr(self,'_gprm') else self._gprm
	@gprm.setter
	def gprm (self, value):
		assert isinstance(value, (Numerical, Analytic))
		self._gprm = value
	@gprm.deleter
	def gprm (self):
		self._gprm = None

	def marginal_init (self, method):
		self.gprm = method( self, self.pmean )

	def marginal_compute_covar (self, Xdata):
		if self.gprm is None:
			return None
		mvar = self.meas_noise_var
		self.gprm.compute_param_covar(Xdata, mvar)

	def marginal_init_and_compute_covar (self, method, Xdata):
		self.marginal_init(method)
		self.marginal_compute_covar(Xdata)

	def marginal_predict (self, xnew):
		if self.gprm is None:
			return None
		return self.gprm(xnew)

