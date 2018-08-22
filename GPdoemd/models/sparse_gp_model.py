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

from GPy.models import SparseGPRegression

from . import GPModel
from ..utils import binary_dimensions

class SparseGPModel (GPModel):
	def __init__ (self, model_dict):
		super().__init__(model_dict)

	"""
	Sparse GP surrogate model
	"""
	def gp_surrogate (self, Z=None, Y=None, kern_x=None, kern_p=None, num_inducing=None):
		self.set_training_data(Z, Y)
		assert self.Z is not None and self.Y is not None

		self.set_kernels(kern_x, kern_p)
		assert self.kern_x is not None and self.kern_p is not None

		R, J = binary_dimensions(self.Z, self.binary_variables)

		gps = []
		for e in range( self.num_outputs ):
			gps.append([])
			for r in R:
				Jr = (J==r)

				if not np.any(Jr):
					gps[e].append(None)
					continue

				dim_xb = self.dim_x - self.dim_b
				dim    = self.dim_x + self.dim_p
				kernx  = self.kern_x(dim_xb, self.non_binary_variables, 'kernx')
				kernp  = self.kern_p(self.dim_p, range(self.dim_x, dim), 'kernp')
				#Zr     = self.Z[ np.ix_(Jr,  I ) ]
				Zr     = self.Z[ Jr ]
				Yr     = self.Y[ np.ix_(Jr, [e]) ]

				numi  = num_inducing
				if numi is None:
					numi = np.max(( 10, np.ceil(np.sqrt(len(Zr))).astype(int) ))

				gp = SparseGPRegression(Zr, Yr, kernx*kernp, num_inducing=numi)
				gps[e].append(gp)
		self.gps = gps
