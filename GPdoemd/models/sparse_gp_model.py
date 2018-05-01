
import numpy as np 
import warnings

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
		Z = self.Z
		Y = self.Y

		self.set_kernels(kern_x, kern_p)
		kern_x = self.kern_x
		kern_p = self.kern_p
		dim_x  = self.dim_x - self.dim_b
		dim_p  = self.dim_p
		dim    = dim_x + dim_p

		R, I, J = binary_dimensions(Z, self.binary_variables)

		assert not np.any([ value is None for value in [Z, Y, kern_x, kern_p] ])

		gps = []
		for e in range( self.num_outputs ):
			gps.append([])
			for r in R:
				Jr = (J==r)

				if not np.any(Jr):
					gps[e].append(None)
					continue

				kernx = kern_x(dim_x, range(dim_x), 'kernx')
				kernp = kern_p(dim_p, range(dim_x, dim), 'kernp')
				kern  = kernx * kernp

				Zr    = Z[ np.ix_(Jr,  I ) ]
				Yr    = Y[ np.ix_(Jr, [e]) ]

				numi  = num_inducing
				if numi is None:
					numi = np.max(( 10, np.ceil(np.sqrt(len(Zr))).astype(int) ))

				gp = SparseGPRegression(Zr, Yr, kern, num_inducing=numi)
				gps[e].append(gp)
		self.gps = gps
