"""
Case study from
J. Vanlier, C. A. Tiemann, P. A. J. Hilbers, and N. A. W. van Riel 
"Optimal experiment design for model selection in biochemical networks" 
BMC Systems Biology, 8(20), 2014.
"""

import numpy as np 

"""
Model super class
"""
class VTHR2014Model:
	@property
	def name (self):
		return 'M' + str(self.num_p)
	@property
	def n_outputs (self):
		return 1
	@property
	def x_bounds (self):
		return np.array([[-2.5, 2.5]])
	@property
	def p_bounds (self):
		return np.array([[-10,10]] * self.num_p)

	def __call__ (self,x,P,grad=False):
		F = [	
			lambda t: t,
			lambda t: t**2,
			lambda t: np.sin(0.2 * t**3),
		 	lambda t: np.sin(2 * t) * t
		 	]
		F  = F[:self.num_p]

		f  = np.sum([ p * f(x) for p,f in zip(P,F)], axis=0)
		df = np.array([ f(x) for f in F ]).T
		return f if not grad else [f, df]

"""
Models
"""
class M1 (VTHR2014Model):
	@property
	def num_p (self):
		return 1

class M2 (VTHR2014Model):
	@property
	def num_p (self):
		return 2

class M3 (VTHR2014Model):
	@property
	def num_p (self):
		return 3

class M4 (VTHR2014Model):
	@property
	def num_p (self):
		return 4

"""
Data generator
"""
class DataGen (M3):
	@property
	def truemodel (self):
		return 2
	@property
	def measvar (self):
		return np.array([0.5])
	@property
	def p (self):
		return [-1, 0.5, -6]

	def __call__ (self,x):
		state = super(DataGen, self).__call__(x,self.p)
		noise = np.sqrt(self.measvar) * np.random.randn()
		return state + noise

"""
Get model functions
"""
def get ():
	return DataGen(), [M1(),M2(),M3(),M4()]
