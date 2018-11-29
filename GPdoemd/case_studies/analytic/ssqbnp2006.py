
import numpy as np

"""
Case study from Schwaab et al. 2006
"A new approach for sequential experimental design for model discrimination"

using parametrisation from:
Christian Hoffman PhD thesis (page 226-227) 

"""

class SSQBNP2006Model:
	def __init__ (self, name='Model'):
		self.name = name	

	def alpha (self, x):
		z1, z2, z3, z4, z5 = x
		return 1 - (z3*z4)/(z1*z2) * np.exp(4.33 - 4577.8/z5)

class M1:
	def __init__ (self, name='M1'):
		self.name = name

	def __call__ (self, x, p):
		return NotImplementedError

class DataGen (M1):
	def __init__ (self, name='DataGen'):
		self.p = [1.6855, 4.5947, 0.94219, 0.89669, 2.4591]
