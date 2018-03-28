"""
Case study from
C. Michalik, M. Stuckert, and W. Marquardt 
"Optimal experimental design for discriminating numerous model candidates: 
The AWDC criterion" 
Ind Eng Chem Res, 49:913–919, 2010.
"""


import numpy as np 

"""
Model super class
"""
class MSM2010Model:
	@property
	def n_outputs (self):
		return 1
	@property
	def x_bounds (self):
		return np.array([[0.,10.],[0.,10.],[0.,100.]])
	@property
	def p_bounds (self):
		return np.array([[0.,1000.]])

	def __call__ (self,x,p,grad=False):
		C  = self.f(x)
		dC = np.array([C])
		return C*p if not grad else [C*p, dC]

"""
Models
"""
class M1 (MSM2010Model):
	@property
	def name (self):
		return 'M1'
	def f (self,x):
		return np.array([1.])
		
class M2 (MSM2010Model):
	@property
	def name (self):
		return 'M2'
	def f (self,x):
		return x[[0]]
		
class M3 (MSM2010Model):
	@property
	def name (self):
		return 'M3'
	def f (self,x):
		return x[[1]]
		
class M4 (MSM2010Model):
	@property
	def name (self):
		return 'M4'
	def f (self,x):
		return x[[2]]
		
class M5 (MSM2010Model):
	@property
	def name (self):
		return 'M5'
	def f (self,x):
		return x[[0]] * x[[1]]
		
class M6 (MSM2010Model):
	@property
	def name (self):
		return 'M6'
	def f (self,x):
		return x[[0]] * x[[2]]
		
class M7 (MSM2010Model):
	@property
	def name (self):
		return 'M7'
	def f (self,x):
		return x[[1]] * x[[2]]
		
class M8 (MSM2010Model):
	@property
	def name (self):
		return 'M8'
	def f (self,x):
		return x[[0]] * x[[1]] * x[[2]]

class M9 (MSM2010Model):
	@property
	def name (self):
		return 'M9'
	def f (self,x):
		return x[[0]]**2 * x[[1]]

class M10 (MSM2010Model):
	@property
	def name (self):
		return 'M10'
	def f (self,x):
		return x[[0]] * x[[1]]**2

"""
Data generator
"""
class DataGen (M9):
	@property
	def truemodel (self):
		return 8
	@property
	def measvar (self):
		return np.array([1.])
	@property
	def p (self):
		return [1.]

	def __call__ (self,x):
		state = super(DataGen, self).__call__(x,self.p)
		noise = np.sqrt(self.measvar) * np.random.randn(self.n_outputs)
		return state + noise

"""
Get model functions
"""
def get (*args):
	return DataGen(), [M1(),M2(),M3(),M4(),M5(),M6(),M7(),M8(),M9(),M10()]




