"""
Case study from
G. Buzzi-Ferraris, P. Forzatti, G. Emig, and H. Hofmann. 
"Sequential experimental design for model discrimination in the case of 
multiple responses" 
Chem Eng Sci, 39(1):81-85, 1984.
"""

import numpy as np 

"""
Model super class
"""
class BFFEH1984Model:
	@property
	def n_outputs (self):
		return 2
	@property
	def x_bounds (self):
		return np.array([[5., 55.]]*2)
	@property
	def p_bounds (self):
		return np.array([[0., 1.]]*4)

"""
Models
"""
class M1 (BFFEH1984Model):
	@property
	def name (self):
		return 'M1'

	def __call__ (self,x,p,grad=False):
		x1,x2 = x[0],x[1]
		k1,k2,K3,K4 = p

		dnm = 1 + K3*x1 + K4*x2
		y   = x1 * x2 / dnm
		if not grad: return np.array([k1*y, k2*y])

		dy  = y/dnm
		dy1 = [ y, 0., -k1*x1*dy, -k1*x2*dy]
		dy2 = [0.,  y, -k2*x1*dy, -k2*x2*dy]
		return np.array([k1*y, k2*y]), np.array([dy1, dy2])

class M2 (BFFEH1984Model):
	@property
	def name (self):
		return 'M2'

	def __call__ (self,x,p,grad=False):
		x1,x2 = x[0],x[1]
		k1,k2,K3,K4 = p

		dm1 = 1. + K3*x1 + K4*x2
		dm2 = 1. + K3*x1
		y1  = x1*x2 / dm1**2
		y2  = x1*x2 / dm2**2
		if not grad: return np.array([k1*y1, k2*y2])

		dy1 = [y1, 0., -2.*k1*x1*y1/dm1, -2.*k1*x2*y1/dm1]
		dy2 = [0., y2, -2.*k1*x1*y2/dm2,               0.]
		return np.array([k1*y1, k2*y2]), np.array([dy1, dy2])

class M3 (BFFEH1984Model):
	@property
	def name (self):
		return 'M3'

	def __call__ (self,x,p,grad=False):
		x1,x2 = x[0],x[1]
		k1,k2,K3,K4 = p

		dm1 = 1.+K3*x2
		dm2 = 1.+K4*x1
		y1  = x1*x2 / dm1**2
		y2  = x1*x2 / denom2**2
		if not grad: return np.array([k1*y1, k2*y2])

		dy1 = [y1, 0., -2.*k1*x2*y1/dm1,  0.]
		dy2 = [0., y2,  0., -2.*k2*x1*y2/dm2]
		return np.array([k1*y1, k2*y2]), np.array([dy1, dy2])

class M4 (BFFEH1984Model):
	@property
	def name (self):
		return 'M4'

	def __call__ (self,x,p,grad=False):
		x1,x2 = x[0],x[1]
		k1,k2,K3,K4 = p

		dm1 = 1 + K3*x1 + K4*x2
		dm2 = 1 + K3*x1
		y1  = x1*x2 / dm1
		y2  = x1*x2 / dm2
		if not grad: return np.array([k1*y1, k2*y2])

		dy1 = [y1, 0., -k1*x1*y1/dm1, -k1*x2*y1/dm1]
		dy2 = [0., y2, -k2*x1*y2/dm2,            0.]
		return np.array([k1*y1, k2*y2]), np.array([dy1, dy2])


"""
Data generator
"""
class DataGen (M1):
	@property
	def truemodel (self):
		return 0
	@property
	def measvar (self):
		return np.array([0.35, 2.3e-3])
	@property
	def p (self):
		return [0.1, 0.01, 0.1, 0.01]

	def __call__ (self,x):
		state = super(Datagen,self).__call__(x,self.p)
		noise = np.sqrt(self.measvar) * np.random.randn(self.n_outputs)
		return state + noise

"""
Get model functions
"""
def get (*args):
	return DataGen(), [M1(),M2(),M3(),M4()]




