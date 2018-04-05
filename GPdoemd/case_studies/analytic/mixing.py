
import numpy as np
from scipy.special import exp1

"""
Model super class
"""
class MicroMacroModel:
	@property
	def n_outputs (self):
		return 1
	@property
	def x_bounds (self):
		return np.array([[   1., 100.],	 # residence time
						 [ 0.01,   1.],	 # initial concentration
						 [   0.,   1.]]) # reactor type
	@property
	def p_bounds (self):
		return np.array([[ 1e-6, 1e-1]]) # reaction constant

"""
PFR reactor models
"""
def PFR_0 (x, p, grad=False):
	R  = p * x[0] / x[1]
	Rt = np.array( R < 1., dtype=float)
	C  = Rt * (1. - R)
	dC = Rt * -x[0]/x[1]
	return C if not grad else [C, dC[None,:]]

def PFR_1 (x, p, grad=False):
	R  = p * x[0]
	C  = np.exp(-R)
	dC = -x[0] * C
	return C if not grad else [C, dC[None,:]]

def PFR_2 (x, p, grad=False):
	R  = p * x[0] * x[1]
	C  = 1. / (1. + R)
	dC = -x[1] * x[0] * C**2
	return C if not grad else [C, dC[None,:]]

"""
CSTR reactor models
"""
def CSTR_0_macro (x, p, grad=False):
	R  = p * x[0] / x[1]
	C  = 1 - R + R * np.exp(-1./R)
	dC = x[0] / x[1] * ((1. + 1./R) * np.exp(-1./R) - 1.)
	return C if not grad else [C, dC[None,:]]

def CSTR_1 (x, p, grad=False):
	R  = p * x[0]
	C  = 1. / (1. + R)
	dC = -x[0] * C**2
	return C if not grad else [C, dC[None,:]]

def CSTR_2_micro (x, p, grad=False):
	R  = p * x[0] * x[1]
	C  = 1. / (2 * R) * ( np.sqrt(1 + 4*R) - 1 )
	dC = ( 1. / np.sqrt(1 + 4*R) - C ) / p
	return C if not grad else [C, dC[None,:]]

def CSTR_2_macro (x, p, grad=False):
	R  = p * x[0] * x[1]
	if R < 1.5e-3:
		# Risk of overflow
		C  = np.array([1.])
		dC = np.array([0.])
	else:
		C  = (1. / R) * np.exp(1. / R) * exp1(1. / R)
		dC = (1. - C*R - C)/(R*p)
	return C if not grad else [C, dC[None,:]]


"""
Models
"""
class M1 (MicroMacroModel):
	# Zero-order reaction, micro mixing
	@property
	def name (self):
		return 'M1'
	def __call__ (self, x, p, grad=False):
		return PFR_0(x, p, grad=grad)

class M2 (MicroMacroModel):
	# Zero-order reaction, macro mixing
	@property
	def name (self):
		return 'M2'
	def __call__ (self, x, p, grad=False):
		reactor = CSTR_0_macro if x[2] <= 0.5 else PFR_0
		return reactor(x, p, grad=grad)

class M3 (MicroMacroModel):
	# First-order reaction
	@property
	def name (self):
		return 'M3'
	def __call__ (self, x, p, grad=False):
		reactor = CSTR_1 if x[2] <= 0.5 else PFR_1
		return reactor(x, p, grad=grad)

class M4 (MicroMacroModel):
	# Second-order reaction, micro mixing
	@property
	def name (self):
		return 'M4'
	def __call__ (self, x, p, grad=False):
		reactor = CSTR_2_micro if x[2] <= 0.5 else PFR_2
		return reactor(x, p, grad=grad)

class M5 (MicroMacroModel):
	# Second-order reaction, macro mixing
	@property
	def name (self):
		return 'M5'
	def __call__ (self, x, p, grad=False):
		reactor = CSTR_2_macro if x[2] <= 0.5 else PFR_2
		return reactor(x, p, grad=grad)

"""
Data generator
"""
class DataGen (M3):
	def __call__ (self, x):
		return 0
	@property
	def measvar (self):
		return np.array([0.05])**2

"""
Get model functions
"""
def get (*args):
	return DataGen(), [M1(),M2(),M3(),M4(),M5()]








