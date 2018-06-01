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

"""
Model super class
"""
class BFFC1990Model:
	"""
	G. Buzzi-Ferraris, P. Forzatti and P. Canu
	"An improved version of a sequential design criterion for discriminating 
	among rival multiresponse models" 
	Chem Eng Sci, 45(2):477-481, 1990
	"""
	@property
	def n_outputs (self):
		return 1
	@property
	def x_bounds (self):
		return np.array([[  703,  753 ],  # Temperature
		                 [  300,  350 ],  # Pressure 
		                 [  0.1,  0.2 ]]) # Inlet ammonia mole fraction

	def get_constant (self, T, p1, p2, grad=False):
		tmp = (T - 700.)/T
		C   = np.exp( p1 - p2 * tmp )
		dC  = np.array([ C, -C*tmp ])
		return C if not grad else [C, dC]

	def get_activity_coeffs (self, T, P):
		"""
		D. C. Dyson and J. M. Simon, 1968. I&EC Fundamentals 7(4).
		"""
		gH2  = np.exp( P * np.exp( 0.541 - 3.8402 * T**0.125 ) \
		             - P**2 * np.exp( -15.98 - 0.1263 * T**0.5 ) \
		             + 300 * np.exp(-5.941 - 0.011901*T) * (np.exp(-P/300.)-1))
		gN2  = 0.93431737 + 3.101804e-4 * T + 2.958960e-4 * P \
		             - 2.707279e-7 * T**2 + 4.775207e-7 * P**2
		gNH3 = 0.14389960 + 2.028538e-3 * T - 4.487672e-4 * P \
		             - 1.142945e-6 * T**2 + 2.761216e-7 * P**2
		return gN2, gH2, gNH3

	def get_equilibrium_constant (self, T):
		"""
		J. Gillespie and J. A. Beattie, 1930. Physical Review 36.
		"""
		log10Keq = 2.6899 - 2.691122 * np.log10(T) - 5.519265e-5 * T \
				 + 1.848863e-7 * T**2 + 2001.6 / T
		return 10.**log10Keq

	def get_fugacities (self, x):
		T, P, X_NH3    = x 
		# Activity coefficients
		gN2, gH2, gNH3 = self.get_activity_coeffs( T, P )
		# Mole fractions (inert-free, stoichiometric reaction)
		X_N2 = 0.25 * (1. - X_NH3)
		X_H2 = 3 * X_N2
		# Fugacities
		f_N2  = X_N2  * P * gN2
		f_H2  = X_H2  * P * gH2
		f_NH3 = X_NH3 * P * gNH3
		# Equilibrium constant
		Keq = self.get_equilibrium_constant( T )
		return f_N2, f_H2, f_NH3, Keq

"""
Models
"""
class M1 (BFFC1990AModel):
	@property
	def name (self):
		return 'M1'

	@property
	def p_bounds (self):
		return np.array([[ 0,  10.], 
		                 [ 0, 100.]])

	def __call__ (self, x, p, grad=False):
		# Constants
		p11, p12 = p
		C1 = self.get_constant(x[0], p11, p12, grad=grad)
		if grad:
			C1, dC1 = C1
		# Fugacities
		f_N2, f_H2, f_NH3, Keq = self.get_fugacities(x)

		nom = f_N2 - f_NH3**2 / ( f_H2**3 * Keq**2 )
		dnm = C1 * f_NH3 / f_H2**1.5
		f   = np.array([ nom / dnm ])
		if not grad:
			return f

		df = np.array([ -f/C1 * dC1 ])
		return f, df
		

class M2 (BFFC1990AModel):
	@property
	def name (self):
		return 'M2'

	@property
	def p_bounds (self):
		return np.array([[ 0,  10.], 
		                 [ 0, 100.]])

	def __call__ (self, x, p, grad=False):
		# Constants
		p11, p12 = p
		C1 = self.get_constant(x[0], p11, p12, grad=grad)
		if grad:
			C1, dC1 = C1
		# Fugacities
		f_N2, f_H2, f_NH3, Keq = self.get_fugacities(x)

		nom = f_N2*f_H2 - f_NH3**2 / ( f_H2 * Keq )**2
		dnm = C1 * f_NH3
		f   = np.array([ nom / dnm ])
		if not grad:
			return f

		df = np.array([ -f/C1 * dC1 ])
		return f, df
		

class M3 (BFFC1990AModel):
	@property
	def name (self):
		return 'M3'

	@property
	def p_bounds (self):
		return np.array([[ 0,  10.], 
		                 [ 0, 100.],
		                 [ 0,  10.], 
		                 [ 0, 100.]])

	def __call__ (self, x, p, grad=False):
		# Constants
		p11, p12, p21, p22 = p
		C1 = self.get_constant(x[0], p11, p12, grad=grad)
		C2 = self.get_constant(x[0], p21, p22, grad=grad)
		if grad:
			C1, dC1 = C1
			C2, dC2 = C2
		# Fugacities
		f_N2, f_H2, f_NH3, Keq = self.get_fugacities(x)

		nom = np.sqrt(f_N2 * f_H2**3) - f_NH3 / Keq
		k1  = f_NH3
		k2  = np.sqrt(f_N2 / f_H2)
		dnm = k1 * C1  + k2 * C2 
		f   = np.array([ nom / dnm ])
		if not grad:
			return f

		dC = np.array([ (k1*dC1).tolist() + (k2*dC2).tolist() ])
		df = -f/dnm * dC
		return f, df
		

class M4 (BFFC1990AModel):
	@property
	def name (self):
		return 'M4'

	@property
	def p_bounds (self):
		return np.array([[ 0,  10.], 
		                 [ 0, 100.],
		                 [ 0,  10.], 
		                 [ 0, 100.],
		                 [ 0,  10.], 
		                 [ 0, 100.]])

	def __call__ (self, x, p, grad=False):
		# Constants
		p11, p12, p21, p22, p31, p32 = p
		C1 = self.get_constant(x[0], p11, p12, grad=grad)
		C2 = self.get_constant(x[0], p21, p22, grad=grad)
		C3 = self.get_constant(x[0], p31, p32, grad=grad)
		if grad:
			C1, dC1 = C1
			C2, dC2 = C2
			C3, dC3 = C3
		# Fugacities
		f_N2, f_H2, f_NH3, Keq = self.get_fugacities(x)

		nom = np.sqrt(f_N2 * f_H2**3) - f_NH3 / Keq
		k1  = f_NH3
		k2  = f_N2
		k3  = f_NH3 / f_N2
		dnm = k1 * C1  + k2 * C2 + k3 * C3
		f   = np.array([ nom / dnm ])
		if not grad:
			return f

		dC = np.array([ (k1*dC1).tolist()+(k2*dC2).tolist()+(k3*dC3).tolist() ])
		df = -f/dnm * dC
		return f, df


"""
Data generator
"""
class DataGen (M1):
	@property
	def truemodel (self):
		return 0
	@property
	def measvar (self):
		return np.array([ 90 ])
	@property
	def p (self):
		return [3.68064, 11.80406] # 8262.84]

	def __call__ (self,x):
		state = super().__call__(x, self.p)
		noise = np.sqrt(self.measvar) * np.random.randn(self.n_outputs)
		return state + noise
