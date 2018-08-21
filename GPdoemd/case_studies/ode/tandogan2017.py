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
from scipy.integrate import odeint

"""
Model super class
"""
class Tandogan2017Model:
	"""
	-- Models based on:
	N. Tandogan, S. Garcia-Munoz, M. Sen, T. M. Wilson, J. Y. Buser,
	S. P. Kolis, I. V. Borkar, and C. A. Alt
	"Use of Model Discrimination Method in Drug Substance Process Development"
	AIChE Annual Meeting, Minneapolis, NM, USA, 2017.

	-- Some chemical constants (e.g. rhoM and M_E) have been set by the
	-- developers of GPdoemd, as well as default model parameter values.
	"""
	@property
	def n_outputs (self):
		return 2
	@property
	def x_bounds (self):
		return np.array([[ 0.1,  0.8 ],    # Ratio A_0 / (A_0 + B_0)
						 [  10,  200 ],    # E_0 (mg)
						 [ 320,  360 ],    # Temperature [K]
						 [   0,    5 ]])   # Measurement time point [h]
	@property
	def p_bounds (self):
		return np.c_[ np.zeros(self.num_p), np.ones(self.num_p) ]

	@property
	def V (self):
		return 750e-6 # litres

	def T (self, t=None):
		T = np.linspace(0, 18000, 18001) # seconds
		return T if t is None else T[ :np.sum( T<=t*3600 ) ]

	def input_to_concentrations (self, x):
		# Concentrations of A and B
		r_AB = x[0]            # Ratio A to A+B
		rhoM = 55.5            # mol / litre (molarity of water in water)
		c_A  = r_AB * rhoM
		c_B  = (1. - r_AB) * rhoM
		# Concentration of E
		m_E  = x[1]            # mg of E
		M_E  = 4.4e5 * self.V  # mg * litre / mol
		c_E  = m_E / M_E       # mol / litre
		return [c_A, c_B, c_E]

	def get_coeff (self, x, p=None):
		T = x[2] # Temperature
		if p is None:
			p = self.p
		i    = int(self.num_p/2)
		A, E = p[:i], p[i:]
		K    = np.exp( A - E / T ) # Arrhenius equation
		return K

	def ODE (self, R, t, K):
		return self.rates( R, K )

	def __call__ (self, x, p=None, all_t=False):
		# Initial state
		R0 = self.R_init( x )
		# Time points
		T  = self.T( None if all_t else x[3] )
		# Rate coefficients
		K  = self.get_coeff(x, p)
		# Solve ODE system
		Y  = odeint( self.ODE, R0, T, args=(K,) )
		Y *= self.V
		return [Y[::10,:5], T[::10]/3600.] if all_t else Y[-1, [2, 3]]


"""
Models
"""
class M1 (Tandogan2017Model):
	@property
	def name (self):
		return 'M1'
	@property
	def num_p (self):
		return 10

	@property
	def p (self):
		A = [ -13.1, -4.2,  10.,  15,  -7.5 ]
		E = [  510.,  33., 526., 775., 605. ]
		return np.array(A + E)

	def R_init (self, x):
		R0          = np.zeros(10)
		R0[[0,1,2]] = self.input_to_concentrations( x )
		return R0

	def rates (self, R, K):
		# Fluxes
		A, B, E, P, D, H, I, N, C, F = R
		k_1,  k_2,  k_3,  k_r3, k_4  = K
		f0 = k_1  * A * B
		f1 = k_2  * D
		f2 = k_3  * E * H
		f3 = k_r3 * F * N
		f4 = k_4  * B * F

		# Rates
		A = -f0
		B = -f0 - f4
		C =  f0
		D =  f0 - f1 + f4
		E = -f2 + f3
		F =  f2 - f3 - f4
		H =  f1 - f2 + f3
		I =  f1
		N =  f2 - f3
		P =  f4
		return np.array([ A, B, E, P, D, H, I, N, C, F ])


class M2 (Tandogan2017Model):
	@property
	def name (self):
		return 'M2'
	@property
	def num_p (self):
		return 8

	@property
	def p (self):
		A = [ -25., -10.8, -19.7, -8.1 ]
		E = [ 357.,  542.,  870., 448. ]
		return np.array(A + E)

	def R_init (self, x):
		R0          = np.zeros(8)
		R0[[0,1,2]] = self.input_to_concentrations( x )
		return R0

	def rates (self, R, K):
		# Fluxes
		A, B, E, P, D, C, F, M = R
		k_1,  k_2,  k_r2, k_3  = K
		f0 = k_1  * A * B
		f1 = k_2  * A * E
		f2 = k_r2 * F * M
		f3 = k_3  * B * F

		# Rates
		A = -f0 - f1 + f2
		B = -f0 - f3
		C =  f0
		D =  f0 + f3
		E = -f1 + f2
		F =  f1 - f2 - f3
		M =  f1 - f2
		P =  f3
		return np.array([ A, B, E, P, D, C, F, M ])


class M3 (Tandogan2017Model):
	@property
	def name (self):
		return 'M3'
	@property
	def num_p (self):
		return 14

	@property
	def p (self):
		A = [ -22.2,  19.5, -19.1, -0.67, 16.9, 16.8, -1.96 ]
		E = [  592.,  46.9,  57.1,  432., 924., 419.,  486. ]
		return np.array(A + E)

	def R_init (self, x):
		R0          = np.zeros(11)
		R0[[0,1,2]] = self.input_to_concentrations( x )
		return R0

	def rates (self, R, K):
		# Fluxes
		A, B, E, P, D, C, F, M, N, H, I     = R
		k_1, k_2, k_3, k_r3, k_4, k_r4, k_5 = K
		f0 = k_1  * A * B
		f1 = k_2  * D
		f2 = k_3  * A * E
		f3 = k_r3 * F * M
		f4 = k_4  * E * H
		f5 = k_r4 * F * N
		f6 = k_5  * F * B

		# Rates
		A = -f0 - f2 + f3
		B = -f0 - f6
		C =  f0
		D =  f0 - f1 + f6
		E = -f2 + f3 - f4 + f5
		F =  f2 - f3 + f4 - f5 - f6
		H =  f1 - f4 + f5
		I =  f1
		M =  f2 - f3
		N =  f4 - f5
		P =  f6
		return np.array([ A, B, E, P, D, C, F, M, N, H, I ])


"""
Data generator
"""
class DataGen (Tandogan2017Model):
	def __init__ (self, i=0):
		self.truemodel = i

	@property
	def truemodel (self):
		return self._truemodel
	@truemodel.setter
	def truemodel (self, value):
		if not hasattr(self, '_truemodel'):
			assert isinstance(value, int) and 0 <= value <= 2
			self._truemodel = value

	@property
	def model (self):
		return [M1(), M2(), M3()][ self.truemodel ]
	@property
	def name (self):
		return self.model.name
	@property
	def p (self):
		return self.model.p[0] + self.model.p[1]
	@property
	def measvar (self):
		return np.array([0.01, 0.01])**2

	def __call__ (self, x, all_t=False):
		state = self.model(x, all_t=all_t)
		if all_t: return state
		noise = np.sqrt(self.measvar) * np.random.randn(self.n_outputs)
		return np.abs( state + noise )


"""
Get model functions
"""
def get (i=0):
	return DataGen(i), [M1(), M2(), M3()]
