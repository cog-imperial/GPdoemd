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
REACTION MODEL
"""
class VTHR2014odeModel:
	"""
	J. Vanlier, C. A. Tiemann, P. A. J. Hilbers and N. A. W. van Riel
	"Optimal experiment design for model selection in biochemical networks"
	BMC Syst Biol 8:20, 2014
	"""
	@property
	def n_outputs (self):
		return 1
	@property
	def x_bounds (self):
		return np.array([[ 1,  100 ],   # Time t
						 [ 0,    1 ],   # Stimulus (binary)
						 [ 0, 4.99 ]])  # Measured output (discrete)
	@property
	def p_bounds (self):
		return np.array( [[ 1e-5, 1. ]] * (10 + 4) )

	def T (self, t=None):
		T = np.linspace(0, 100, 10001)
		return T if t is None else T[ :np.sum( T<=t ) ]

	def init_fluxes (self, R, K):
		A,  Ap,  B, Bp,  C, Cp, BpCp, D, Dp     = R
		k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = K
		# Fluxes
		F    = np.zeros( 10 )
		F[0] = k1  * A
		F[1] = k2  * Ap
		F[2] = k3  * B * Ap
		F[3] = k4  * Bp
		F[4] = k5  * Bp * Cp
		F[5] = k6  * BpCp
		F[6] = k7  * D 
		F[7] = k8  * Dp 
		F[8] = k10 * C
		F[9] = k4  * Cp
		return F

	def stoichiometry (self, F):
		# Fluxes
		f1, f2, f3, f4, f5, f6, f7, f8, f9, f10 = F
		# Rates
		A, Ap, B, Bp, C, Cp, BpCp, D, Dp = range( 9 )
		dR       =  np.zeros( 9 )
		dR[A]    = -f1 + f2
		dR[Ap]   =  f1 - f2
		dR[B]    = -f3 + f4
		dR[Bp]   =  f3 - f4 - f5 + f6
		dR[C]    = -f9 + f10
		dR[Cp]   = -f5 + f6 + f9 - f10
		dR[BpCp] =  f5 - f6
		dR[D]    = -f7 + f8
		dR[Dp]   =  f7 - f8
		return dR

	def ODE (self, R, t, K):
		F  = self.fluxes( R, K )
		dR = self.stoichiometry( F )
		return dR

	def __call__ (self, x, p, all_t=False):
		t, s, m = x
		K, R0   = p[:10], p[10:]
		# Stimulus
		#K    = p.copy() # p[:10]
		K[0] = K[0] if s <= 0.5 else 2*K[0]
		# Initial state
		R0   = np.array([ R0[0], 0, R0[1], 0, R0[2], 0, 0, R0[3], 0 ])

		# Time points
		T = self.T( None if all_t else t )
		# Solve ODE system
		Y = odeint( self.ODE, R0, T, args=(K,) )

		# Return measurable output m at time t
		_, Ap, _, Bp, _, Cp, BpCp, _, Dp = range( 9 )
		m = [Ap, Bp, Cp, BpCp, Dp]
		m = m if all_t else m[ int( np.floor(m) ) ]
		return [Y[::10,m], T[::10]] if all_t else Y[-1, m]


"""
MODELS
"""
class M1 (VTHR2014odeModel):
	@property
	def name (self):
		return 'M1'

	def fluxes (self, R, K):
		F     = self.init_fluxes( R, K )
		F[2] *= 1 / ( K[8] + R[6] ) # 1 / (k9 + BpCp)
		return F

class M2 (VTHR2014odeModel):
	@property
	def name (self):
		return 'M2'

	def fluxes (self, R, K):
		F     = self.init_fluxes( R, K )
		F[0] *= 1 / ( K[8] + R[6] ) # 1 / (k9 + BpCp)
		return F

class M3 (VTHR2014odeModel):
	@property
	def name (self):
		return 'M3'

	def fluxes (self, R, K):
		F     = self.init_fluxes( R, K )
		F[2] *= 1 / ( K[8] + R[8] ) # 1 / (k9 + Dp)
		return F

class M4 (VTHR2014odeModel):
	@property
	def name (self):
		return 'M4'

	def fluxes (self, R, K):
		F     = self.init_fluxes( R, K )
		F[2] *= 1 / ( K[8] + R[7] ) # 1 / (k9 + D)
		return F


"""
Data generator
"""
class DataGen (M1):
	@property
	def truemodel (self):
		return 0
	@property
	def measvar (self):
		return np.array([ 0.03**2 ])
	@property
	def p (self):
		return [0.1554, 0.0090, 0.1247, 0.1000, 0.1144, 0.0178, 0.1517, \
				0.0175, 0.0294, 1.3218, 0.4837, 0.4547, 0.4710, 0.4526]

	def __call__ (self, x):
		state = super().__call__(x, self.p)
		noise = 0.03 * np.random.randn(self.n_outputs)
		return state + noise
 
"""
Get model functions
"""
def get (*args):
	return DataGen(), [ M1(), M2(), M3(), M4() ]

