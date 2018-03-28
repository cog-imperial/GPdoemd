
import numpy as np 
from scipy.integrate import odeint

"""
Model super class
"""
class TGS2017Model:
	#self.ttrans = lambda t: np.log(0.1+t) # Time transformation
	#self.tinvtrans = lambda t: np.exp(t)-0.1 # Time inverse transformation
	def __init__ (self,temperature_dependence):
		self.temp = temperature_dependence # Temperature dependence
	@property
	def outputs (self):
		return [1,3,5,6,8]
	@property
	def n_outputs (self):
		return len( self.outputs )
	@property
	def num_param (self):
		n = self.num_p
		return 2 * n if self.temp else n
	@property
	def x_bounds (self):
		bnd_x1 = [  0.,   1.]
		bnd_x2 = [ 10., 200.]
		bnd_T  = [320., 360.]
		bnd_t  = [  0.,  10.]
		bnds   = np.array([bnd_x1, bnd_x2, bnd_T, bnd_t])
		return bnds if self.t else bnds[[0,1,3]]
	@property
	def p_bounds (self):
		return np.array( [[0.,1.]] * self.num_param )
	@property
	def temp (self):
		return self._temp
	@temp.setter
	def temp (self,temperature_dependence):
		self._temp = temperature_dependence

	def input_to_concentrations (self,x,num_reactants):
		# Concentrations of A and B
		r_AB = x[0] 	# Ratio A to A+B
		rhoM = 55.5 	# mol / litre (molarity of water in water)
		c_A  = r_AB * rhoM
		c_B  = rhoM - c_A
		# Concentration of E
		m_E  = x[1] 						# mg of E
		M_E  = 440000. 						# mg / mol
		totV = 750e-6 						# total volume in litres
		c_E  = m_E / ( total_vol * M_E )	# mol / litre
		return np.array( [c_A,c_B,c_E] + [0.]*(num_reactants-3) )

	def solve(self,ode_system,x,num_reactants,all_t=False):
		# Initial concentrations
		init_conc = self.input_to_concentrations(x,num_reactants)
		t0 = self.tinvtrans(x[-1])
		t = np.linspace(0,10,num=5000) # Time steps for solving ODE
		if not all_t and t0 < t[1]: 
			# Zero:th time point, do not need to solve ODE
			return init_conc[2:4]
		if not all_t: 
			t_ind = 1 + np.sum(t<t0)
			concentrations = odeint(ode_system,init_conc,t[:t_ind])
			return concentrations[-1,2:4]
		concentrations = odeint(ode_system,init_conc,t)
		return concentrations[:,2:4], t

	def get_coeff (self,x,p):
		if self.temp:
			RT = x[2]/1000. # Approximation: p[i]/RT = 1000 * p[i] / (R * T)
			ind = int(self.num_p/2)
			return p[:ind]*np.exp(-p[ind:]/RT)
		return 0.5*p

"""
Models
"""
class M1 (TGS2017Model):
	def __init__ (self,temperature_dependence=False):
		TGS2017Model.__init__(self,temperature_dependence)
	@property
	def name (self):
		return 'M1'
	@property
	def num_p (self):
		return 5 # number of model parameters without temperature dependence

	def __call__ (self, x, p, all_t=False):
		# Reactions coefficients
		k_1, k_2, k_3, k_r3, k_4 = self.get_coeff(x,p)
		# Number of reactants
		num_reactants = 10
		# Reactant indices
		A, B, E, P, D, H, I, N, C, F = range(num_reactants)
		
		def ode_system (conc,t):
			rates = np.zeros(num_reactants)
			# Rate equations
			rates[E] = -k_3*conc[E]*conc[H] + k_r3*conc[F]*conc[N]
			rates[B] = -k_1*conc[B]*conc[A] - k_4*conc[B]*conc[F]
			rates[A] = -k_1*conc[B]*conc[A]
			rates[P] =  k_4*conc[B]*conc[F]
			rates[D] =  k_1*conc[B]*conc[A] - k_2*conc[D] + k_4*conc[B]*conc[F]
			rates[H] =  k_2*conc[D] - k_3*conc[E]*conc[H] + k_r3*conc[F]*conc[N]
			rates[I] =  k_2*conc[D]
			rates[N] =  k_3*conc[E]*conc[H] - k_r3*conc[F]*conc[N]
			rates[C] =  k_1*conc[B]*conc[A]
			rates[F] =  k_3*conc[E]*conc[H] - k_r3*conc[F]*conc[N] \
						- k_4*conc[F]*conc[B]
			return rates

		# Solve ODE systen
		return self.solve(ode_system,x,num_reactants,all_t=all_t)

class M2 (TGS2017Model):
	def __init__ (self,temperature_dependence=False):
		TGS2017Model.__init__(self,temperature_dependence)
	@property
	def name (self):
		return 'M2'
	@property
	def num_p (self):
		return 4 # number of model parameters without temperature dependence

	def __call__ (self, x, p, all_t=False):
		# Reactions coefficients
		k_1, k_2, k_3, k_r2 = self.get_coeff(x,p)
		# Number of reactants
		num_reactants = 8
		# Reactant indices
		A, B, E, P, D, C, F, M = range(num_reactants)

		def ode_system (conc,t):
			rates = np.zeros(num_reactants)
			# Rate equations
			rates[E] = -k_2*conc[E]*conc[A] + k_r2*conc[F]*conc[M]
			rates[B] = -k_1*conc[B]*conc[A] - k_3*conc[B]*conc[F]
			rates[A] = -k_1*conc[B]*conc[A] - k_2*conc[E]*conc[A] \
						+ k_r2*conc[F]*conc[M]
			rates[P] =  k_3*conc[B]*conc[F]
			rates[D] =  k_1*conc[B]*conc[A] + k_3*conc[B]*conc[F]
			rates[C] =  k_1*conc[B]*conc[A]
			rates[F] =  k_2*conc[E]*conc[A] - k_r2*conc[F]*conc[M] \
						- k_3*conc[F]*conc[B]
			rates[M] =  k_2*conc[E]*conc[A] - k_r2*conc[F]*conc[M]
			return rates

		# Solve ODE systen
		return self.solve(ode_system,x,num_reactants,all_t=all_t)

class M3 (TGS2017Model):
	def __init__ (self,temperature_dependence=False):
		TGS2017Model.__init__(self,temperature_dependence)
	@property
	def name (self):
		return 'M3'
	@property
	def num_p (self):
		return 7 # number of model parameters without temperature dependence

	def __call__ (self, x, p, all_t=False):
		# Reactions coefficients
		k_1, k_2, k_3, k_4, k_5, k_r3, k_r4 = self.get_coeff(x,p)
		# Number of reactants
		num_reactants = 11
		# Reactant indices
		A, B, E, P, D, C, F, M, N, H, I = range(num_reactants)

		def ode_system (conc,t):
			rates = np.zeros(num_reactants)
			# Rate equations
			rates[E] = -k_3*conc[E]*conc[A] + k_r3*conc[F]*conc[M] \
						- k_4*conc[E]*conc[H] + k_r4*conc[F]*conc[N]
			rates[B] = -k_1*conc[B]*conc[A] - k_5*conc[B]*conc[F]
			rates[A] = -k_1*conc[B]*conc[A] - k_3*conc[E]*conc[A] \
						+ k_r3*conc[F]*conc[M]
			rates[P] =  k_5*conc[B]*conc[F]
			rates[D] =  k_1*conc[B]*conc[A] - k_2*conc[D] + k_5*conc[B]*conc[F]
			rates[C] =  k_1*conc[B]*conc[A]
			rates[F] =  k_3*conc[E]*conc[A] - k_r3*conc[F]*conc[M] \
						+ k_4*conc[E]*conc[H] - k_r4*conc[F]*conc[N] \
						- k_5*conc[F]*conc[B]
			rates[M] =  k_3*conc[E]*conc[A] - k_r3*conc[F]*conc[M]
			rates[N] =  k_4*conc[E]*conc[H] - k_r4*conc[F]*conc[N]
			rates[H] =  k_2*conc[D] - k_4*conc[E]*conc[H] + k_r4*conc[F]*conc[N]
			rates[I] =  k_2*conc[D]
			return rates

		# Solve ODE systen
		return self.solve(ode_system,x,num_reactants,all_t=all_t)

"""
Data generator
"""
class DataGen (M3):
	def __init__ (self,temperature_dependence=False):
		M3.__init__(self,temperature_dependence)
	@property
	def truemodel (self):
		return 2
	@property
	def measvar (self):
		return np.array([0.01, 0.01])**2
	@property
	def p (self):
		added_p = self.num_p - 7
		return [0.15,0.35,0.25,0.8,0.1,0.1,0.05] + [0.25]*added_p

	def __call__ (self, x, all_t=False):
		state = super(DataGen, self).__call__(x, self.p, all_t=all_t)
		noise = np.sqrt(self.measvar) * np.random.randn(self.n_outputs)
		return state if all_t else state + noise

"""
Get model functions
"""
def get_functions (temperature_dependence=False):
	return DataGen(temp), [M1(temp),M2(temp),M3(temp)]


