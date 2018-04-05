
import numpy as np 
from scipy.integrate import odeint

from pdb import set_trace as st

"""
Model super class
"""
class VTHR2014Model:
	#self.ttrans = lambda t: np.log(0.1+t) # Time transformation
	#self.tinvtrans = lambda t: np.exp(t)-0.1 # Time inverse transformation
	@property
	def num_reactants (self):
		return 9
	@property
	def outputs (self):
		return [1,3,5,6,8]
	@property
	def n_outputs (self):
		return len( self.outputs )
	@property
	def x_bounds (self):
		return np.array([[0, 20]] + [[0., 1.]]*4)
	@property
	def p_bounds (self):
		return np.array([[0.,1.]]*10)

	def init_conc (self,A0,B0,C0,D0,stimulus=False):
		c = np.array([A0, 0., B0, 0., C0, 0., 0., D0, 0.])
		if stimulus:
			c[0] *= 2.
		return c

	def solve (self,ode_system,x,all_t=False):
		# Initial concentrations
		#Bp0,Dp0 = x[1:]
		#init_conc = np.array([1,0.1,1,Bp0,1,0.1,0.1,1,Dp0])
		A0,B0,C0,D0 = [0.4837, 0.4547, 0.4710, 0.4526]
		init_conc = self.init_conc(A0,B0,C0,D0,stimulus=)
		# Stopping time
		#t0 = self.tinvtrans(x[0])
		t_stop  = x[0]
		t_steps = np.linspace(*self.x_bounds[0], num=500) # Time steps
		if not all_t: 
			if t_stop < t_steps and not all_t:
				return init_conc[self.outputs]
			t_ind = 1 + np.sum(t<t0)
			concentrations = odeint(ode_system,init_conc,t[:t_ind])
			return concentrations[-1,self.outputs]
		concentrations = odeint(ode_system,init_conc,t)
		return concentrations[:,self.outputs], t

"""
Models
"""
class M1 (VTHR2014Model):
	@property
	def name (self):
		return 'M1'

	def __call__ (self,x,p,all_t=False):
		A, Ap, B, Bp, C, Cp, BpCp, D, Dp = range(self.num_reactants)
		k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = p
		def ode_system (conc,t):
			f21 = k2*conc[Ap]   - k1*conc[A]
			f43 = k4*conc[Bp]   - k3*conc[B]*conc[Ap]/(k9+conc[BpCp])
			f65 = k6*conc[BpCp] - k5*conc[Bp]*conc[Cp]
			f78 = k7*conc[D]    - k8*conc[Dp]
			f90 = k10*conc[C]   - k4*conc[Cp]
			return np.array([f21,-f21,f43,f65-f43,-f90,f65+f90,-f65,-f78,f78])
		return self.solve(ode_system,x,all_t=all_t)

class M2 (VTHR2014Model):
	@property
	def name (self):
		return 'M2'
		
	def __call__ (self,x,p,all_t=False):
		A, Ap, B, Bp, C, Cp, BpCp, D, Dp = range(self.num_reactants)
		k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = p
		def ode_system (conc,t):
			f21 = k2*conc[Ap]   - k1*conc[A]/(k9+conc[BpCp])
			f43 = k4*conc[Bp]   - k3*conc[B]*conc[Ap]
			f65 = k6*conc[BpCp] - k5*conc[Bp]*conc[Cp]
			f78 = k7*conc[D]    - k8*conc[Dp]
			f90 = k10*conc[C]   - k4*conc[Cp]
			return np.array([f21,-f21,f43,f65-f43,-f90,f65+f90,-f65,-f78,f78])
		return self.solve(ode_system,x,all_t=all_t)

class M3 (VTHR2014Model):
	@property
	def name (self):
		return 'M3'
		
	def __call__ (self,x,p,all_t=False):
		A, Ap, B, Bp, C, Cp, BpCp, D, Dp = range(self.num_reactants)
		k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = p
		def ode_system (conc,t):
			f21 = k2*conc[Ap]   - k1*conc[A]
			f43 = k4*conc[Bp]   - k3*conc[B]*conc[Ap]/(k9+conc[Dp])
			f65 = k6*conc[BpCp] - k5*conc[Bp]*conc[Cp]
			f78 = k7*conc[D]    - k8*conc[Dp]
			f90 = k10*conc[C]   - k4*conc[Cp]
			return np.array([f21,-f21,f43,f65-f43,-f90,f65+f90,-f65,-f78,f78])
		return self.solve(ode_system,x,all_t=all_t)

class M4 (VTHR2014Model):
	@property
	def name (self):
		return 'M4'
		
	def __call__ (self,x,p,all_t=False):
		A, Ap, B, Bp, C, Cp, BpCp, D, Dp = range(self.num_reactants)
		k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = p
		def ode_system (conc,t):
			f21 = k2*conc[Ap]   - k1*conc[A]
			f43 = k4*conc[Bp]   - k3*conc[B]*conc[Ap]/(k9+conc[D])
			f65 = k6*conc[BpCp] - k5*conc[Bp]*conc[Cp]
			f78 = k7*conc[D]    - k8*conc[Dp]
			f90 = k10*conc[C]   - k4*conc[Cp]
			return np.array([f21,-f21,f43,f65-f43,-f90,f65+f90,-f65,-f78,f78])
		return self.solve(ode_system,x,all_t=all_t)

"""
Data generator
"""
class DataGen (M3):
	@property
	def truemodel (self):
		return 2
	@property
	def measvar (self):
		return 9e-4 * np.ones( self.n_outputs )
	@property
	def p (self):
		#self.p = np.array([0.15,0.35,0.25,0.8,0.1,0.1,0.05])
		#self.p = 0.2 * np.random.rand(10)
		#self.p[3] = 0.1
		#self.p[9] *= 10
		return [0.1554,0.0090,0.1247,0.1,0.1144,0.0178,0.1517,0.0175,0.0294,1.3218]

	def __call__ (self,x,all_t=False):
		state = super(DataGen, self).__call__(x,self.p,all_t=all_t)
		noise = np.sqrt(self.measvar)*np.random.randn(self.n_outputs)
		return state if all_t else state + noise

"""
Get model functions
"""
def get_functions ():
	return DataGen(), [M1(),M2(),M3(),M4()]


import matplotlib.pyplot as plt 
if __name__ == '__main__':
	model = M1()
	pbnds = model.p_bounds
	for n in range(5):
		p = 0.2 * np.random.rand(10)
		p[3] = 0.1
		p[9] *= 10
		#p = np.random.uniform(pnds[:,0],pbnds[:,1])
		fig = plt.figure(figsize=(12,8))
		fig.suptitle(p)
		axs = [plt.subplot(151+i) for i in range(5)]
		
	p = 0.5*np.ones(len(model.p_bounds))
	x = np.array([0.2,0.1,0.1])
	print(model(x,p))
	"""
	f,t = model(x,p,True)
	plt.figure(figsize=(12,8))
	ind = [range(5),range(5,9)]
	ind = [(0,i) for i in range(5)] + [(1,i) for i in range(4)]
	axs = [plt.subplot2grid((2,5),IJ) for IJ in ind]
	"""

