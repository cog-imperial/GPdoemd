
import numpy as np 
from scipy.integrate import ode

"""
Reactor model super class
"""
class ReactorModel:
	@property
	def t_meas (self):
		return [7., 9., 11., 13., 15.]	# Time points
	@property
	def n_outputs (self):
		return 3 * len( self.t_meas )
	@property
	def x_bounds (self):
		return np.array([[0., 0.03],
						 [0., 0.20],
						 [0., 0.15],
						 [0., 0.21],
						 [0., 0.45]])
	@property
	def e_coli_p_init (self):
		return [1.30, 15.6, 0.06, 0.04]
	@property
	def e_coli_p_bounds (self):
		return [[  1., 1.5],	# qSmax
				[ 10., 20.],	# qOmax
				[0.01, 0.1],	# qAcmax
				[0.01, 0.1]]	# qm
	@property
	def Sfeed (self):
		return 520	# [g/L]
	@property
	def X0 (self):
		return [ 13.86,  # S0
				  0.02,  # A0
				 0.035,  # X0
				   7.0]  # V0

	def x_constraints (self, x):
		# Non-linear constraints
		if (x[0] > 0.0110 and x[4] > 0.22) or \
		   (x[0] > 0.0096 and x[4] > 0.25) or \
		   (x[1] > 0.0350 and x[4] > 0.25) or \
		   (x[1] > 0.0400 and x[4] > 0.24) or \
		   (x[2] > 0.0370 and x[4] > 0.28):
			return False
		elif (0.350 <= x[3] <= 0.0430 and  25.0*x[3] + x[4] > 1.325) or \
			 (0.043 <= x[3] <= np.inf and -1.11*x[3] + x[4] > 0.200):
			return False

		# Linear constraints
		As = np.array([	
			[ 34.56,  9.784,  9.377, 0.2334, 1.3e-03 ],  # 1.
			[ 98.08,  91.11,  89.41,   9.41, 4.7e-03 ],  # 10.
			[  3.67,     1.,     0.,     0.,      0. ],  # 0.11
			[    4.,     0.,     1.,     0.,      0. ],  # 0.12
			[  1.33,     0.,     0.,     1.,      0. ],  # 0.18
			[   -4.,     0.,     0.,     0.,      1. ],  # 0.4
			[    0.,    1.1,     1.,     0.,      0. ],  # 0.115
			[    0.,   0.25,     0.,     1.,      0. ],  # 0.175
			[    0.,  -1.67,     0.,     0.,      1. ],  # 0.4
			[    0.,     0.,  -0.67,     1.,      0. ],  # 0.18
			[    0.,     0.,   1.61,     1.,      0. ],  # 0.326
			[    0.,     0.,    11.,     1.,      0. ],  # 1.27
			[    0.,     0.,   3.33,     0.,      1. ],  # 0.57
			[    0.,     0.,  -1.76,     0.,      1. ],  # 0.4
			[    0.,     0.,     0.,  -1.67,      1. ],  # 0.4
			[    0.,     0.,     0.,     3.,      1. ],  # 0.69
			[    0.,     0.,     0.,   6.75,      1. ]]) # 1.22

		bs = [ 1., 10., 0.11, 0.12, 0.18, 0.4, 0.115, 0.175, 0.4, 0.18, 
			  0.326, 1.27, 0.57, 0.4, 0.4, 0.69, 1.22]

		for A,b in zip(As,bs):
			if np.sum(A*x)>b:
				return False
		return True

	def get_x_grid (self, N=7):
		a = [np.linspace(*xbnd,num=N) for xbnd in self.x_bounds]
		X = np.meshgrid(*a)
		X = np.vstack( map(np.ravel, X) ).T
		X = np.array([x for x in X if self.x_constraints(x)])
		return X

	def __call__ (self, x, p, all_t=False):
		Yinit = self.get_initial_state()
		T0 = np.array([0] + self.t_meas)
		if all_t:
			Y = [Yinit.copy()]
			T = np.linspace(0,15,301)
		else:
			Y = []
			T = np.array([0] + self.t_meas)

		dt     = 15./2000.
		t_prev = 0.
		i      = 1
		for j in range(len(self.t_meas)):
			r = ode( self.ode_system(x[j],p) )
			r.set_integrator('vode', nsteps=1000, method='bdf')
			r.set_initial_value(Yinit, T0[j])

			while r.successful() and r.t < T0[j+1]:
				y = r.integrate(r.t + dt)
				if all_t and r.successful() and t_prev < T[i] <= r.t:
					Y.append(y)
					t_prev = r.t
					i += 1
			if not r.successful():
				while i < len(T):
					Y.append(np.nan * np.ones( len(Yinit) ))
					i += 1
				print('ODE integration unsuccessful')
			if not all_t:
				Y.append(y)
			Yinit = y.copy()

		Y = np.array(Y)
		return [Y,T] if all_t else Y


"""
E. coli model
"""
def EColiModel(S,A,p):
	"""
	E.Coli Model from Xu et al. 1999 
	"Modeling of Overflow Metabolism in Batch and Fed-Batch Cultures of 
	Escherichia coli"
	
	Lukas Hebing, 2018
	Bayer AG
	"""

	# parameter vector translation
	qSmax,qOmax,qAcMax,qm = p
	Ca, Cs, Cx = 1/30, 1/30, 0.04
	KA, KiO, KIs, KS = 0.05, 4, 5, 0.05
	#qAcMax, qm, qOmax, qSmax = 0.06, 0.04, 15.6, 1.30
	Y_AS,Y_OA,Y_OS,Y_XA,Y_XS_of,Y_XS_ox = 0.667, 1.067, 1.067, 0.4, 0.15, 0.51

	""" Cell model """
	# Glucose uptake [g/gCDW/h]
	qS = qSmax * (1/(1+A/KIs)) * (S/(S+KS))

	# maximum oxygen uptake 
	qOsMax = qOmax/(1+A/KiO)
	qOsMax = qOsMax * 2.*16./1000. # mmol/gCDW/h -> g/gCDW/h

	# calculate overflow flux
	Y1 = Y_XS_ox * Cx/Cs  # help variable 1
	Y2 = Y_OS             # help variable 2
	qSox_OxLimit = (qOsMax-qm*Y1*Y2)/(Y2*(1-Y1))

	# overflow and oxydative flow
	# case 1) oxygen limited
	if qSox_OxLimit < qS:
		qSox    = qSox_OxLimit
		qSof    = qS - qSox
		qAc     = 0
	else: # case 2) not oxygen limited
		qSox    = qS
		qSof    = 0
		qAc     = qAcMax * A /(A + KA)

	# flux in anabolism
	qSoxAn = (qSox - qm) * Y1
	# flux for the aerobic energy metabolism
	qSoxEn = qSox - qSoxAn

	# oxygen used for glucose oxidation
	qOs = qSoxEn * Y2
	# contribution to growth from this overflow glucose flux
	qSofAn = qSof * Y_XS_of*Cx/Cs
	# remaining flux is used for the energy production via acetate formation
	qSofEn = qSof - qSofAn
	# acetate formation rate
	qAp = qSofEn * Y_AS

	# second switch: flux of acetate for respiratory combustion
	# a constraint for this is that the remaining respiration capacity is 
	# liberated from the declining glucose metabolism
	qAcEnMax    = (qOmax - qOs)/Y_OA
	Y1          = Y_XA*Cx/Ca # helping variable

	# case 1) remaining respiration capacity is enough
	if qAc*(1-Y1) < qAcEnMax:
		qAcEn = qAc*(1-Y1)
	else: # case 2) maximum respiration capacity is reached
		qAc = qAcEnMax/(1-Y1)
		qAcEn = qAc*(1-Y1)

	# The total oxygen consumption rate
	qO = qOs + qAcEn * Y_OA
	qO = qO * 1000./(2.*16.) # g/gCDW/h -> mmol/gCDW/h
	# The specific growth rate obtained from three substrate fluxes
	mu = (qSox - qm)*Y_XS_ox + qSof * Y_XS_of + qAc*Y_XA
	# Acetate net flux [g/gCDW/h]
	qA = qAp - qAc

	return qS,qA,mu,qO


"""
Reactor model types
"""
# Continuous stirred-tank reactor
class CSTRmodel (ReactorModel):
	def get_initial_state (self):
		# Initial state
		return np.array(self.X0)

	def ode_system (self,x,p):
		"""
		Simulate the CSTR model with the E.Coli cell model from Xu et al. 1999 

		x 	Design vector
			Feed rate in time intervals [0,7), [7,9), [9,11), [11,13), [13,15)
		p 	E.Coli cell model parameters
		
		Lukas Hebing, 2018
		Bayer AG
		"""
		def _ode_system (t,X):
			# States
			S = X[0]   # Substrate [g/l]
			A = X[1]   # Acetate   [g/l]
			V = X[3]   # Volume
			X = X[2]   # Biomass

			""" derivatives """
			# Specific rates
			qS,qA,mu,qO = EColiModel(S,A,p)
			# Dilution rate [1/h]
			F = x
			D = F/V

			dX = np.zeros(4)
			# Substrate
			dX[0] = D*(self.Sfeed - S) - qS*X
			# Acetate
			dX[1] = qA*X - D*A
			# Biomass
			dX[2] = (mu - D) * X
			# Volume 
			dX[3] = F
			# Oxygen consumption rate
			OCR = (qO/32.)*X*1000.

			return dX
		return _ode_system


class PFR_STRmodel (ReactorModel):
	@property
	def nDisc (self):
		return int(20)

	def get_initial_state (self):
		# Initial state
		nDisc = self.nDisc 
		return np.array(self.X0 + \
			  [self.X0[0]]*nDisc + [self.X0[1]]*nDisc + [self.X0[2]]*nDisc)

	def ode_system (self,x,p):
		"""
		Simulate PFR/STR model with the E.Coli cell model from Xu et al. 1999

		x 	Design vector
			Feed rate in time intervals [0,7), [7,9), [9,11), [11,13), [13,15)
		p 	Parameter vector:
			p[0]   - residence time in PFR
			p[1]   - volume ratio PFR to STR
			p[2]   - (Only PFRtau model) measurement point in PFR
			p[-4:] - E.Coli cell model parameters
		
		Lukas Hebing, 2018
		Bayer AG
		"""
		tauPFR, alpha = p[:2]
		e_coli_p = p[-4:]
		
		def _ode_system(t,X):
			""" Split state vector """
			# Concentrations of S,A,X in STR
			S0 = X[0]
			A0 = X[1] 
			X0 = X[2]
			# Overall Volume
			V = X[3]
			# concentrations of S,A,X in PFR
			S = X[4              :4+1*self.nDisc]
			A = X[4+1*self.nDisc :4+2*self.nDisc]
			X = X[4+2*self.nDisc :4+3*self.nDisc]
			# Volume
			Ffeed = x
			dXvolume = np.array([Ffeed])

			""" STR part """
			# specific rates
			qS,qA,mu,_ = EColiModel(S0,A0,e_coli_p)

			# Dilution rate [1/h]
			D = alpha/tauPFR

			dXstr = np.zeros(3)
			# substrate
			dXstr[0] = D*(S[-1] - S0) - qS*X0
			# Acetate
			dXstr[1] = D*(A[-1] - A0) + qA*X0
			# Biomass
			dXstr[2] = D*(X[-1] - X0) + mu*X0

			""" Concentrations in the inled of the PFR """
			# Volume flow into and out of PFR [L/h]
			Fcirc   = alpha*V/tauPFR
			# Volume flow out of STR
			FoutStr = Fcirc - Ffeed

			# mixing rule
			S1 = (S0*FoutStr+self.Sfeed*Ffeed)/Fcirc
			A1 = (A0*FoutStr)/Fcirc
			X1 = (X0*FoutStr)/Fcirc

			""" PFR part """
			# approximated with forward differences (due to lack of diffusion)
			dtau = tauPFR/self.nDisc

			dS = np.zeros(self.nDisc)
			dA = np.zeros(self.nDisc)
			dX = np.zeros(self.nDisc)

			qS,qA,mu,_ = EColiModel(S[0],A[0],e_coli_p)
			dS[0] = (S1-S[0])/dtau - X[0]*qS
			dA[0] = (A1-A[0])/dtau + X[0]*qA
			dX[0] = (X1-X[0])/dtau + X[0]*mu

			for z in range(1,self.nDisc):
				# specific rates
				qS,qA,mu,_ = EColiModel(S[z],A[z],e_coli_p)
				dS[z] = (S[z-1]-S[z])/dtau - X[z]*qS
				dA[z] = (A[z-1]-A[z])/dtau + X[z]*qA
				dX[z] = (X[z-1]-X[z])/dtau + X[z]*mu

			dXpfr = np.concatenate((dS,dA,dX))

			# gradient
			return np.concatenate((dXstr,dXvolume,dXpfr))
		return _ode_system


"""
Models
"""
class CSTR (CSTRmodel):
	# Continuous stirred-tank reactor
	@property
	def name (self):
		return 'CSTR'
	@property
	def p_bounds (self):
		# Model parameter bounds
		return np.array( self.e_coli_p_bounds )
	@property
	def p_init (self):
		# Initial parameter guess
		return self.e_coli_p_init

	def __call__ (self, x, p, all_t=False):
		Y = super(CSTR,self).__call__(x, p, all_t=all_t)
		return Y[:,:3].T.flatten() if not all_t else [ Y[0][:,:3], Y[1] ]


class PFR (PFR_STRmodel):
	# PFR + STR model, with measurements in STR
	@property
	def name (self):
		return 'PFR'
	@property
	def p_bounds (self):
		# Model parameter bounds
		return np.array([[1./120, 1./30]] + 	# Residence time
						[[  0.01,  0.99]] + 	# Volume ratio
						self.e_coli_p_bounds )
	@property
	def p_init (self):
		# Initial parameter guess
		return [1./60, 0.1] + self.e_coli_p_init

	def __call__ (self,x, p, all_t=False):
		Y = super(PFR,self).__call__(x, p, all_t=all_t)
		return Y[:,:3].T.flatten() if not all_t else [ Y[0][:,:3], Y[1] ]

class PFRtau (PFR_STRmodel):
	# PFR + STR model, with measurements in PFR
	@property
	def name (self):
		return 'PFRtau'
	@property
	def p_bounds (self):
		# Model parameter bounds
		return np.array([[1./120, 1./30]] + 	# Residence time
						[[  0.01,  0.99]] + 	# Volume ratio
						[[    0.,    1.]] +		# Measurement location
						self.e_coli_p_bounds )
	@property
	def p_init (self):
		# Initial parameter guess
		return [1./60, 0.1, 0.5] + self.e_coli_p_init

	def __call__ (self,x,p,all_t=False):
		Y = super(PFRtau,self).__call__(x,p,all_t=all_t)
		if all_t:
			Y, t = Y
		# Get measurement
		nDisc  = self.nDisc
		tau = p[2]
		def interpolate (y):
			x    = np.linspace(0.,1.,len(y))
			i, j = int(np.sum(x<=tau)-1), int(len(x)-np.sum(x>tau))
			if i < 0: 
				return y[0]
			if j >= len(x): 
				return y[-1]
			a = ( y[j] - y[i] ) / ( x[j] - x[i] )
			b =   y[i] - a * x[i]
			return a * tau + b
		S = [interpolate(y[4        :4+1*nDisc]) for y in Y]
		A = [interpolate(y[4+1*nDisc:4+2*nDisc]) for y in Y]
		X = [interpolate(y[4+2*nDisc:4+3*nDisc]) for y in Y]
		Y = np.array([S,A,X])
		return Y.flatten() if not all_t else Y.T, t


"""
Get model functions
"""
def get_functions ():
	return [CSTR(), PFR(), PFRtau()]










