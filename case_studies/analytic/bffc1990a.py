"""
Case study from
G. Buzzi-Ferraris, P. Forzatti and P. Canu
"An improved version of a sequential design criterion for discriminating among
rival multiresponse models"
Chem Eng Sci 45(2):477-481, 1990.
"""

import numpy as np 

"""
Model super class
"""
class BFFC1990Model:
	@property
	def n_outputs (self):
		return 1
	@property
	def x_bounds (self):
		return np.array([[300,350], [703,753], [0.1,0.2]])
	@property
	def p_bounds (self):
		return np.array([[0.,10.]]*num_p)

	def thermodynamics (self,P,T,yNH3):
		# D.C. Dyson and J.M. Simon (1968)
		# =================================
		# Thermodynamic equilibrium
		log10Keq = 2.6899 - 2.691122*np.log10(T) - 5.519265e-5*T \
				+ 1.848863e-7*T**2 + 2001.6/T
		Keq = 10**log10Keq
		# Fugacities
		fH2 = P*0

		return 10.,1.,5.,1. #fN2,fH2,fNH3,Keq

	def C (self,T,p1,p2):
		return np.exp(p1+p2*(1000./T-1000./700.))

"""
Models
"""
class M1 (BFFC1990Model):
	@property
	def name (self):
		return 'M1'
	@property
	def num_p (self):
		return 2	

	def __call__ (self,x,p,grad=False):
		P,T,yNH3 = x
		fN2,fH2,fNH3,Keq = self.thermodynamics(P,T,yNH3)
		C1 = self.C(T,p[0],p[1])
		nom, denom = fN2-fNH3**2/(fH2**3*Keq**2), fNH3/(fH2**1.5)
		y = nom/(C1*denom)
		if not grad: return np.array([y])
		dp1 = -y
		dp2 = -y*(1000./T-1000./700.)
		return np.array([y]), np.array([[dp1,dp2]])

class M2 (BFFC1990Model):
	@property
	def name (self):
		return 'M2'
	@property
	def num_p (self):
		return 2

	def __call__ (self,x,p,grad=False):
		P,T,yNH3 = x
		fN2,fH2,fNH3,Keq = self.thermodynamics(P,T,yNH3)
		C1 = self.C(T,p[0],p[1])
		nom, denom = fN2*fH2-(fNH3/(fH2*Keq))**2, fNH3
		y = nom/(C1*denom)
		if not grad: return np.array([y])
		dp1 = -y
		dp2 = -y*(1000./T-1000./700.)
		return np.array([y]), np.array([[dp1,dp2]])

class M3 (BFFC1990Model):
	@property
	def name (self):
		return 'M3'
	@property
	def num_p (self):
		return 4

	def __call__ (self,x,p,grad=False):
		P,T,yNH3 = x
		fN2,fH2,fNH3,Keq = self.thermodynamics(P,T,yNH3)
		C1 = self.C(T,p[0],p[1])
		C2 = self.C(T,p[2],p[3])
		sqrt_fN2_fH2 = np.sqrt(fN2/fH2)
		nom, denom = np.sqrt(fN2*fH2**3)-fNH3/Keq, C1*fNH3+C2*sqrt_fN2_fH2
		y = nom/denom
		if not grad: return np.array([y])
		dp1 = -y*fNH3/denom * C1
		dp2 = -y*fNH3/denom * C1 * (1000./T-1000./700.)
		dp3 = -y*sqrt_fN2_fH2/denom * C2
		dp4 = -y*sqrt_fN2_fH2/denom * C2 * (1000./T-1000./700.)
		return np.array([y]), np.array([[dp1,dp2,dp3,dp4]])

class M4 (BFFC1990Model):
	@property
	def name (self):
		return 'M4'
	@property
	def num_p (self):
		return 6

	def __call__ (self,x,p,grad=False):
		P,T,yNH3 = x
		fN2,fH2,fNH3,Keq = self.thermodynamics(P,T,yNH3)
		C1 = self.C(T,p[0],p[1])
		C2 = self.C(T,p[2],p[3])
		C3 = self.C(T,p[4],p[5])
		nom = np.sqrt(fN2*fH2**3) - fNH3/Keq
		denom = C1*fNH3 + C2*fN2 + C3*fNH3/fN2
		y = nom/denom
		if not grad: return np.array([y])
		dp1 = -y*fNH3/denom * C1
		dp2 = -y*fNH3/denom * C1 * (1000./T-1000./700.)
		dp3 = -y*fN2/denom * C2
		dp4 = -y*fN2/denom * C2 * (1000./T-1000./700.)
		dp5 = -y*fNH3/(fN2*denom) * C3
		dp6 = -y*fNH3/(fN2*denom) * C3 * (1000./T-1000./700.)
		return np.array([y]), np.array([[dp1,dp2,dp3,dp4,dp5,dp6]])

"""
Data generator
"""
class DataGen (M1):
	@property
	def truemodel (self):
		return 0
	@property
	def measvar (self):
		return 90
	@property
	def p (self):
		return [3.68064, 8.26284]

	def __call__ (self,x):
		state = super(DataGen, self).__call__(x,self.p)
		noise = np.sqrt(self.measvar)*np.random.randn(self.n_outputs)
		return state + noise

"""
Get model functions
"""
def get (*args):
	return DataGen(), [M1(),M2(),M3(),M4(),M5()]


X = np.array([	[753., 350, 0.20],
				[753., 350, 0.10],
				[753., 300, 0.20],
				[753., 300, 0.10],
				[703., 350, 0.20],
				[703., 350, 0.10],
				[703., 300, 0.20],
				[703., 300, 0.10],
				[753., 350, 0.13],
				[736., 300, 0.13],
				[736., 300, 0.13],
				[736., 300, 0.13],
				[753., 300, 0.20],
				[736., 300, 0.13],
				[736., 300, 0.13],
				[753., 300, 0.20],
				[736., 300, 0.13],
				[703., 350, 0.10]])

Y = np.array([	[213.3],
				[672.0],
				[140.3],
				[498.4],
				[108.8],
				[307.9],
				[ 85.3],
				[234.4],
				[429.9],
				[253.6],
				[260.4],
				[246.1],
				[137.5],
				[246.3],
				[259.8],
				[154.1],
				[252.2],
				[331.1]])

import checkgrad
if __name__ == '__main__':
	model = M4()
	bnds = model.x_bounds
	x = np.random.uniform(bnds[:,0],bnds[:,1])
	def func (P):
		n,D = P.shape
		y = np.zeros(n)
		dy = np.zeros((n,D))
		for i,p in enumerate(P):
			yt,dyt = model(x,p,True)
			y[i] = yt[0]
			dy[i] = dyt[0]
		return y,dy
	bnds = model.p_bounds
	P = np.random.uniform(bnds[:,0],bnds[:,1],size=(3,len(bnds)))
	d,dydh = checkgrad.check_gradient(func,P,flatten=True)
	print(dydh)



