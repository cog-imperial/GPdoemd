
import numpy as np 

from .marginal import Marginal


"""
First-order Taylor approximation
"""
class TaylorFirstOrder (Marginal):
	def __init__(self, gps, param_mean):
		super().__init__(gps, param_mean)
	
	def __call__ (self, xnew):
		n = len(xnew)
		E = len(self.gps)
		D = len(self.param_mean)

		M   = np.zeros((n,E))
		s2  = np.zeros((n,E))
		dmu = np.zeros((n,E,D))
		
		for e1 in range(E):
			Z   = self.get_Z(xnew)
			tmp = self.gps[e1].predict(Z)

			M[:,e1]   = tmp[0][:,0]
			s2[:,e1]  = tmp[1][:,0]
			dmu[:,e1] = self.d_mu_d_p(self.gps[e1], xnew)

		S = np.array([ np.diag(s2[i]) + \
						np.matmul(dmu[i],np.matmul(self.Sigma,dmu[i].T)) \
						for i in range(n)])
		for e in range(E):
			S[:,e,e] = np.maximum(1e-15,S[:,e,e])
		return M, S

