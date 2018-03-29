
import numpy as np 
import GPy


class Trainer:
	def __init__ (self,model,meas_var,kern_x=None,verbose=True):
		self.model = model
		self.meas_var = meas_var
		# Time kernel
		self.kern_x = kern_x
		# Variable bounds
		self.v_bounds = self.model.v_bounds
		assert isinstance(self.model.v_bounds, np.ndarray) \
				and self.model.v_bounds.ndim == 2, \
				'Variable bounds must be a 2D numpy array' #
		# Parameter bounds
		self.p_bounds = self.model.p_bounds
		assert isinstance(self.p_bounds, np.ndarray) \
				and self.p_bounds.ndim == 2, \
				'Parameter bounds must be a 2D numpy array'
		# Number of output dimensions
		self.E = self.model.E
		# Verbosity
		self.verbose = True

	def __call__ (self,mu,n_samples=1000):
		Y, X = self.collect_data(mu,n_samples=n_samples)
		return self.train_gps(Y,X)

	def _sample_param (self,mu,n_samples):
		# Draw samples
		# bf1_dv: 0.01
		# lilly: 0.001
		std = 0.01 if self.E == 2 else 0.001
		samps = np.array([mu + std*np.random.randn(len(mu)) \
							for i in range(n_samples)])
		# Enforce parameter bounds
		return np.maximum(np.minimum(samps,self.p_bounds[:,1]), \
						  self.p_bounds[:,0])

	def train_gps(self,Y,X):
		minx, maxx = np.min(X,axis=0), np.max(X,axis=0)
		X = (X-minx)/(maxx-minx)
		miny, maxy = np.min(Y,axis=0), np.max(Y,axis=0)
		Y = (Y-miny)/(maxy-miny)
		class Trainobject:
			def __init__ (self,gp_func,e):
				self.y = Y[:,e][:,None]
				self.gp = gp_func(X,self.y,e)
		trainobjects = [Trainobject(self._gpr_object,e) \
						for e in range(self.E)]
		return [c.gp for c in trainobjects], X, Y, minx, maxx, miny, maxy

	def _gpr_object (self,X,y,e,settings=None):
		dimx, dimp = len(self.v_bounds), len(self.p_bounds)
		dim = dimx+dimp
		kern_x = GPy.kern.RBF(input_dim=dimx,active_dims=range(dimx),\
								name='kernx',ARD=True)
		kern_t = GPy.kern.RBF(input_dim=dimp,active_dims=range(dimx,dim),\
								name='kernp',ARD=True)
		gp = GPy.models.GPRegression(X,y,kern_x*kern_t)
		if settings is None:
			# Avoid numerical problems later
			gp.Gaussian_noise.variance.constrain_fixed(1e-5) # 1e-9
			# lilly - taylor1 : 1e-3
			# lilly - taylor2 : 1e-7
			# Optimise remaining hyperparameters
			gp.optimize()
		else:
			gp.update_model(False)
			gp.initialize_parameter()
			gp[:] = settings
			gp.update_model(True)
		return gp

	def gpr_objects (self,X,Y,settings=None):
		gps = []
		for e in range(self.E):
			gps.append(self._gpr_object(X,Y[:,e][:,None],e, \
				settings=None if settings is None else settings[e]))
		return gps

	def _eval_model (self,X,P,N=None):
		delete_ind = []
		Y = np.zeros((len(P),self.E))
		accepted = 0
		for i,p in enumerate(P):
			try:
				Y[i] = self.model(X[i],p)
				accepted += 1
				if (N is not None) and accepted >= N:
					Y, X, P = Y[:N], X[:N], P[:N]
					break
			except:
				delete_ind.append(i)
				continue
		return  np.delete(Y,delete_ind,axis=0),\
				np.delete(X,delete_ind,axis=0), \
				np.delete(P,delete_ind,axis=0)

	def collect_data (self,mu,n_samples=1000):
		# Sample design variable values
		bnds = self.v_bounds
		X = np.array([bnds[:,0] + (bnds[:,1]-bnds[:,0]) * \
			np.random.rand(len(bnds)) for i in range(2*n_samples)])
		# Sample parameter values
		P = self._sample_param(mu,2*n_samples)
		# Evaluate model at sampled parameter values
		Y,X,P = self._eval_model(X,P,n_samples)
		assert len(Y) == len(X) == len(P), \
				'Data output and input have different sizes'
		if self.verbose: print("    " + self.model.name + \
				" training data collected: " + str(len(P)))
		return Y, np.c_[X,P]





