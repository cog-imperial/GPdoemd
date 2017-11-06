
import numpy as np 

class Model:
	"""
	Model class

	Initialised using a dictionary of the form
		model_dict = {
						'name': Model name (string)
						'call': Function handle f
						't_bounds': numpy.array([lower_time_bound, upper_time_bounds])
						'v_bounds': numpy.array([[low_v1_bound,up_v1_bound],[low_v2_bound,up_v2_bound],...])
						'p_bounds': numpy.array([[low_p1_bound,up_p1_bound],[low_p2_bound,up_p2_bound],...])
						'num_outputs': (Integer) the number of model outputs/observable states
					}

	The call handle has to accept 
		- f(x,p): return y
			* y [num_outputs]			model predictions given design x and parameters p
	"""
	def __init__ (self,model_dict):
		# Model name
		self.name = model_dict['name']
		assert isinstance(self.name,str), 'Model name has to be a valid str'
		# Model function handle
		self.call = model_dict['call']
		# Parameter bounds
		self.p_bounds = model_dict['p_bounds']
		assert isinstance(self.p_bounds,np.ndarray) and self.p_bounds.ndim == 2 and self.p_bounds.shape[1] == 2, \
				'Parameter bounds for model ' + self.name + ' have to be a numpy array of shape (num_of_param,2)'
		# Variable bounds
		self.v_bounds = model_dict['v_bounds']
		assert isinstance(self.v_bounds,np.ndarray) and self.v_bounds.ndim == 2 and self.v_bounds.shape[1] == 2, \
				'Variable bounds for model ' + self.name + ' have to be a numpy array of shape (num_variables,2)'
		# Number of outputs/observable states
		self.E = model_dict['num_outputs']
		assert isinstance(self.E,int) and self.E > 0, 'Number of outputs has to be an integer value > 0'


	def __call__ (self,x,p):
		return self.call(x,p)



