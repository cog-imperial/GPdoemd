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
class VTHR2014Model:
	"""
	Case study from
	J. Vanlier, C. A. Tiemann, P. A. J. Hilbers, and N. A. W. van Riel 
	"Optimal experiment design for model selection in biochemical networks" 
	BMC Systems Biology, 8(20), 2014.
	"""
	@property
	def name (self):
		return 'M' + str(self.num_p)
	@property
	def n_outputs (self):
		return 1
	@property
	def x_bounds (self):
		return np.array([[-2.5, 2.5]])
	@property
	def p_bounds (self):
		return np.array([[-10,10]] * self.num_p)

	def __call__ (self, x, P, grad=False):
		F = [	
			lambda t: t,
			lambda t: t**2,
			lambda t: np.sin(0.2 * t**3),
		 	lambda t: np.sin(2 * t) * t
		 	]
		F  = F[:self.num_p]

		f  = np.sum([ p * f(x) for p,f in zip(P,F)], axis=0)
		df = np.array([ f(x) for f in F ]).T
		return f if not grad else [f, df]

"""
Models
"""
class M1 (VTHR2014Model):
	@property
	def num_p (self):
		return 1

class M2 (VTHR2014Model):
	@property
	def num_p (self):
		return 2

class M3 (VTHR2014Model):
	@property
	def num_p (self):
		return 3

class M4 (VTHR2014Model):
	@property
	def num_p (self):
		return 4

"""
Data generator
"""
class DataGen (M3):
	@property
	def truemodel (self):
		return 2
	@property
	def measvar (self):
		return np.array([0.5])
	@property
	def p (self):
		return [-1, 0.5, -6]

	def __call__ (self,x):
		state = super(DataGen, self).__call__(x,self.p)
		noise = np.sqrt(self.measvar) * np.random.randn()
		return state + noise

"""
Get model functions
"""
def get ():
	return DataGen(), [M1(),M2(),M3(),M4()]
