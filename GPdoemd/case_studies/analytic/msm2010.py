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
class MSM2010Model:
	"""
	C. Michalik, M. Stuckert, and W. Marquardt 
	"Optimal experimental design for discriminating numerous model candidates: 
	The AWDC criterion" 
	Ind Eng Chem Res, 49:913–919, 2010.
	"""
	@property
	def n_outputs (self):
		return 1
	@property
	def x_bounds (self):
		return np.array([[0.,10.],[0.,10.],[0.,100.]])
	@property
	def p_bounds (self):
		return np.array([[0.,1000.]])

	def __call__ (self, x, p, grad=False):
		C  = self.f(x)
		dC = np.array([C])
		return C*p if not grad else [C*p, dC]

"""
Models
"""
class M1 (MSM2010Model):
	@property
	def name (self):
		return 'M1'
	def f (self,x):
		return np.array([1.])
		
class M2 (MSM2010Model):
	@property
	def name (self):
		return 'M2'
	def f (self,x):
		return x[[0]]
		
class M3 (MSM2010Model):
	@property
	def name (self):
		return 'M3'
	def f (self,x):
		return x[[1]]
		
class M4 (MSM2010Model):
	@property
	def name (self):
		return 'M4'
	def f (self,x):
		return x[[2]]
		
class M5 (MSM2010Model):
	@property
	def name (self):
		return 'M5'
	def f (self,x):
		return x[[0]] * x[[1]]
		
class M6 (MSM2010Model):
	@property
	def name (self):
		return 'M6'
	def f (self,x):
		return x[[0]] * x[[2]]
		
class M7 (MSM2010Model):
	@property
	def name (self):
		return 'M7'
	def f (self,x):
		return x[[1]] * x[[2]]
		
class M8 (MSM2010Model):
	@property
	def name (self):
		return 'M8'
	def f (self,x):
		return x[[0]] * x[[1]] * x[[2]]

class M9 (MSM2010Model):
	@property
	def name (self):
		return 'M9'
	def f (self,x):
		return x[[0]]**2 * x[[1]]

class M10 (MSM2010Model):
	@property
	def name (self):
		return 'M10'
	def f (self,x):
		return x[[0]] * x[[1]]**2

"""
Data generator
"""
class DataGen (M9):
	@property
	def truemodel (self):
		return 8
	@property
	def measvar (self):
		return np.array([1.])
	@property
	def p (self):
		return [1.]

	def __call__ (self,x):
		state = super(DataGen, self).__call__(x,self.p)
		noise = np.sqrt(self.measvar) * np.random.randn(self.n_outputs)
		return state + noise

"""
Get model functions
"""
def get (*args):
	return DataGen(), [M1(),M2(),M3(),M4(),M5(),M6(),M7(),M8(),M9(),M10()]




