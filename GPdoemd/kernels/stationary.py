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

import GPy.kern


"""
RBF / squared exponntial kernel
"""
class RBF (GPy.kern.RBF):
	def __init__ (self, d, drange, name):
		GPy.kern.RBF.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)

"""
Exponential kernel
"""
class Exponential (GPy.kern.Exponential):
	def __init__ (self, d, drange, name):
		GPy.kern.Exponential.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)

	def dK2_drdr(self, r):
		return self.K_of_r(r)

"""
Matern-3/2 kernel
"""
class Matern32 (GPy.kern.Matern32):
	def __init__ (self, d, drange, name):
		GPy.kern.Matern32.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)

	def dK2_drdr(self, r):
		ar = np.sqrt(3.)*r
		return 3. * self.variance * (ar - 1) * np.exp(-ar)

"""
Matern-5/2 kernel
"""
class Matern52 (GPy.kern.Matern52):
	def __init__ (self, d, drange, name):
		GPy.kern.Matern52.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)

	def dK2_drdr(self, r):
		ar = np.sqrt(5)*r 
		return 5./3. * self.variance * (ar**2 - ar - 1) * np.exp(-ar)

"""
Cosine kernel
"""
class Cosine (GPy.kern.Cosine):
	def __init__ (self, d, drange, name):
		GPy.kern.Cosine.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)

	def dK2_drdr(self, r):
		return -self.K_of_r(r)

"""
Rational quadratic kernel
"""
class RatQuad(GPy.kern.RatQuad):
	def __init__ (self, d, drange, name):
		GPy.kern.RatQuad.__init__(
			self, input_dim=d, active_dims=drange, name=name, ARD=True)

	def dK2_drdr(self, r):
		r2  = np.square(r)
		lp  = np.log1p(r2 / 2.)
		a   = (self.power + 1) * lp
		dr1 = -self.variance * self.power * np.exp(-a)
		a  += lp
		dr2 = self.variance * self.power * (self.power + 1) * r2 * np.exp(-a)
		return dr1 + dr2