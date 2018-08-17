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

class BoxTransform:
	def __init__ (self, xmin, xmax):
		self.xmin = xmin
		self.xmax = xmax
		self.diff = self.xmax - self.xmin

	def __call__ (self, X, back=False):
		if back:
			return self.xmin + X * self.diff
		return (X - self.xmin) / self.diff

	def var (self, X, back=False):
		if back:
			return X * self.diff**2
		return X / self.diff**2

	def cov (self, X, back=False):
		if back:
			return X * (self.diff[:,None] * self.diff[None,:])
		return X / (self.diff[:,None] * self.diff[None,:])


class MeanTransform:
	def __init__ (self, mean, std):
		self.mean = mean
		self.std  = std

	def __call__ (self, X, back=False):
		if back:
			return self.mean + X * self.std
		return (X - self.mean) / self.std

	def var (self, X, back=False):
		if back:
			return X * self.std**2
		return X / self.std**2

	def cov (self, X, back=False):
		if back:
			return X * (self.std[:,None] * self.std[None,:])
		return X / (self.std[:,None] * self.std[None,:])

	def prediction (self, M, S, back=False):
		Mt = self(M, back=back)
		if S.ndim == 2:
			return M, self.var(S, back=back)
		return M, self.cov(S, back=back)
