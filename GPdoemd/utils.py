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

def binary_dimensions (Z, binary_variables):
	"""
	Outputs
		Range ( len( binary_variables ) )
		Row indices for different binary variables
	"""
	lb = len( binary_variables )
	if lb == 0:
		n1, n2 = Z.shape
		return [0], np.zeros(n1)

	B = np.meshgrid( *( [[-1, 1]] * lb ) )
	B = np.vstack( map( np.ravel, B) ).T

	Zt = Z[:, binary_variables] - 0.5
	J  = []
	for z in Zt:
		for j, b in enumerate(B):
			if np.all(b * z > 0):
				J.append(j)
				break
	return range( 2**lb ), np.array(J)

