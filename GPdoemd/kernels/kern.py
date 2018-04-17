

class Kern:
	def d_r_d_x (self, X1, X2):
		return NotImplementedError

	def d2_r_d_x2 (self, X1, X2):
		return NotImplementedError

	def d_k_d_x (self, X1, X2):
		return NotImplementedError

	def d2_k_d_x2 (self, X1, X2):
		return NotImplementedError
