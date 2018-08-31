
try:
	from scipy.optimize import differential_evolution
	from .diff_evol import diff_evol
except:
	print('Could not import param_estim.diff_evol')

try:
	from scipy.optimize import least_squares as lstsq
	from .least_squares import least_squares
except:
	print('Could not import param_estim.least_squares')