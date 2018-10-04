
from .model import Model

from .analytic_model import AnalyticModel
from .numerical_model import NumericalModel

from .surrogate_model import SurrogateModel
from .gp_model import GPModel
from .sparse_gp_model import SparseGPModel

try:
	import gp_grief.models
	from .gp_grief_model import GPGriefModel
except:
	print('Could not import GPGriefModel - ensure gp_grief package is installed.')
	print('NOTE: install forked version from https://github.com/scwolof/gp_grief')