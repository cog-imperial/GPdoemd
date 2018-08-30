
import numpy as np 
import matplotlib.pyplot as plt

from GPdoemd.models import GPModel
from GPdoemd.kernels import RBF, Matern52
from GPdoemd.marginal import taylor_first_order
from GPdoemd.param_estim import diff_evol
from GPdoemd.param_covar import laplace_approximation
from GPdoemd.design_criteria import HR, BH, BF, AW, JR
from GPdoemd.case_studies.analytic import mixing
from GPdoemd.discrimination_criteria import aicw



class Graphics:
    """
    Case study demo graphics
    - For producing the plots
    """
    def __init__ (self, X, Y, mvar):
        self.X      = X 
        self.Y      = Y
        assert len(X) == len(Y)
        self.N      = len( X )
        self.mvar   = mvar
        self.mstd   = np.sqrt(self.mvar)
        self.tau    = np.linspace(1.,100.,200)
        self.cols   = ['r','m','b','g','xkcd:mustard']
        self.legend = [ r"$\mathcal{M}_1$",
                        r"$\mathcal{M}_2$",
                        r"$\mathcal{M}_3$",
                        r"$\mathcal{M}_4$",
                        r"$\mathcal{M}_5$" ]

    def get_ax (self, N=None, height=4):
        if N is None:
            N = self.N
        plt.figure( figsize=(np.min( (4*N, 15)), height) )
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.tight_layout(True)
        axs = [plt.subplot2grid((1,N),(0,i)) for i in range(N)]
        for ax in axs:
            ax.set_ylabel(r"$y$",fontsize=14)
            ax.set_xlabel(r"$x_1$",fontsize=14)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_ylim([-0.05, 1.05])
            ax.set_xlim([-3, 103])
        return axs

    def plot_data (self, axs):
        for y, x, ax in zip(self.Y, self.X, axs):
            ax.errorbar(x[0],y[0],yerr=2*self.mstd[0],c='k',ecolor='k',fmt='x')

    def plot_prediction (self, ax, c, mu, s2=None):
        mu = mu.flatten()
        ax.plot(self.tau, mu, c=c)
        if s2 is not None:
            s2 = np.sqrt(s2.flatten())
            ax.fill_between(self.tau, mu+2*s2, mu-2*s2, facecolor=c, alpha=0.2)


    def plot_fitted_models (self, Ms):
        axs  = self.get_ax()
        self.plot_data(axs)
        for x, ax in zip(self.X, axs):
            for M, c in zip(Ms, self.cols):
                y = np.array([ M.call([t,x[1],x[2]], M.pmean) for t in self.tau ])
                self.plot_prediction(ax,c,y)
        axs[1].legend(self.legend,loc=3,fontsize=14)
        plt.suptitle('Parameter estimation - models fitted to data')
        plt.tight_layout(True)
        plt.subplots_adjust(top=0.85)
        plt.show()

    def gp_plot (self, Ms):
        axs  = self.get_ax()
        self.plot_data(axs)
        for x, ax in zip(self.X, axs):
            xnew = np.array([[t, x[1], x[2]] for t in self.tau])
            for M, c in zip(Ms, self.cols):
                mu,s2 = M.predict( xnew )
                self.plot_prediction(ax, c, mu, s2)
        axs[1].legend(self.legend,loc=3,fontsize=14)
        plt.suptitle('Surrogate model predictions (for best-fit parameter values)')
        plt.tight_layout(True)
        plt.subplots_adjust(top=0.85)
        plt.show()

    def marginal_plot (self, Ms):
        axs  = self.get_ax()
        self.plot_data(axs)
        for x, ax in zip(self.X, axs):
            xnew = np.array([[t, x[1], x[2]] for t in self.tau])
            for M, c in zip(Ms, self.cols):
                mu,s2 = taylor_first_order( M, xnew )
                self.plot_prediction(ax, c, mu, s2)
        axs[1].legend(self.legend,loc=3,fontsize=14)
        plt.suptitle('Marginal predictive distributions')
        plt.tight_layout(True)
        plt.subplots_adjust(top=0.85)
        plt.show()

    def design_plot (self, Ms, DCs, designs):
        axs  = self.get_ax(len(DCs), height=3)
        for x, dc, ax in zip(designs, DCs, axs):
            ax.set_title(dc)
            ax.plot([x[0],x[0]],[0,1],c='k',linestyle='--')
            
            xnew = np.array([[t, x[1], x[2]] for t in self.tau])
            for M, c in zip(Ms, self.cols):
                mu,s2 = taylor_first_order( M, xnew )
                self.plot_prediction(ax, c, mu, s2)
        plt.suptitle('Marginal predictive distributions at suggested next designs')
        plt.tight_layout(True)
        plt.subplots_adjust(top=0.85)
        plt.show()






##################
###            ###
###  D E M O   ###
###            ###
##################

print( "== Loading case study ==" )
measurement, models = mixing.get(i=2)
print( "Number of models: %d" % len(models) )

x_bounds = measurement.x_bounds   # Design variable bounds
dim_x    = len( x_bounds )        # Number of design variables
measvar  = measurement.measvar    # Measurement noise variance
E        = 1                      # Number of outputs/target dimensions

# Initial observations.
X = np.array([[ 20., 0.50, 0 ],
              [ 20., 0.75, 0 ]])
Y = np.array([[ 0.766 ],
              [ 0.845 ]])


"""
Initialise the GP surrogate models
"""
Ms = []
print( "== Initialising models ==" )
for m in models:
    dim_p = len( m.p_bounds )
    d = {
        'name':             m.name,       # Model name
        'call':             m,            # Function handle for evaluating the model
        'dim_x':            dim_x,        # Number of design variables
        'dim_p':            dim_p,        # Number of model parameters
        'p_bounds':         m.p_bounds,   # Bounds on the model parameters
        'num_outputs':      E,            # Number of outputs
        'binary_variables': [2],          # x_3 is a binary design variable
        'meas_noise_var':   measvar       # Measurement noise variance
    }
    Ms.append( GPModel(d) )               # Add initialised GPModel to list
    print( "Initialised model %s" % m.name )

# Initialise model posterios
# - all models equally probable for now
pis = np.array([ 1. / len(Ms) for _ in Ms ])


"""
Design more experiments while we do not have a 'winning' model
and we have not exhausted the experimental budget
"""
max_no_experiments = 10  # Experimental budget

for _ in range( max_no_experiments ):
    # Graphics object
    # - custom interface for plotting the results in this demo
    graphics = Graphics(X, Y, measvar)

    """
    Parameter estimation
    """
    print( "== Parameter estimation ==" )
    for M in Ms:
        # Estimate parameters with differential evolution (diff_evol) method
        M.param_estim(X, Y, diff_evol, M.p_bounds)
        print( "Model %s's parameter(s) estimated: %s " % (M.name, str(M.pmean)) )

    # Best-fit model predictions
    graphics.plot_fitted_models(Ms)


    """
    Train surrogate models
    """
    # Mesh grid of design variable values
    def generate_x_mesh ():
        X = np.meshgrid( np.linspace(*x_bounds[0], num=20),   # x_1
                         np.linspace(*x_bounds[1], num=20),   # x_2
                         [0, 1])                              # x_3 - binary
        return np.vstack( map(np.ravel, X) ).T  # stack columns

    # Sample N model parameter values around best-fit value pmean
    def generate_p_values (pmean, N):
        return np.random.uniform(0.995*pmean, 1.005*pmean, size=[N, len(pmean)])

    # Generate training data
    def training_data (model):
        Xsim = generate_x_mesh()
        Xsim = np.vstack( (Xsim, Xsim) )
        Psim = generate_p_values( M.pmean, len(Xsim) )
        Zsim = np.c_[Xsim, Psim]
        Ysim = np.array([M.call(x, p) for x, p in zip(Xsim, Psim)])
        return Zsim, Ysim
        
    print( "== Train surrogate models ==" )
    for M in Ms:
        # Generate training data
        Zsim, Ysim = training_data(M)
        M.set_training_data(Zsim, Ysim)

        # Choosing covariance functions
        # - we make an arbitrary choice here
        M.kern_x = Matern52  # Design variable covariance function
        M.kern_p = RBF       # Model parameter covariance function

        # Learn GP hyperparameters (evidence maximisation)
        M.gp_surrogate()     # Initialise GP models
        M.gp_optimise()      # Optimise GP model hyperparameters
        print( "Trained model %s" % M.name )

    # Plot GP predictions
    graphics.gp_plot(Ms)


    """
    Model parameter covariance
    """
    print( "== Model parameter covariance ==" )
    for M in Ms:
        M.Sigma = laplace_approximation(M, X)
    # Compute the marginal predictive distributions of each model.
    graphics.marginal_plot(Ms)


    """
    Model posteriors
    """
    print( "== Model posteriors ==" )
    mu = np.zeros((len(X), len(Ms), E))     # Means
    s2 = np.zeros((len(X), len(Ms), E, E))  # Covariances
    D  = np.array([M.dim_p for M in Ms])    # No. of model parameters for each model

    for i, M in enumerate(Ms):
        # First-order Taylor approximation of marginal predictive distribution
        mu[:,i], s2[:,i] = taylor_first_order( M, X )

    # Compute normalised Akaike weights
    pis = aicw(Y, mu, s2, D)
    print( " - Normalised Akaike weights" )
    for M, p in zip(Ms, pis):
        print( "%s: %.5f" % (M.name, p) )

    # Check for winner
    if np.any(pis >= 0.999):
        print( "Model %s is the winner!" % Ms[np.argmax(pis)].name )
        break


    """
    Design the next experiment
    """
    Xtest = generate_x_mesh()                      # Test points
    mu    = np.zeros((len(Xtest), len(Ms), E))     # Means
    s2    = np.zeros((len(Xtest), len(Ms), E, E))  # Covariances

    # Marginal predictive distributions
    for i, M in enumerate(Ms):
        mu[:,i], s2[:,i] = taylor_first_order( M, Xtest )

    Xnext = []
    print('Recommended experiments:\n       x_1    x_2   x_3')
    for DC in ['HR', 'BH', 'BF', 'AW', 'JR']:
        dc = eval( DC + '(mu, s2, measvar, pis)' )  # Call design criterion
        xn = Xtest[ np.argmax(dc) ]                 # Optimal next experiment
        Xnext.append(xn)
        print('%s: %6.1f %6.2f %5.0f' % (DC, *xn))

    graphics.design_plot(Ms, ['HR', 'BH', 'BF', 'AW', 'JR'], Xnext)


    """
    Run next experiment
    """
    print( "== Running new experiment ==" )
    # We choose the experiment chosen by the JR design criterion
    xnext = Xnext[-1]
    ynext = measurement(xnext)

    X = np.vstack(( X, xnext ))
    Y = np.vstack(( Y, ynext ))


