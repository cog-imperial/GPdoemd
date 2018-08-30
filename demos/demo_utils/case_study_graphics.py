
import numpy as np 
import matplotlib.pyplot as plt

from GPdoemd.marginal import taylor_first_order


class Graphics:
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
        plt.show()

    def marginal_plot (self, Ms):
        axs  = self.get_ax()
        self.plot_data(axs)
        for x, ax in zip(self.X, axs):
            xnew = np.array([[t, x[1], x[2]] for t in self.tau])
            for M, c in zip(Ms, self.cols):
                mu,s2 = taylor_first_order( M, xnew )
                self.plot_prediction(ax, c, mu, s2)
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
        plt.show()


