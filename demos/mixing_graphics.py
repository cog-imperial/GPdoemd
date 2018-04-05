
import numpy as np 
import matplotlib.pyplot as plt

class MixingGraphics:
	def __init__ (self, X, Y, mvar):
		self.X    = X 
		self.Y    = Y
		self.N    = len(X)
		self.mstd = np.sqrt(mvar)
		self.tau  = np.linspace(1.,100.,200)
		self.cols = ['r','m','b','g','xkcd:mustard']

	def get_ax (self, N=None):
		if N is None:
			N = self.N
		plt.figure( figsize=(np.min( (4*N, 15)), 4) )
		return [ plt.subplot2grid((1,N),(0,i)) for i in range(N) ]

	def plot_data (self, axs):
		for y, x, ax in zip(self.Y, self.X, axs):
			ax.errorbar(x[0],y[0],yerr=2*self.mstd[0],c='k',ecolor='k',fmt='x')

	def plot_prediction (self, ax, c, mu, s2=None):
		mu = mu.flatten()
		ax.plot(self.tau, mu, c=c)
		if s2 is not None:
			s2 = np.sqrt(s2.flatten())
			ax.fill_between(self.tau, mu+2*s2, mu-2*s2, facecolor=c, alpha=0.2)

	def model_plot (self, Ms):
		axs  = self.get_ax()
		self.plot_data(axs)
		for x, ax in zip(self.X, axs):
			for M, c in zip(Ms, self.cols):
				y = np.array([ M([t,x[1]], M.pmean) for t in self.tau ])
				self.plot_prediction(ax,c,y)
		plt.show()

	def gp_plot (self, Ms):
		axs  = self.get_ax()
		self.plot_data(axs)
		for x, ax in zip(self.X, axs):
			xnew = np.array([[t, x[1]] for t in self.tau])
			for M, c in zip(Ms, self.cols):
				mu,s2 = M.predict(xnew)
				self.plot_prediction(ax,c,mu,s2)
		plt.show()

	def marginal_plot (self, Ms):
		axs  = self.get_ax()
		self.plot_data(axs)
		for x, ax in zip(self.X, axs):
			xnew = np.array([[t, x[1]] for t in self.tau])
			for M, c in zip(Ms, self.cols):
				mu,s2 = M.marginal_predict(xnew)
				self.plot_prediction(ax,c,mu,s2)
		plt.show()

	def design_plot (self, Ms, DCs, designs):
		axs  = self.get_ax(len(DCs))
		for x, dc, ax in zip(designs, DCs, axs):
			ax.set_title(dc)
			ax.plot([x[0],x[0]],[0,1],c='k',linestyle='--')
			
			xnew = np.array([[t, x[1]] for t in self.tau])
			for M, c in zip(Ms, self.cols):
				mu,s2 = M.marginal_predict(xnew)
				self.plot_prediction(ax,c,mu,s2)
		plt.show()


