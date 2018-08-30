
import numpy as np 
import matplotlib.pyplot as plt


class Graphics:
    
    def plot_predictions (self, X, M, S):
        # Create figure and axes
        plt.figure(figsize=(14,5));
        axs = [plt.subplot2grid((1,2),(0,e)) for e in [0,1]]
        axs[0].set_ylabel(r"$f_{i,(1)}$",fontsize=14)
        axs[1].set_ylabel(r"$f_{i,(2)}$",fontsize=14)
        for ax in axs:
            ax.set_xlabel(r"$x$",fontsize=14)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

        def plot_output (e):
            for i, c, l in zip( [0,1], ['r', 'b'], ['-','--']):
                # Mean
                axs[e].plot(X, M[:,i,e],c=c, linestyle=l)
                # Mean with one standard deviations
                ms = [ M[:,i,e] + (-1)**j * np.sqrt(S[:,i,e,e]) for j in [0,1] ]
                axs[e].fill_between(X, ms[0], ms[1], facecolor=c, alpha=0.2)
                
        for e in [0,1]:
            plot_output(e)
        axs[0].legend([r"$\mathcal{M}_1$",r"$\mathcal{M}_2$"],loc=9,fontsize=14)
        plt.show()

    def plot_designs (self, X, dcs):
        names = ['HR', 'BH', 'BF', 'AW', 'JR']
        cols  = ['r','m','b','g','xkcd:mustard']
        lines = ['-','--','-.',':', '-']

        # Create figure and axes
        fig, ax = plt.subplots(1,1)
        ax.set_ylabel(r"$D(x)$",fontsize=14)
        ax.set_xlabel(r"$x$",fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        for dc,c,l,n in zip(dcs,cols,lines,names):
            dc = (dc-np.min(dc))/(np.max(dc)-np.min(dc))
            plt.plot(X, dc, c=c, linestyle=l, label=n)
        #ax.legend(names,loc=0,fontsize=14,ncol=2)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=14)
        plt.show()

