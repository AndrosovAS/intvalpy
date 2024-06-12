import numpy as np
import matplotlib.pyplot as plt


class IPlot:

    def __init__(self, figsize=(8, 8)):
        self.figsize = figsize


    def subplots(self, nrows=None, ncols=None, figsize=None):
        self.nrows = 1 if nrows is None else nrows
        self.ncols = 1 if ncols is None else ncols
        if not figsize is None:
            self.figsize = figsize

        self.fig, self.ax = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)
        return self.fig, self.ax

    def set_ax(self, axindex):
        if self.nrows == 1 and self.ncols == 1:
            return self.ax
        else:
            if axindex is None:
                if self.nrows > 1 and self.ncols > 1:
                    axindex = (0, 0)
                else:
                    axindex = 0
            return self.ax[axindex]


    def scatter(self, x, y, color='gray', alpha=0.5, s=10, axindex=None):
        ax = self.set_ax(axindex)

        flatx = x.a == x.b
        flaty = y.a == y.b

        index_dot = flatx & flaty
        if index_dot.any():
            ax.scatter(x.a[index_dot], y.a[index_dot], s=s, color=color, alpha=alpha)


        full_fill = (flatx | flaty) & (~index_dot)
        scinny_fill = ~(flatx | flaty) & (~index_dot)
        ox = np.array([x.a, x.a, x.b, x.b])
        oy = np.array([y.a, y.b, y.b, y.a])

        ax.fill(ox[:, full_fill], oy[:, full_fill], alpha=alpha, fill=False, color=color)
        ax.fill(ox[:, scinny_fill], oy[:, scinny_fill], alpha=alpha, color=color)


    def lineqs(self, vertices, color='gray', alpha=0.5, s=10, axindex=None):
        ax = self.set_ax(axindex)

        x, y = vertices[:, 0], vertices[:, 1]
        ax.fill(x, y, linestyle='-', linewidth=1, color=color, alpha=alpha)
        ax.scatter(x, y, s=s, color='black', alpha=1)


    def IntLinIncR2(self, vertices, color='gray', alpha=0.5, s=10, axindex=None):
        for v in vertices:
            if len(v)>0:
                self.lineqs(v, color=color, alpha=alpha, s=s, axindex=axindex)






