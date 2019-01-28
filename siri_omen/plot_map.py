"""
Make 2D map plots.
"""
import os
import numpy
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.grid_finder as grid_finder
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from .plot_taylor_diag import get_point_style_cycler
from . import utility


class GeographicPlot(object):
    """
    A map plot object.
    """
    def __init__(self, fig=None, rect=None, projection=None):

        if projection is None:
            projection = ccrs.Mercator(
                central_longitude=20., min_latitude=0.,
                max_latitude=80., latitude_true_scale=60.0
            )

        self.fig = plt.figure(figsize=(12, 12)) if fig is None else fig
        if rect is None:
            rect = 111

        ax = self.fig.add_subplot(rect, projection=projection)
        self.ax = ax

        ax.set_extent([6, 32, 53, 66])

        gl = ax.gridlines(draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        self.point_style = get_point_style_cycler()
        self.point_style_iter = iter(self.point_style)

        self.sample_points = []

    def add_station(self, x, y, label, label_to_legend=False, textargs=None, **kwargs):
        if label_to_legend:
            kwargs.setdefault('label', label)
        else:
            kwargs.pop('label', None)
        sty = next(self.point_style_iter)
        sty.update(kwargs)
        p = self.ax.plot(x, y, transform=ccrs.Geodetic(), **sty)
        if label_to_legend:
            self.sample_points.append(p)
        else:
            if textargs is None:
                textargs = {}
            textargs.setdefault('horizontalalignment', 'left')
            textargs.setdefault('verticalalignment', 'top')
            textargs.setdefault('fontsize', 8)
            self.ax.text(x+0.1, y, label, transform=ccrs.Geodetic(),
                         **textargs)

    def add_feature(self, feature_name):
        if feature_name == 'land':
            land_feature = cfeature.NaturalEarthFeature(
                'physical', 'land', '50m', edgecolor='none', facecolor='0.8')
            self.ax.add_feature(land_feature)
        elif feature_name == 'coastlines':
            self.ax.coastlines(resolution='50m')

    def add_legend(self, **kwargs):
        """
        Add a legend to the plot
        """
        kwargs.setdefault('prop', dict(size='small'))
        kwargs.setdefault('loc', 'upper left')
        kwargs.setdefault('bbox_to_anchor', (1.01, 1.0))
        nsamples = len(self.sample_points)
        ncolumns = int(numpy.ceil(float(nsamples) / 40))
        kwargs.setdefault('ncol', ncolumns)
        self.ax.legend(numpoints=1, **kwargs)

    def add_title(self, title):
        """
        Add a title to the plot
        """
        self.ax.set_title(title)
