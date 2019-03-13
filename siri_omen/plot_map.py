"""
Make 2D map plots.
"""
import numpy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from .plot_taylor_diag import get_point_style_cycler
from matplotlib.tri import Triangulation


class GeographicPlot(object):
    """
    A map plot object.
    """
    def __init__(self, fig=None, rect=None, projection=None,
                 extent=[6, 32, 53, 66]):

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

        if extent is not None:
            ax.set_extent(extent)

        gl = ax.gridlines(draw_labels=True, zorder=2)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        self.gl = gl

        self.point_style = get_point_style_cycler()
        self.point_style_iter = iter(self.point_style)

        self.sample_points = []

    def add_station(self, x, y, label, label_to_legend=False, textargs=None,
                    transform=None,
                    **kwargs):
        if transform is None:
            transform = ccrs.Geodetic()
        if label_to_legend:
            kwargs.setdefault('label', label)
        else:
            kwargs.pop('label', None)
        kwargs.setdefault('zorder', 3)
        sty = next(self.point_style_iter)
        sty.update(kwargs)
        p = self.ax.plot(x, y, transform=transform, **sty)
        if label_to_legend:
            self.sample_points.append(p)
        else:
            if textargs is None:
                textargs = {}
            textargs.setdefault('horizontalalignment', 'left')
            textargs.setdefault('verticalalignment', 'top')
            textargs.setdefault('fontsize', 8)
            self.add_text(x+0.1, y, label, transform=transform, **textargs)

    def add_text(self, x, y, text, transform=None, **kwargs):
        if transform is None:
            transform = ccrs.Geodetic()
        self.ax.text(x, y, text, transform=transform, **kwargs)

    def add_unstructured_mesh(self, x=None, y=None, connectivity=None,
                              tri=None, transform=None, **kwargs):
        if transform is None:
            transform = ccrs.Geodetic()
        has_xyc = x is not None and y is not None and connectivity is not None
        has_tri = tri is not None
        msg = 'either x, y, connectivity or tri arguments must be provided'
        assert (has_tri and not has_xyc) or (has_xyc and not has_tri), msg

        kwargs.setdefault('linewidth', 0.3)
        kwargs.setdefault('zorder', 2)

        if not has_tri:
            tri = Triangulation(x, y, connectivity)

        p = self.ax.triplot(tri, transform=transform, **kwargs)
        return p

    def add_unstructured_field(self, x=None, y=None, connectivity=None,
                               tri=None, values=None, transform=None,
                               levels=31,
                               **kwargs):
        if transform is None:
            transform = ccrs.Geodetic()
        has_xyc = x is not None and y is not None and connectivity is not None
        has_tri = tri is not None
        msg = 'either x, y, connectivity or tri arguments must be provided'
        assert (has_tri and not has_xyc) or (has_xyc and not has_tri), msg
        assert values is not None, 'values must be provided'

        kwargs.setdefault('zorder', 2)

        if not has_tri:
            tri = Triangulation(x, y, connectivity)

        p = self.ax.tricontourf(tri, values, levels, transform=transform,
                                **kwargs)
        return p

    def add_feature(self, feature_name, scale='50m', **kwargs):
        kwargs.setdefault('zorder', 1)
        if feature_name == 'land':
            land_feature = cfeature.NaturalEarthFeature(
                'physical', 'land', scale, edgecolor='none', facecolor='0.8')
            self.ax.add_feature(land_feature, **kwargs)
        elif feature_name == 'coastlines':
            self.ax.coastlines(resolution='50m', **kwargs)

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

    def add_colorbar(self, p, label=None, pad=0.03, **kwargs):
        # create colorbar
        width = 0.02
        pos = self.ax.get_position().bounds
        x = pos[0] + pos[2] + pad*pos[2]
        cax = self.fig.add_axes([x, pos[1], width, pos[3]])
        cb = plt.colorbar(p, cax=cax, **kwargs)
        if label is not None:
            cb.set_label(label)

    def add_title(self, title):
        """
        Add a title to the plot
        """
        self.ax.set_title(title)
