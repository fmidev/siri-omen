"""
Target diagram implementation.

"""
import os
import numpy
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.grid_finder as grid_finder
from cycler import cycler
from . import utility
from . import statistics
from .plot_taylor_diag import get_point_style_cycler, get_cube_stats


class TargetDiagram(object):
    """
    Target diagram.

    Plot model bias and centered root mean square error to reference (data)
    sample in a single plot, with y=bias and x=crmse. Then radial distance from
    the origin is root mean square error.
    """

    def __init__(self, fig=None, rect=111):
        """
        Set up Target diagram axes.

        Parameters:

        * fig: input Figure or None
        * rect: subplot definition
        """

        if fig is None:
            self.fig = plt.figure()
        else:
            self.fig = fig

        ax = self.fig.add_subplot(rect)
        self.ax = ax

        # Adjust axes
        #ax.set_aspect('equal')
        ax.set_aspect('equal', adjustable='box')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        #ax.spines['left'].set_smart_bounds(True)
        #ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlabel('cRMSE', ha='right', va='bottom')
        ax.set_ylabel('BIAS', ha='right', va='top')
        ax.xaxis.set_label_coords(1.0, 0.505)
        ax.yaxis.set_label_coords(0.51, 1.0)

        self.max_datalim = 0.
        self._update_axes(1.01, expand_factor=1.0)

        # set color and marker styles
        self.ax.set_prop_cycle(get_point_style_cycler())

        # Collect sample points for latter use (e.g. legend)
        self.sample_points = []

    def _update_axes(self, *new_values, expand_factor=1.1):
        new_max = max([abs(v) for v in new_values])
        if new_max > self.max_datalim:
            self.max_datalim = new_max
            bounds = [-expand_factor*self.max_datalim,
                      expand_factor*self.max_datalim]
            self.ax.set_xlim(bounds)
            self.ax.set_ylim(bounds)

    def add_sample(self, crmse, bias, *args, **kwargs):
        """
        Add sample (*crmse*, *bias*) to the diagram. *args* and *kwargs* are
        directly propagated to the `Figure.plot` command.
        """
        kwargs.setdefault('zorder', 2)
        kwargs.setdefault('alpha', 0.7)
        l, = self.ax.plot(crmse, bias, *args, **kwargs)
        self.sample_points.append(l)
        self._update_axes(crmse, bias)
        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""
        kwargs.setdefault('zorder', 0)
        self.ax.grid(*args, **kwargs)

    def add_contours(self, levels=None, **kwargs):
        """
        Add constant centered RMSE difference contours, defined by *levels*.
        """
        bounds = self.ax.get_xlim()
        xs, ys = numpy.meshgrid(numpy.linspace(*bounds),
                                numpy.linspace(*bounds))
        # Compute centered RMSE difference
        rms = numpy.sqrt(xs**2 + ys**2)
        if levels is not None:
            c_list = numpy.linspace(0, bounds[1], levels + 1)[1:]
        else:
            ticks = self.ax.get_xticks()
            select_ticks = (ticks > 0) * (ticks <= bounds[1])
            c_list = ticks[select_ticks]
        contours = self.ax.contour(xs, ys, rms, c_list, **kwargs)
        return contours

    def add_legend(self, **kwargs):
        """
        Add a legend to the plot
        """
        kwargs.setdefault('prop', dict(size='small'))
        kwargs.setdefault('loc', 'upper left')
        kwargs.setdefault('bbox_to_anchor', (1.01, 1.0))
        nsamples = len(self.sample_points)
        ncolumns = int(numpy.ceil(float(nsamples) / 20))
        kwargs.setdefault('ncol', ncolumns)
        self.ax.legend(numpoints=1, **kwargs)

    def add_title(self, title):
        """
        Add a title to the plot
        """
        self.ax.set_title(title)


def _plot_target(cube_pairs, normalized,
                 label_attr='dataset_id', title=None):

    fig = plt.figure(figsize=(9, 9))
    dia = TargetDiagram(fig=fig)
    dia.add_grid()

    for o, m in cube_pairs:
        obs_stats, mod_stats = get_cube_stats(o, m, normalized,
                                              add_crmse_sign=True)
        label = m.attributes.get(label_attr)
        dia.add_sample(mod_stats['crmse'], mod_stats['bias'],
                       label=label)
    dia.add_legend()

    if title is None:
        var_list = [p[0].standard_name.replace('_', ' ') for p in cube_pairs]
        var_str = ' '.join(utility.unique(var_list))
        title = var_str
        dia.add_title(var_str)

    # update axis labels
    xlabel = dia.ax.get_xlabel()
    ylabel = dia.ax.get_ylabel()
    if normalized:
        dia.ax.set_xlabel('Normalized ' + xlabel)
        dia.ax.set_ylabel('Normalized ' + ylabel)
    else:
        unit_list = utility.unique([str(p[0].units) for p in cube_pairs])
        assert len(unit_list) == 1, \
            'multiple different units found {:}'.format(unit_list)
        unit_str = ' [{:}]'.format(str(unit_list[0]))
        dia.ax.set_xlabel(xlabel + unit_str)
        dia.ax.set_ylabel(ylabel + unit_str)

    contours = dia.add_contours(colors='0.7', zorder=0, linewidth=0.6)
    plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

    return dia


def plot_target_diagram(reference, cube_list, label_attr='dataset_id'):
    normalized = False
    cube_pairs = [(reference, c) for c in cube_list]
    return _plot_target(cube_pairs, normalized, label_attr=label_attr)


def plot_normalized_target_diagram(cube_pairs, label_attr='dataset_id'):
    normalized = True
    return _plot_target(cube_pairs, normalized, label_attr=label_attr)


def save_target_diagram(cube_pairs, output_dir=None, **kwargs):
    """
    Makes a default target diagram and saves it to disk.
    """
    dia = plot_normalized_target_diagram(cube_pairs, **kwargs)
    fig = dia.fig

    cube_list = [p[0] for p in cube_pairs] + [p[1] for p in cube_pairs]
    imgfile = utility.generate_img_filename(cube_list,
                                            loc_str='stations',
                                            root_dir=output_dir,
                                            prefix='target')
    dir, filename = os.path.split(imgfile)
    utility.create_directory(dir)

    print('Saving image {:}'.format(imgfile))
    fig.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
