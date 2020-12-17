"""
Target diagram implementation.

"""
import os
import numpy
import matplotlib.pyplot as plt
from . import utility
from .plot_taylor_diag import get_point_style_cycler, CubeStyler, cycler
from .plot_taylor_diag import _compute_pairwise_stats
from . import plot_taylor_diag as taylor

__all__ = [
    'TargetDiagram',
    'plot_target_diagram',
    'plot_normalized_target_diagram',
    'plot_normalized_taylor_target_diagram',
    'save_target_diagram',
    'save_taylor_target_diagram',
]


class TargetDiagram(object):
    """
    Target diagram.

    Plot model bias and centered root mean square error to reference (data)
    sample in a single plot, with y=bias and x=crmse. Then radial distance from
    the origin is root mean square error.
    """

    def __init__(self, fig=None, rect=None, datalim=None):
        """
        Set up Target diagram axes.

        Parameters:

        * fig: input Figure or None
        * rect: subplot definition
        """

        self.fig = plt.figure() if fig is None else fig
        if rect is None:
            rect = 111

        ax = self.fig.add_subplot(rect)
        self.ax = ax

        # Adjust axes
        ax.set_aspect('equal', adjustable='box')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlabel('cRMSE', ha='right', va='bottom')
        ax.set_ylabel('BIAS', ha='right', va='top')
        ax.xaxis.set_label_coords(1.0, 0.505)
        ax.yaxis.set_label_coords(0.51, 1.0)

        self.max_datalim = 0.
        self.fixed_axes = False
        if datalim is not None:
            self._update_axes(datalim, expand_factor=1.0)
            self.fixed_axes = True
        else:
            self._update_axes(1.01, expand_factor=1.0)

        # set color and marker styles
        self.ax.set_prop_cycle(get_point_style_cycler())

        # Collect sample points for latter use (e.g. legend)
        self.sample_points = []

    def _update_axes(self, *new_values, expand_factor=1.1):
        new_max = max([abs(v) for v in new_values])
        if new_max > self.max_datalim:
            self.max_datalim = new_max
            bounds = [-expand_factor * self.max_datalim,
                      expand_factor * self.max_datalim]
            if not self.fixed_axes:
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
        ncolumns = int(numpy.ceil(float(nsamples) / 30))
        kwargs.setdefault('ncol', ncolumns)
        self.ax.legend(numpoints=1, **kwargs)

    def add_title(self, title):
        """
        Add a title to the plot
        """
        self.ax.set_title(title)


def _plot_target(cube_pairs, normalized,
                 pair_stats=None,
                 fig=None, rect=None, add_legend=True,
                 datalim=None,
                 label_attr='dataset_id', title=None,
                 label_alias=None, style=None,
                 styler_args=None, legend_args=None):

    if fig is None:
        fig = plt.figure(figsize=(9, 9))
    dia = TargetDiagram(fig=fig, rect=rect, datalim=datalim)
    dia.add_grid()

    if pair_stats is None:
        pair_stats = _compute_pairwise_stats(cube_pairs, normalized,
                                             add_crmse_sign=True)

    if styler_args is not None:
        mod_list = [t[1] for t in cube_pairs]
        styler = CubeStyler(mod_list, *styler_args)

    seen = set()
    for o, m, o_stats, m_stats in pair_stats:
        label = m.attributes.get(label_attr)
        if label_alias is not None:
            label = label_alias.get(label, label)
        if label in seen:
            label = '_' + label
        else:
            seen.add(label)
        kw = {}
        if styler_args is not None:
            kw.update(styler.get_style(m))
        if style is not None:
            key = m.attributes.get(label_attr)
            if key is not None and key in style:
                kw.update(style[key])
        dia.add_sample(m_stats['crmse'], m_stats['bias'],
                       label=label, **kw)

    if add_legend:
        kw = {}
        if legend_args is not None:
            kw.update(legend_args)
        dia.add_legend(**kw)

    if title is None:
        var_list = [p[0].standard_name.replace('_', ' ') for p in cube_pairs]
        var_str = ' '.join(utility.unique(var_list))
        dataset_list = [p[0].attributes['dataset_id'] for p in cube_pairs]
        dataset_list += [p[1].attributes['dataset_id'] for p in cube_pairs]
        dataset_str = ' '.join(utility.unique(dataset_list))
        title = dataset_str + ': ' + var_str
    dia.add_title(title)

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

    contours = dia.add_contours(colors='0.7', zorder=0, linewidths=1.0)
    plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

    return dia


def plot_target_diagram(reference, cube_list, **kwargs):
    normalized = False
    cube_pairs = [(reference, c) for c in cube_list]
    return _plot_target(cube_pairs, normalized, **kwargs)


def plot_normalized_target_diagram(cube_pairs, **kwargs):
    normalized = True
    return _plot_target(cube_pairs, normalized, **kwargs)


def save_target_diagram(cube_pairs, output_dir=None,
                        plot_root_dir=None, **kwargs):
    """
    Makes a default target diagram and saves it to disk.
    """
    dia = plot_normalized_target_diagram(cube_pairs, **kwargs)
    fig = dia.fig

    cube_list = [p[0] for p in cube_pairs] + [p[1] for p in cube_pairs]
    imgfile = utility.generate_img_filename(cube_list,
                                            loc_str='stations',
                                            output_dir=output_dir,
                                            root_dir=plot_root_dir,
                                            prefix='target')
    dir, filename = os.path.split(imgfile)
    utility.create_directory(dir)

    print('Saving image {:}'.format(imgfile))
    fig.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _plot_taylor_target(cube_pairs, normalized, **kwargs):

    fig = plt.figure(figsize=(18, 9))
    plt.subplots_adjust(wspace=0.08)

    target_datalim = kwargs.pop('target_datalim', 1.2)
    pair_stats = kwargs.pop('pair_stats', None)
    if pair_stats is None:
        pair_stats = _compute_pairwise_stats(cube_pairs, normalized,
                                             add_crmse_sign=True)
    title = kwargs.pop('title', None)
    add_legend = kwargs.pop('add_legend', True)
    taylor_dia = taylor.plot_normalized_taylor_diagram(
        cube_pairs, pair_stats=pair_stats,
        fig=fig, rect=121, title='', add_legend=False, **kwargs)
    target_dia = plot_normalized_target_diagram(
        cube_pairs, pair_stats=pair_stats,
        fig=fig, rect=122, datalim=target_datalim,
        add_legend=add_legend, **kwargs)
    if title is None:
        title = target_dia.ax.get_title()
    target_dia.ax.set_title(title, x=-0.04)

    return taylor_dia, target_dia


def plot_normalized_taylor_target_diagram(cube_pairs, **kwargs):
    normalized = True
    return _plot_taylor_target(cube_pairs, normalized, **kwargs)


def save_taylor_target_diagram(cube_pairs, output_dir=None,
                               plot_root_dir=None, **kwargs):
    """
    Makes a default taylor-target diagram and saves it to disk.
    """
    tay_dia, tar_dia = plot_normalized_taylor_target_diagram(cube_pairs,
                                                             **kwargs)
    fig = tay_dia.fig

    cube_list = [p[0] for p in cube_pairs] + [p[1] for p in cube_pairs]
    imgfile = utility.generate_img_filename(cube_list,
                                            loc_str='stations',
                                            output_dir=output_dir,
                                            root_dir=plot_root_dir,
                                            prefix='taylortarget')
    dir, filename = os.path.split(imgfile)
    utility.create_directory(dir)

    print('Saving image {:}'.format(imgfile))
    fig.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
