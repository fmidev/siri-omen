"""
Taylor diagram (Taylor, 2001) implementation.

Based on implementation by Yannick Copin <yannick.copin@laposte.net>
See: https://gist.github.com/ycopin/3342888
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
from collections import OrderedDict, defaultdict
from itertools import cycle

__all__ = [
    'TaylorDiagram',
    'plot_taylor_diagram',
    'plot_normalized_taylor_diagram',
    'save_taylor_diagram',
    'get_point_style_cycler',
    'get_cube_stats',
    'CubeStyler',
    'cycler',
]


def get_point_style_cycler():
    """
    Returns default color,marker cycler.
    """
    # default color cycler
    color_cy = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                             '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                             '#bcbd22', '#17becf'])
    # marker cycler with sizes
    markerkind_cy = cycler(marker=['d', 'o', 's', 'v', '^',
                                   '*', 'P', '<', '>', 'X'])
    markersize_cy = cycler(markersize=[8, 8, 7, 8, 8, 10, 8, 8, 8, 8])
    marker_cy = markerkind_cy + markersize_cy
    linestyle_cy = cycler(linestyle=[''])
    # loop first over colors, then markers
    style = marker_cy * color_cy * linestyle_cy
    return style


class CubeStyler:
    """
    Associates unique marker style to cube objects.

    CubeStyler(cube_list, ('attribute', cycler), ...)

    e.g. set color for each cube by its `dataset_id`:

        cycler = cycler(color=['r', 'b', 'k'])
        styler = CubeStyler(cube_list, ('dataset_id', cycler))
        for c in cube list:
            kw = styler.get_style(c)
            plot(..., **kw)

    """
    def __init__(self, cube_list, *args):
        """
        Create a Styler object.

        CubeStyler(cube_list, ('attribute', cycler), ...)
        """

        def _get_styles(attrname, cycler):
            # find all unique attribute values
            all_vals = set([c.attributes.get(attrname) for c in cube_list])
            all_vals = sorted(list(all_vals))
            # map to style
            styles = {}
            cy_iter = cycle(cycler)
            for v in all_vals:
                styles[v] = next(cy_iter)
            return styles

        self.styles = OrderedDict()
        for attrname, cycler in args:
            self.styles[attrname] = _get_styles(attrname, cycler)

    def get_style(self, cube):
        out = {}
        for attrname, style in self.styles.items():
            attr = cube.attributes.get(attrname, None)
            out.update(style[attr])
        return out


class TaylorDiagram(object):
    """
    Taylor diagram.

    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(self, refstd, fig=None, rect=None, label='_',
                 srange=(0, 1.5), extend=False):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.

        Parameters:

        * refstd: reference standard deviation to be compared to
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        * srange: stddev axis extension, in units of *refstd*
        * extend: extend diagram to negative correlations
        """

        self.refstd = refstd            # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = numpy.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = numpy.pi
            rlocs = numpy.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = numpy.pi / 2
        tlocs = numpy.arccos(rlocs)        # Conversion to polar angles
        gl1 = grid_finder.FixedLocator(tlocs)    # Positions
        tf1 = grid_finder.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        # set s locator
        gl2 = grid_finder.MaxNLocator(7)    # Max 7 ticks

        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = floating_axes.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1,
            grid_locator2=gl2)

        self.fig = plt.figure() if fig is None else fig
        if rect is None:
            rect = 111

        ax = floating_axes.FloatingSubplot(self.fig, rect, grid_helper=ghelper)
        self.fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")    # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)          # Unused

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        t = numpy.linspace(0, self.tmax)
        r = numpy.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_', zorder=1)

        # Collect sample points for latter use (e.g. legend)
        self.sample_points = [l]

        # set color and marker styles
        self.ax.set_prop_cycle(get_point_style_cycler())

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """
        kwargs.setdefault('zorder', 2)
        kwargs.setdefault('alpha', 0.7)
        l, = self.ax.plot(numpy.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta, radius)
        self.sample_points.append(l)

        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        kwargs.setdefault('zorder', 0)
        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered cRMSE difference contours, defined by *levels*.
        """

        rs, ts = numpy.meshgrid(numpy.linspace(self.smin, self.smax),
                                numpy.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = numpy.sqrt(self.refstd**2 + rs**2
                         - 2 * self.refstd * rs * numpy.cos(ts))
        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        return contours

    def add_legend(self, **kwargs):
        """
        Add a legend to the plot
        """
        kwargs.setdefault('prop', dict(size='small'))
        kwargs.setdefault('loc', 'upper left')
        kwargs.setdefault('bbox_to_anchor', (0.98, 1.0))
        nsamples = len(self.sample_points)
        ncolumns = int(numpy.ceil(float(nsamples) / 30))
        kwargs.setdefault('ncol', ncolumns)
        self.ax.legend(numpoints=1, **kwargs)

    def add_title(self, title):
        """
        Add a title to the plot
        """
        self._ax.set_title(title)


def get_cube_stats(r, p, normalized, add_crmse_sign=False):
    r_stats, p_stats = utility.compute_cube_statistics(r, p)
    if normalized:
        r_stats, p_stats = statistics.normalize_statistics(
            r_stats, p_stats, add_crmse_sign=add_crmse_sign)
    return r_stats, p_stats


def _compute_pairwise_stats(cube_pairs, normalized, add_crmse_sign=False):
    pair_stats = []
    for o, m in cube_pairs:
        try:
            obs_stats, mod_stats = get_cube_stats(o, m, normalized,
                                                add_crmse_sign=add_crmse_sign)
            pair_stats.append((o, m, obs_stats, mod_stats))
        except AssertionError as e:
            print('Cannot compute statistics, skipping: {:} {:}'.format(
                o.attributes['dataset_id'], o.attributes['location_name']))
    return pair_stats


def _plot_taylor(cube_pairs, ref_stddev, normalized,
                 pair_stats=None,
                 fig=None, rect=None, add_legend=True,
                 label_attr='dataset_id', ref_label=None, title=None,
                 label_alias=None, style=None,
                 styler_args=None, legend_args=None):

    if fig is None:
        fig = plt.figure(figsize=(9, 9))
    if ref_label is None:
        ref_label = '_'
    srange = (0, 1.49) if normalized else (0, 1.5)
    dia = TaylorDiagram(ref_stddev, label=ref_label, srange=srange,
                        fig=fig, rect=rect)
    dia.add_grid()
    contours = dia.add_contours(colors='0.6', zorder=0, linewidths=1.1)
    plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

    if pair_stats is None:
        pair_stats = _compute_pairwise_stats(cube_pairs, normalized,
                                             add_crmse_sign=True)
    # print_stats_summary(pair_stats, 'crmse')

    if styler_args is not None:
        mod_list = [t[1] for t in cube_pairs]
        styler = CubeStyler(mod_list, *styler_args)

    seen = set()
    for i, entry in enumerate(pair_stats):
        o, m, o_stats, m_stats = entry
        label = m.attributes.get(label_attr)
        if label_alias is not None:
            label = label_alias.get(label, label)
        if label in seen:
            label = '_' + label
        else:
            seen.add(label)
        kw = {}
        kw['zorder'] = 2 + float(i)/10.
        if styler_args is not None:
            kw.update(styler.get_style(m))
        if style is not None:
            key = m.attributes.get(label_attr)
            if key is not None and key in style:
                kw.update(style[key])
        dia.add_sample(m_stats['stddev'], m_stats['corrcoef'],
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

    # update standard deviation label
    s_ax = dia._ax.axis["left"]
    slabel = s_ax.label.get_text()
    if normalized:
        s_ax.label.set_text('Normalized ' + slabel.lower())
    else:
        unit_list = utility.unique([str(p[0].units) for p in cube_pairs])
        assert len(unit_list) == 1, \
            'multiple different units found {:}'.format(unit_list)
        unit_str = ' [{:}]'.format(str(unit_list[0]))
        s_ax.label.set_text(slabel + unit_str)

    return dia


def plot_taylor_diagram(reference, cube_list, **kwargs):
    normalized = False
    ref_stddev = statistics.compute_stddev(reference.data)
    cube_pairs = [(reference, c) for c in cube_list]
    return _plot_taylor(cube_pairs, ref_stddev, normalized, **kwargs)


def plot_normalized_taylor_diagram(cube_pairs, **kwargs):
    normalized = True
    ref_stddev = 1.0
    return _plot_taylor(cube_pairs, ref_stddev, normalized, **kwargs)


def save_taylor_diagram(cube_pairs, output_dir=None,
                        plot_root_dir=None, **kwargs):
    """
    Makes a default Taylor diagram and saves it to disk.
    """
    dia = plot_normalized_taylor_diagram(cube_pairs, **kwargs)
    fig = dia.fig

    cube_list = [p[0] for p in cube_pairs] + [p[1] for p in cube_pairs]
    imgfile = utility.generate_img_filename(cube_list,
                                            loc_str='stations',
                                            output_dir=output_dir,
                                            root_dir=plot_root_dir,
                                            prefix='taylor')
    dir, filename = os.path.split(imgfile)
    utility.create_directory(dir)

    print('Saving image {:}'.format(imgfile))
    fig.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)


def print_stats_summary(pair_stats, metric):
    data = defaultdict(dict)
    for o, m, o_stats, m_stats in pair_stats:
        model_id = m.attributes['dataset_id']
        loc = m.attributes['location_name']
        data[loc][model_id] = abs(m_stats[metric])
    all_locations = sorted([k for k in data if len(data[k]) == 2])
    model_id_list = sorted(list(set([l for k in data for l in data[k]])))
    print('{:20s}'.format(metric))
    print('{:20s}: {:7s} {:7s}'.format('station', *model_id_list))
    for loc in all_locations:
        vals = [data[loc][id] for id in model_id_list]
        better = vals[1] < vals[0]
        marker = '*' if better else ''
        print('{:20s}: {:7.3f} {:7.3f} {:}'.format(loc, vals[0], vals[1], marker))
