"""
Timeseries plotting routines
"""
import numpy
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import os
from . import utility
import matplotlib.dates as mdates


plot_unit = {
    'icearea': 'km2',
    'iceextent': 'km2',
    'icevol': 'km3',
}


def plot_timeseries(ax, cube_list, label_attr='dataset_id', time_lim=None,
                    title=None, time_extent=None,
                    label_alias=None,
                    style=None, ylim=None,
                    add_legend=True, legend_kwargs=None,
                    start_time=None, end_time=None, **kwargs):
    """
    Plots time series objects in the given axes.

    :arg cube_list: list of cube time series objects to plot. Can be any
        iterable container.
    """

    def get_label(cube, attr_name):
        return cube.attributes.get(attr_name)

    kwargs.setdefault('linewidth', 1.2)
    for c in cube_list:
        _c = utility.constrain_cube_time(c, start_time, end_time)
        label = get_label(_c, label_attr)
        if label_alias is not None:
            label = label_alias.get(label, label)
        if isinstance(_c.data, numpy.ma.MaskedArray) \
                and numpy.all(_c.data.mask):
            # if all data is bad, skip
            continue
        kw = dict(kwargs)
        if style is not None:
            dataset_id = _c.attributes.get('dataset_id')
            st = style.get(dataset_id, None)
            if st is not None:
                kw.update(st)
        var = utility.map_var_short_name[_c.standard_name]
        if var in plot_unit:
            _c_units = _c.copy()
            _c_units.convert_units(plot_unit[var])
        else:
            _c_units = _c
        qplt.plot(_c_units, axes=ax, label=label, **kw)
    if add_legend:
        lkw = {'loc': 'upper left', 'bbox_to_anchor': (1.02, 1.0)}
        if legend_kwargs is not None:
            lkw.update(legend_kwargs)
        ax.legend(**lkw)
    if title is None:
        loc_names = [c.attributes['location_name'] for c in cube_list]
        dep_strs = ['{:.1f} m'.format(
            c.coord('depth').points.mean()) for c in cube_list]
        titles = utility.unique([' '.join(a) for
                                 a in zip(loc_names, dep_strs)])
        title = ', '.join(titles).strip()
    ax.set_title(title)

    xlim = ax.get_xlim()
    range_days = xlim[1] - xlim[0]

    if range_days < 15:
        major_locator = mdates.DayLocator()
        minor_locator = mdates.HourLocator(interval=6)
    elif range_days < 30:
        major_locator = mdates.DayLocator([1, 5, 10, 15, 20, 25])
        minor_locator = mdates.DayLocator()
    elif range_days < 80:
        major_locator = mdates.DayLocator([1, 10, 20])
        minor_locator = mdates.DayLocator()
    elif range_days < 200:
        major_locator = mdates.DayLocator([1, 15])
        minor_locator = mdates.DayLocator()
    elif range_days < 370:
        major_locator = mdates.MonthLocator()
        minor_locator = mdates.DayLocator([1, 5, 10, 15, 20, 25])
    else:
        major_locator = mdates.AutoDateLocator(minticks=7, maxticks=16,
                                               interval_multiples=False)
        minor_locator = mdates.MonthLocator(bymonthday=[1, 15], interval=1)

    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_minor_locator(minor_locator)
    ax.tick_params(axis='x', which='major', length=7)

    ax.grid(which='major', linewidth=0.7, color='0.7')
    ax.grid(which='minor', linestyle='dashed', linewidth=0.3, color='0.7')

    ax.autoscale(enable=True, axis='x', tight=True)
    if start_time is None and end_time is None and time_extent is not None:
        start_time, end_time = utility.get_common_time_overlap(cube_list,
                                                               time_extent)
    ax.set_xlim(start_time, end_time)
    if time_lim is not None:
        ax.set_xlim(time_lim)
    ax.figure.autofmt_xdate()

    if ylim is not None:
        ax.set_ylim(ylim)


def make_timeseries_figure(cube_list, **kwargs):
    """
    Makes a default time series plot in a new figure.
    """
    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(111)

    time_extent = kwargs.pop('time_extent', None)
    start_time = kwargs.pop('start_time', None)
    end_time = kwargs.pop('end_time', None)
    if start_time is None and end_time is None and time_extent is not None:
        start_time, end_time = utility.get_common_time_overlap(cube_list,
                                                               time_extent)
    plot_timeseries(ax, cube_list, time_extent=time_extent,
                    start_time=start_time, end_time=end_time, **kwargs)

    return fig


def save_timeseries_figure(cube_list, output_dir=None, plot_root_dir=None,
                           imgfile=None, **kwargs):
    """
    Makes a default time series plot and saves it to disk.
    """
    start_time = kwargs.get('start_time', None)
    end_time = kwargs.get('end_time', None)

    fig = make_timeseries_figure(cube_list, **kwargs)
    if imgfile is None:
        imgfile = utility.generate_img_filename(cube_list,
                                                output_dir=output_dir,
                                                root_dir=plot_root_dir,
                                                start_time=start_time,
                                                end_time=end_time)
    dir, filename = os.path.split(imgfile)
    utility.create_directory(dir)

    print('Saving image {:}'.format(imgfile))
    fig.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
