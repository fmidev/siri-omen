"""
Timeseries plotting routines
"""
from .utility import *  # NOQA

import matplotlib.pyplot as plt
import iris.quickplot as qplt
import os


def get_common_time_overlap(cube_list, mode='union'):
    """
    Find a common overlapping time interval of the cubes.

    :arg cube_list: list of cubes
    :arg mode: either 'union' or 'intersection'. If 'intersection' will return
    the time interval in which all cubes have data. If 'union' will find the
    time span that contains all the data.
    """

    def get_datetime(cube, index):
        t = cube.coord('time')
        return t.units.num2date(t.points[index])

    st_op = min if mode == 'union' else max
    et_op = max if mode == 'union' else min
    start_time = st_op([get_datetime(c, 0) for c in cube_list])
    end_time = et_op([get_datetime(c, -1) for c in cube_list])
    assert end_time > start_time, 'Could not find overlapping time stamps'
    return start_time, end_time


def generate_img_filename(cube_list, root_dir=None,
                          start_time=None, end_time=None,
                          time_extent='union'):
    """
    Generate a canonical name for a timeseries image file.
    """
    prefix = 'ts'

    var_str = '-'.join(
        unique([map_var_short_name.get(c.standard_name, c.standard_name)
                for c in cube_list])
    )
    loc_str = '-'.join(
        unique([map_var_short_name.get(c.attributes['location_name'],
                                       c.attributes['location_name'])
                for c in cube_list])
    )

    depth_str = '-'.join(unique([get_depth_sring(c) for c in cube_list]))

    if start_time is None and end_time is None:
        start_time, end_time = get_common_time_overlap(cube_list, time_extent)
    date_str = '_'.join(
        [d.strftime('%Y-%m-%d') for d in [start_time, end_time]])

    imgfile = '_'.join((prefix, loc_str, depth_str, var_str, date_str))
    imgfile += '.png'

    if root_dir is not None:
        create_directory(root_dir)
        imgfile = os.path.join(root_dir, imgfile)

    return imgfile


def plot_timeseries(ax, cube_list, label_attr='dataset_id', time_lim=None,
                    title=None, time_extent=None,
                    start_time=None, end_time=None, **kwargs):
    """
    Plots time series objects in the given axes.

    :arg cube_list: list of cube time series objects to plot. Can be any
        iterable container.
    """

    def get_label(cube, attr_name):
        return cube.attributes.get(attr_name)

    for c in cube_list:
        label = get_label(c, label_attr)
        if isinstance(c.data, numpy.ma.MaskedArray) \
                and numpy.all(c.data.mask):
            # if all data is bad, skip
            continue
        qplt.plot(c, axes=ax, label=label, **kwargs)
    if start_time is None and end_time is None and time_extent is not None:
        start_time, end_time = get_common_time_overlap(cube_list, time_extent)
    ax.set_xlim(start_time, end_time)
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    if title is None:
        loc_names = [c.attributes['location_name'] for c in cube_list]
        dep_strs = ['{:.1f} m'.format(
            c.coord('depth').points.mean()) for c in cube_list]
        titles = unique([' '.join(a) for a in zip(loc_names, dep_strs)])
        title = ', '.join(titles).strip()
        ax.set_title(title)
    if time_lim is not None:
        ax.set_xlim(time_lim)


def save_timeseries_figure(cube_list, output_dir=None, **kwargs):
    """
    Makes a default time series plot and saves it to disk.
    """
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    time_extent = kwargs.pop('time_extent', None)
    start_time = kwargs.pop('start_time', None)
    end_time = kwargs.pop('end_time', None)
    plot_timeseries(ax, cube_list, time_extent=time_extent,
                    start_time=start_time, end_time=end_time, **kwargs)

    imgfile = generate_img_filename(cube_list, root_dir=output_dir,
                                    start_time=start_time, end_time=end_time,
                                    time_extent=time_extent)

    print('Saving image {:}'.format(imgfile))
    fig.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
